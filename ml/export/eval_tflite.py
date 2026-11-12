#!/usr/bin/env python3
"""Score a TFLite artifact (fp32 or INT8) on the eval set — gate G4.

Same 147 photos, same centre-crop protocol, same text embeddings, same prompt
strings as the fp32 reference. The prompt list is re-derived from
`prompts.json.template` and its sha256 is asserted against the hash recorded at
export time: if the strings moved, G4 would be measuring a different model and
this script refuses to produce a number.

Runs under the TFLite venv (tensorflow), not the torch venv.

Usage:
  .venv-tflite/bin/python ml/export/eval_tflite.py \
      --tflite data/eval/artifacts/vision_tower_224_int8.tflite \
      --label int8 --out data/eval/results/tflite_int8.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from PIL import Image

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# ---------------------------------------------------------------- metrics ---
def wilson(hits: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, centre - half) * 100, 1), round(min(1.0, centre + half) * 100, 1)]


def metrics(sim: np.ndarray, y: np.ndarray) -> dict:
    n = sim.shape[0]
    if n == 0:
        return {"n": 0}
    top1 = sim.argmax(axis=1)
    k = min(5, sim.shape[1])
    top5 = np.argsort(-sim, axis=1)[:, :k]
    h1 = int((top1 == y).sum())
    h5 = int((top5 == y[:, None]).any(axis=1).sum())
    return {"n": int(n), "top1": round(100 * h1 / n, 2), "top1_ci95": wilson(h1, n),
            "top5": round(100 * h5 / n, 2), "top5_ci95": wilson(h5, n)}


# ----------------------------------------------------------- preprocessing ---
def load_image(path: str, size: int) -> np.ndarray:
    """PIL bicubic shorter-side resize + centre crop + CLIP normalise. HWC float32."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = size / min(w, h)
        im = im.resize((max(size, round(w * scale)), max(size, round(h * scale))),
                       Image.BICUBIC)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return ((arr - CLIP_MEAN) / CLIP_STD).astype(np.float32)


def quantize_input(x: np.ndarray, detail: dict) -> np.ndarray:
    """Apply the model's input quantisation, exactly as a device runtime would."""
    dtype = detail["dtype"]
    if dtype == np.float32:
        return x
    scale, zero = detail["quantization"]
    if scale == 0:
        raise RuntimeError("quantized input with zero scale")
    q = np.round(x / scale + zero)
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def dequantize_output(y: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype == np.float32:
        return y.astype(np.float32)
    scale, zero = detail["quantization"]
    return (y.astype(np.float32) - zero) * scale


# ------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--label", required=True, help="e.g. fp32 or int8")
    ap.add_argument("--manifest", default="data/eval/inat/manifest.csv")
    ap.add_argument("--images-root", default="data/eval/inat")
    ap.add_argument("--artifact-dir", default="data/eval/artifacts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", default="data/species/hk_candidates_200.csv")
    ap.add_argument("--checklist", default="data/species/hk_checklist.csv")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--save-sim", default=None,
                    help="write the 147x200 cosine matrix as .npz for offline re-ranking studies")
    args = ap.parse_args()

    import tensorflow as tf  # imported late so --help is cheap

    # ---- prompt-string integrity -------------------------------------------
    with open(os.path.join(args.artifact_dir, "prompts.json")) as f:
        prompts_doc = json.load(f)
    with np.load(os.path.join(args.artifact_dir, "text_embeddings_200.npz")) as z:
        text = z["embeddings"].astype(np.float32)
        names = [str(s) for s in z["scientific_name"]]
        npz_prompts = [str(s) for s in z["prompt"]]
    rebuilt = [prompts_doc["template"].format(name=n) for n in names]
    import hashlib
    rebuilt_hash = hashlib.sha256("\n".join(rebuilt).encode()).hexdigest()
    if rebuilt_hash != prompts_doc["prompt_list_sha256"]:
        print("PROMPT INTEGRITY FAILURE: rebuilt strings do not match export hash",
              file=sys.stderr)
        return 2
    if npz_prompts != rebuilt:
        print("PROMPT INTEGRITY FAILURE: npz prompt column differs from template",
              file=sys.stderr)
        return 2
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    print(f"prompt integrity OK ({len(rebuilt)} prompts, sha256 {rebuilt_hash[:16]}…, "
          f"template {prompts_doc['template']!r})", file=sys.stderr)

    # ---- data ---------------------------------------------------------------
    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    with open(args.candidates) as f:
        cand = list(csv.DictReader(f))
    with open(args.checklist) as f:
        checklist = {r["scientific_name"]: int(r["observations"]) for r in csv.DictReader(f)}
    index = {s["scientific_name"]: i for i, s in enumerate(cand)}
    eval_names = sorted({r["scientific_name"] for r in rows})
    common20 = set(sorted(eval_names, key=lambda n: -checklist.get(n, 0))[:20])

    # ---- model --------------------------------------------------------------
    interp = tf.lite.Interpreter(model_path=args.tflite, num_threads=4)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print(f"input {in_det['shape']} {in_det['dtype'].__name__} q={in_det['quantization']}, "
          f"output {out_det['shape']} {out_det['dtype'].__name__}", file=sys.stderr)

    feats, kept, lat = [], [], []
    for row in rows:
        path = os.path.join(args.images_root, row["file"])
        try:
            x = load_image(path, args.resolution)
        except Exception as e:  # noqa: BLE001
            print(f"  ! unreadable {path}: {e}", file=sys.stderr)
            continue
        # onnx2tf emits NHWC by default; accept whichever the artifact declares.
        if in_det["shape"][-1] == 3 and in_det["shape"][1] != 3:
            xin = x[None, ...]                       # 1,H,W,3
        else:
            xin = np.transpose(x, (2, 0, 1))[None, ...]  # 1,3,H,W
        xin = quantize_input(xin, in_det)
        interp.set_tensor(in_det["index"], xin)
        t0 = time.perf_counter()
        interp.invoke()
        lat.append((time.perf_counter() - t0) * 1000)
        y = interp.get_tensor(out_det["index"])
        y = dequantize_output(y, out_det).reshape(-1)
        feats.append(y / (np.linalg.norm(y) + 1e-8))
        kept.append(row)

    feats = np.stack(feats)
    y_idx = np.array([index[r["scientific_name"]] for r in kept])
    sim = feats @ text.T

    overall = metrics(sim, y_idx)
    by_group = {}
    for group in sorted({r["taxon_group"] for r in kept}):
        mask = np.array([r["taxon_group"] == group for r in kept])
        by_group[group] = metrics(sim[mask], y_idx[mask])
    common_mask = np.array([r["scientific_name"] in common20 for r in kept])
    common = metrics(sim[common_mask], y_idx[common_mask])

    import hashlib as _h
    with open(args.tflite, "rb") as f:
        blob = f.read()
    result = {
        "config": {
            "label": args.label,
            "tflite": os.path.abspath(args.tflite),
            "tflite_sha256": _h.sha256(blob).hexdigest(),
            "tflite_size_mb": round(len(blob) / 1e6, 1),
            "resolution": args.resolution,
            "n_images": len(kept),
            "prompt_list_sha256": rebuilt_hash,
            "prompt_template": prompts_doc["template"],
            "manifest_sha256": _h.sha256(open(args.manifest, "rb").read()).hexdigest(),
            "host_invoke_ms_p50": round(float(np.percentile(lat, 50)), 1),
            "host_note": "host CPU ms is NOT gate G5; it only shows relative cost",
        },
        "overall": overall,
        "by_group": by_group,
        "common20": common,
        "predictions": [
            {"file": r["file"], "true": r["scientific_name"],
             "top1": cand[int(sim[i].argmax())]["scientific_name"],
             "correct_top1": bool(sim[i].argmax() == y_idx[i])}
            for i, r in enumerate(kept)
        ],
    }
    if args.save_sim:
        np.savez(args.save_sim, sim=sim, files=np.array([r["file"] for r in kept]),
                 y=y_idx, names=np.array([c["scientific_name"] for c in cand]),
                 months=np.array([r["observed_on"][5:7] for r in kept]))
        print(f"wrote {args.save_sim}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.label}] top-1 {overall['top1']}% {overall['top1_ci95']}  "
          f"top-5 {overall['top5']}%  common20 {common.get('top1', 0)}%  "
          f"size {result['config']['tflite_size_mb']} MB  "
          f"host {result['config']['host_invoke_ms_p50']} ms/img", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
