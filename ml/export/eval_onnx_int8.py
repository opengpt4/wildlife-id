#!/usr/bin/env python3
"""INT8 vs fp32 accuracy on the identical eval set — G4, measured through ONNX
Runtime dynamic quantisation.

WHY NOT TFLite: the TFLite conversion toolchain (tensorflow + onnx2tf) could not
be made to work on this macOS host — importing onnx after tensorflow aborts the
process with a libc++ mutex error, and importing tensorflow after onnx deadlocks.
That is an *environment* blocker on the conversion path, not a statement about
the model. It is recorded in docs/phase0-results.md; conversion must be run in a
Linux container/CI before G4/G7 can be called closed against the shipped
artifact. This script measures the thing G4 is actually about: how much accuracy
INT8 costs on this eval set, with everything else identical.

Runs in the torch venv (onnxruntime + PIL + numpy; no tensorflow needed).

Usage:
  .venv/bin/python ml/export/eval_onnx_int8.py --label fp32 --onnx <fp32.onnx> --out ...
  .venv/bin/python ml/export/eval_onnx_int8.py --label int8 --quantize --out ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_tflite import load_image, metrics  # noqa: E402  (no TF import at module level)


def prompt_integrity(artifact_dir: str):
    with open(os.path.join(artifact_dir, "prompts.json")) as f:
        doc = json.load(f)
    with np.load(os.path.join(artifact_dir, "text_embeddings_200.npz")) as z:
        text = z["embeddings"].astype(np.float32)
        names = [str(s) for s in z["scientific_name"]]
        npz_prompts = [str(s) for s in z["prompt"]]
    rebuilt = [doc["template"].format(name=n) for n in names]
    h = hashlib.sha256("\n".join(rebuilt).encode()).hexdigest()
    if h != doc["prompt_list_sha256"] or npz_prompts != rebuilt:
        raise SystemExit("PROMPT INTEGRITY FAILURE — refusing to score")
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    print(f"prompt integrity OK: {len(rebuilt)} prompts, sha256 {h[:16]}…, "
          f"template {doc['template']!r}", file=sys.stderr)
    return text, names, h, doc["template"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--onnx", default="data/eval/artifacts/vision_tower_224.onnx")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--manifest", default="data/eval/inat/manifest.csv")
    ap.add_argument("--images-root", default="data/eval/inat")
    ap.add_argument("--artifact-dir", default="data/eval/artifacts")
    ap.add_argument("--candidates", default="data/species/hk_candidates_200.csv")
    ap.add_argument("--checklist", default="data/species/hk_checklist.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import onnxruntime as ort
    import csv

    text, _, prompt_hash, template = prompt_integrity(args.artifact_dir)

    model_path = args.onnx
    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        model_path = os.path.join(args.artifact_dir, "vision_tower_224_int8_dynamic.onnx")
        t0 = time.time()
        quantize_dynamic(args.onnx, model_path, weight_type=QuantType.QInt8,
                         per_channel=True, reduce_range=False)
        print(f"quantized in {time.time()-t0:.0f}s -> {model_path}", file=sys.stderr)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_shape = sess.get_inputs()[0].shape
    print(f"input {in_name} {in_shape}", file=sys.stderr)

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    with open(args.candidates) as f:
        cand = list(csv.DictReader(f))
    with open(args.checklist) as f:
        checklist = {r["scientific_name"]: int(r["observations"]) for r in csv.DictReader(f)}
    index = {s["scientific_name"]: i for i, s in enumerate(cand)}
    eval_names = sorted({r["scientific_name"] for r in rows})
    common20 = set(sorted(eval_names, key=lambda n: -checklist.get(n, 0))[:20])

    feats, kept, lat = [], [], []
    for row in rows:
        path = os.path.join(args.images_root, row["file"])
        x = load_image(path, 224)
        xin = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
        t0 = time.perf_counter()
        y = sess.run(None, {in_name: xin})[0].reshape(-1)
        lat.append((time.perf_counter() - t0) * 1000)
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
    common = metrics(sim[np.array([r["scientific_name"] in common20 for r in kept])],
                     y_idx[np.array([r["scientific_name"] in common20 for r in kept])])

    blob = open(model_path, "rb").read()
    result = {
        "config": {
            "label": args.label, "model": os.path.abspath(model_path),
            "model_sha256": hashlib.sha256(blob).hexdigest(),
            "size_mb": round(len(blob) / 1e6, 1),
            "quantisation": "onnxruntime dynamic int8 (per-channel weights)" if args.quantize else "fp32",
            "n_images": len(kept), "prompt_list_sha256": prompt_hash,
            "prompt_template": template,
            "manifest_sha256": hashlib.sha256(open(args.manifest, "rb").read()).hexdigest(),
            "host_ms_p50": round(float(np.percentile(lat, 50)), 1),
            "host_note": "host CPU ms is NOT gate G5",
        },
        "overall": overall, "by_group": by_group, "common20": common,
        "predictions": [
            {"file": r["file"], "true": r["scientific_name"],
             "top1": cand[int(sim[i].argmax())]["scientific_name"],
             "correct_top1": bool(sim[i].argmax() == y_idx[i])}
            for i, r in enumerate(kept)
        ],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.label}] top-1 {overall['top1']}% {overall['top1_ci95']}  "
          f"top-5 {overall['top5']}%  common20 {common.get('top1',0)}%  "
          f"size {result['config']['size_mb']} MB  host {result['config']['host_ms_p50']} ms",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
