#!/usr/bin/env python3
"""Score the owner field slice — the half of the eval set that predicts the product.

`data/eval/owner/` has been waiting for 50 real phone photos since Phase 0 started.
This script is the reason that backlog is not also blocked on code: drop the photos
and `owner_metadata.csv` in, run this, get a number with error bars.

It deliberately reuses `eval_tflite.load_image` rather than reimplementing
preprocessing — a second resize/crop implementation is worth ~1.4 accuracy points
(`docs/phase0-results.md` §7), which would silently show up as a "field gap".

Usage:
  .venv-tflite/bin/python ml/eval/score_owner_slice.py \
      --tflite data/eval/artifacts/vision_tower_224_int8_dynamic.tflite \
      --label int8_dynamic --out data/eval/results/owner_int8_dynamic.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import eval_tflite as E  # noqa: E402 - single definition of preprocessing and metrics


def load_metadata(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("file") or "").strip()]


def register_heic() -> bool:
    """iPhone photos are usually HEIC. pillow-heif is optional, not required."""
    try:
        import pillow_heif  # noqa: F401
        pillow_heif.register_heif_opener()
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--metadata", default="data/eval/owner/owner_metadata.csv")
    ap.add_argument("--images-root", default="data/eval/owner")
    ap.add_argument("--artifact-dir", default="data/eval/artifacts")
    ap.add_argument("--candidates", default="data/species/hk_candidates_200.csv")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_metadata(args.metadata)
    if not rows:
        print(f"no labelled photos in {args.metadata}", file=sys.stderr)
        print("\nThe owner slice is still empty. Instructions:\n"
              "  data/eval/owner/README.md\n\n"
              "Copy `owner_metadata.csv.template` to `owner_metadata.csv`, drop the\n"
              "photos in this directory, and re-run. 20 photos scores; 50 is the target.",
              file=sys.stderr)
        return 2

    import tensorflow as tf

    # Same prompt-integrity gate as the iNat scorer: a moved prompt is a different model.
    with open(os.path.join(args.artifact_dir, "prompts.json")) as f:
        prompts_doc = json.load(f)
    with np.load(os.path.join(args.artifact_dir, "text_embeddings_200.npz")) as z:
        text = z["embeddings"].astype(np.float32)
        names = [str(s) for s in z["scientific_name"]]
    rebuilt = [prompts_doc["template"].format(name=n) for n in names]
    rebuilt_hash = hashlib.sha256("\n".join(rebuilt).encode()).hexdigest()
    if rebuilt_hash != prompts_doc["prompt_list_sha256"]:
        print("PROMPT INTEGRITY FAILURE: rebuilt strings do not match export hash",
              file=sys.stderr)
        return 2
    text /= np.linalg.norm(text, axis=1, keepdims=True)

    with open(args.candidates) as f:
        cand = list(csv.DictReader(f))
    index = {s["scientific_name"]: i for i, s in enumerate(cand)}

    heic = register_heic()
    interp = tf.lite.Interpreter(model_path=args.tflite, num_threads=4)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    feats, kept, skipped, out_of_set = [], [], [], []
    for row in rows:
        path = os.path.join(args.images_root, row["file"])
        label = row["scientific_name"].strip()
        if label not in index:
            # Not a miss to hide: the candidate set cannot return it at all.
            out_of_set.append(label)
            continue
        try:
            x = E.load_image(path, args.resolution)
        except Exception as e:  # noqa: BLE001
            hint = "" if heic else " (HEIC needs `pip install pillow-heif`)"
            skipped.append((row["file"], f"{type(e).__name__}: {e}{hint}"))
            continue
        xin = x[None, ...] if in_det["shape"][-1] == 3 else np.transpose(x, (2, 0, 1))[None, ...]
        interp.set_tensor(in_det["index"], E.quantize_input(xin, in_det))
        interp.invoke()
        y = E.dequantize_output(interp.get_tensor(out_det["index"]), out_det).reshape(-1)
        feats.append(y / (np.linalg.norm(y) + 1e-8))
        kept.append(row)

    if not feats:
        print("no photos could be scored — see the skip list", file=sys.stderr)
        for f, why in skipped:
            print(f"  ! {f}: {why}", file=sys.stderr)
        return 1

    sim = np.stack(feats) @ text.T
    y_idx = np.array([index[r["scientific_name"]] for r in kept])
    overall = E.metrics(sim, y_idx)

    by_group = {}
    for group in sorted({(r.get("taxon_group") or "unknown") for r in kept}):
        mask = np.array([(r.get("taxon_group") or "unknown") == group for r in kept])
        if mask.sum():
            by_group[group] = E.metrics(sim[mask], y_idx[mask])

    misses = [{"file": r["file"], "true": r["scientific_name"],
               "predicted": cand[int(sim[i].argmax())]["scientific_name"]}
              for i, r in enumerate(kept) if sim[i].argmax() != y_idx[i]]

    result = {
        "config": {
            "label": args.label,
            "tflite": os.path.abspath(args.tflite),
            "tflite_sha256": hashlib.sha256(open(args.tflite, "rb").read()).hexdigest(),
            "prompt_list_sha256": rebuilt_hash,
            "prompt_template": prompts_doc["template"],
            "n_scored": len(kept),
            "n_skipped": len(skipped),
            "n_out_of_candidate_set": len(out_of_set),
            "heic_support": heic,
        },
        "overall": overall,
        "by_group": by_group,
        "misses": misses,
        "skipped": [{"file": f, "reason": why} for f, why in skipped],
        "out_of_candidate_set": sorted(set(out_of_set)),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{args.label}] owner slice: top-1 {overall['top1']}% "
          f"{overall.get('top1_ci95')}  top-5 {overall['top5']}%  n={overall['n']}",
          file=sys.stderr)
    if out_of_set:
        print(f"  {len(out_of_set)} photos labelled with species outside the 200-set "
              f"(excluded, not counted as misses): {sorted(set(out_of_set))[:5]}",
              file=sys.stderr)
    if skipped:
        print(f"  {len(skipped)} unreadable photos, see {args.out}", file=sys.stderr)
    if overall["n"] < 20:
        print("  wide error bars below 20 photos — the CI is in the JSON", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
