#!/usr/bin/env python3
"""Diagnostic: is the fp32 embedding space good enough that a linear head fixes G2?

This is NOT the shipped head and its number is NOT a Phase 0 gate result. It is a
ceiling probe: with the frozen BioCLIP features, how separable are the eval
species? If a linear probe cannot clear 85% on the common-20 slice, then the
head is not the fix for G2 and the problem is data or backbone, not the head.

Method: closed-form ridge regression on one-hot species targets, leave-one-out by
photo (147 solves of a 512x512 system — seconds). LOO is used because a head fit
and evaluated on the same 147 photos would be meaningless.

Caveat that must travel with the number: only the ~49 classes present in the eval
set have any training examples, so this probe is easier than the real 200-way head
trained on a real training set. It bounds *separability*, not achievable accuracy.

Usage:
  .venv/bin/python ml/eval/linear_probe.py --manifest data/eval/inat/manifest.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as S  # reuse loader / candidate sets / metric definitions


def fit_loo_ridge(x: np.ndarray, y_onehot: np.ndarray, lam: float):
    """LOO ridge, closed form. Returns (argmax preds, top-5 hit flags)."""
    n, d = x.shape
    a = x.T @ x
    b = x.T @ y_onehot
    reg = lam * np.eye(d)
    preds = np.empty(n, dtype=np.int64)
    top5 = np.zeros(n, dtype=bool)
    for i in range(n):
        ai = a - np.outer(x[i], x[i])
        bi = b - np.outer(x[i], y_onehot[i])
        scores = x[i] @ np.linalg.solve(ai + reg, bi)
        preds[i] = int(np.argmax(scores))
        top5[i] = preds[i] in np.argsort(-scores)[:5]
    return preds, top5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--candidates", default="data/species/hk_candidates_200.csv")
    ap.add_argument("--checklist", default="data/species/hk_checklist.csv")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--lambda-grid", default="0.1,1,10,100")
    ap.add_argument("--out", default="data/eval/results/linear_probe.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    root = os.path.dirname(os.path.abspath(args.manifest))
    rows = S.read_manifest(args.manifest, split=None)
    species = S.read_species(args.candidates)
    index = {s["scientific_name"]: i for i, s in enumerate(species)}
    checklist = S.read_species(args.checklist)
    eval_names = sorted({r["scientific_name"] for r in rows})
    common20 = set(sorted(eval_names, key=lambda n: -int(
        next(s["observations"] for s in checklist if s["scientific_name"] == n)))[:20])

    model, _, preprocess = S.open_clip.create_model_and_transforms(
        S.MODEL_NAME, force_image_size=args.resolution)
    model = model.to(device).eval()
    feats, kept = S.encode_images(model, preprocess, rows, root, device)
    x = feats.numpy()
    kept_rows = [rows[i] for i in kept]
    y_idx = np.array([index[r["scientific_name"]] for r in kept_rows])
    y_onehot = np.zeros((len(x), len(species)), dtype=np.float32)
    y_onehot[np.arange(len(x)), y_idx] = 1.0
    print(f"{len(x)} photos, {len(species)} candidates, {len(set(y_idx.tolist()))} populated classes",
          file=sys.stderr)

    results = {"config": {"model": S.ARCH, "resolution": args.resolution,
                          "n_photos": len(x), "n_candidates": len(species),
                          "method": "leave-one-out ridge, closed form",
                          "manifest_sha256": S.sha256(args.manifest)},
               "fits": []}
    for lam in [float(v) for v in args.lambda_grid.split(",")]:
        preds, top5 = fit_loo_ridge(x, y_onehot, lam)
        hit1 = preds == y_idx
        common = np.array([r["scientific_name"] in common20 for r in kept_rows])
        rec = {
            "lambda": lam,
            "top1": round(100 * float(hit1.mean()), 2),
            "top5": round(100 * float(top5.mean()), 2),
            "common20_top1": round(100 * float(hit1[common].mean()), 2),
            "common20_top5": round(100 * float(top5[common].mean()), 2),
            "n_common": int(common.sum()),
        }
        results["fits"].append(rec)
        print(f"lambda {lam:6.1f}: top1 {rec['top1']:5.1f}  top5 {rec['top5']:5.1f}  "
              f"common20 top1 {rec['common20_top1']:5.1f}", file=sys.stderr)

    best = max(results["fits"], key=lambda r: r["top1"])
    results["best"] = best
    results["note"] = ("Diagnostic ceiling only: LOO probe on 147 photos, ~49 of 200 "
                       "classes populated. Not a gate result.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
