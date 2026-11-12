#!/usr/bin/env python3
"""Measure the month arm of the regional prior — the half of decision 3 nobody built.

Charter decision 3 says "GPS + month + habitat narrows the candidate list before
ranking", and F3 in `docs/phase0-results.md` measured only the *size* arm (200 vs
828 species: +9.5 points). Whether a month filter on top of the 200-list buys
anything has never been measured, because it needs per-species seasonality.

This re-ranks the frozen fp32 similarity matrix under a month filter and reports the
paired delta against the unfiltered 200-list. It never touches the model, so it
cannot move G1/G3/G4 — the question is purely whether filtering helps or hurts.

Leakage, stated up front: the iNat monthly counts include the 147 eval photos.
~3 photos per species against hundreds to thousands of observations, so the bias is
small but it points *towards* the prior looking useful. Treat a single-digit gain
as an upper bound.

Usage:
  .venv-tflite/bin/python ml/eval/month_prior.py \
      --sim data/eval/results/sim_fp32.npz \
      --profile data/species/month_profile.json \
      --out data/eval/results/month_prior.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

THRESHOLDS = (0.0, 0.005, 0.01, 0.02)


def wilson(hits: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, centre - half) * 100, 1), round(min(1.0, centre + half) * 100, 1)]


def score(masked: np.ndarray, y: np.ndarray) -> dict:
    n = masked.shape[0]
    top1 = masked.argmax(axis=1)
    k = min(5, masked.shape[1])
    top5 = np.argsort(-masked, axis=1)[:, :k]
    h1 = int((top1 == y).sum())
    h5 = int((top5 == y[:, None]).any(axis=1).sum())
    return {"n": n, "top1": round(100 * h1 / n, 2), "top1_ci95": wilson(h1, n),
            "top5": round(100 * h5 / n, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="data/eval/results/sim_fp32.npz")
    ap.add_argument("--profile", default="data/species/month_profile.json")
    ap.add_argument("--checklist", default="data/species/hk_checklist.csv")
    ap.add_argument("--out", default="data/eval/results/month_prior.json")
    ap.add_argument("--bootstrap", type=int, default=4000)
    args = ap.parse_args()

    z = np.load(args.sim, allow_pickle=False)
    sim, y, months = z["sim"], z["y"], list(z["months"])
    names = [str(n) for n in z["names"]]
    n_img, n_cand = sim.shape

    with open(args.profile) as f:
        prof = json.load(f)["species"]

    # counts[c, m] — observations of candidate c in month m+1; unknown species all 0
    counts = np.zeros((n_cand, 12), dtype=float)
    unknown = []
    for c, name in enumerate(names):
        entry = prof.get(str(name))
        if not entry:
            unknown.append(str(name))
            continue
        for m in range(1, 13):
            counts[c, m - 1] = float(entry["months"].get(str(m), 0))
    totals = counts.sum(axis=1)

    with open(args.checklist) as f:
        import csv
        checklist = {r["scientific_name"]: int(r["observations"]) for r in csv.DictReader(f)}
    eval_names = sorted({names[int(i)] for i in y})
    common20 = sorted(eval_names, key=lambda nm: -checklist.get(nm, 0))[:20]
    common_idx = np.array([names.index(nm) for nm in common20])
    # Only photos whose true species is in the common-20 subset can be scored there.
    common_rows = np.isin(y, common_idx)
    common_y = np.array([list(common_idx).index(int(v)) for v in y[common_rows]])

    month_idx = np.array([int(m) - 1 for m in months])
    base = score(sim, y)
    base_common = score(sim[np.ix_(common_rows, common_idx)], common_y)

    results = {"baseline": {"all": base, "common20": base_common.get("top1")},
               "thresholds": {}, "unknown_species": unknown,
               "leakage_note": "iNat monthly counts include the 147 eval photos"}

    rng = np.random.default_rng(0)
    for t in THRESHOLDS:
        # plausible[c, m]: observed in month m at all, and above t of the year's
        # observations. Species with no profile data are never filtered out — absent
        # data is not evidence of absence.
        plausible = (counts > 0) & (counts >= t * np.maximum(totals, 1)[:, None])
        plausible[totals == 0] = True

        keep = plausible[np.arange(n_cand)[None, :], month_idx[:, None]]
        masked = np.where(keep, sim, -np.inf)
        # A row can be all -inf only if every candidate is implausible; the baseline
        # row is then unrestored so the score stays defined.
        dead = ~np.isfinite(masked).any(axis=1)
        masked[dead] = sim[dead]

        covered = keep[np.arange(n_img), y]
        res = score(masked, y)
        res_common = score(masked[np.ix_(common_rows, common_idx)], common_y)

        cur_top1 = (masked.argmax(axis=1) == y).astype(float)
        base_top1 = (sim.argmax(axis=1) == y).astype(float)
        idx = rng.integers(0, n_img, (args.bootstrap, n_img))
        boot = np.array([(cur_top1[i] - base_top1[i]).mean() * 100 for i in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5])

        results["thresholds"][f"{t:g}"] = {
            "relative_threshold": t,
            "mean_candidates": round(float(keep.sum(axis=1).mean()), 1),
            "min_candidates": int(keep.sum(axis=1).min()),
            "oracle_coverage": round(100 * float(covered.mean()), 1),
            "all": res,
            "common20_top1": res_common.get("top1"),
            "delta_top1": round((cur_top1.mean() - base_top1.mean()) * 100, 2),
            "delta_ci95": [round(lo, 2), round(hi, 2)],
            "broken": int(((base_top1 == 1) & (cur_top1 == 0)).sum()),
            "fixed": int(((base_top1 == 0) & (cur_top1 == 1)).sum()),
        }

    print(f"baseline (flat 200):        top-1 {base['top1']:5.2f}%  "
          f"common-20 {base_common.get('top1')}%  n={n_img}")
    for t, r in results["thresholds"].items():
        print(f"month >= {t:>5} of yearly: top-1 {r['all']['top1']:5.2f}%  "
              f"delta {r['delta_top1']:+6.2f} [{r['delta_ci95'][0]:+.2f},{r['delta_ci95'][1]:+.2f}]  "
              f"cands {r['mean_candidates']:5.1f}  coverage {r['oracle_coverage']:5.1f}%  "
              f"broke {r['broken']} fixed {r['fixed']}")
    if unknown:
        print(f"! {len(unknown)} candidates had no month profile: {unknown[:3]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
