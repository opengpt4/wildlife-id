#!/usr/bin/env python3
"""Per-species month-of-year observation profile for the HK candidate set.

The month half of charter decision 3 ("GPS + month narrows the candidate list")
has never been measured — only the candidate-set *size* arm was (F3 in
`docs/phase0-results.md`). This fetches the twelve monthly buckets for each of the
200 candidate species once and caches them, so `ml/eval/month_prior.py` can re-rank
offline without touching the network again.

Leakage note that must travel with this data: iNat counts include the 147 eval
photos themselves. At ~3 photos per species out of hundreds to thousands of
observations the effect is negligible, but it is not zero and it biases *towards*
the month prior looking useful. Stated again in the results.

Usage:
  python ml/eval/fetch_month_profile.py            # first run, ~200 requests
  python ml/eval/fetch_month_profile.py --force    # refetch everything
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_eval_set as F  # noqa: E402 - reuse the rate-limited client


def profile(taxon_id: int) -> dict[str, int]:
    """Twelve buckets, month 1..12 -> research-grade HK observation count."""
    d = F.get(
        "observations/histogram",
        {
            "place_id": F.PLACE_ID,
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "interval": "month_of_year",
            "date_field": "observed_on",
        },
    )
    buckets = (d.get("results") or {}).get("month_of_year") or {}
    return {str(m): int(buckets.get(str(m), 0)) for m in range(1, 13)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/species/hk_candidates_200.csv")
    ap.add_argument("--cache", default="data/species/month_profile.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(args.candidates) as f:
        species = list(csv.DictReader(f))

    cache: dict = {"place_id": F.PLACE_ID, "interval": "month_of_year",
                   "source": "iNaturalist observations/histogram, research grade",
                   "species": {}}
    if os.path.exists(args.cache) and not args.force:
        with open(args.cache) as f:
            cache = json.load(f)

    todo = [s for s in species if s["scientific_name"] not in cache["species"]]
    print(f"{len(species)} candidates, {len(todo)} to fetch", file=sys.stderr)

    for i, s in enumerate(todo, 1):
        name = s["scientific_name"]
        try:
            months = profile(int(s["taxon_id"]))
        except Exception as e:  # noqa: BLE001
            print(f"  ! {name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        cache["species"][name] = {"taxon_id": int(s["taxon_id"]),
                                  "taxon_group": s["taxon_group"],
                                  "months": months}
        total = sum(months.values())
        print(f"  [{i}/{len(todo)}] {name:32s} total {total:6d}  "
              f"peak month {max(months, key=months.get)}", file=sys.stderr)
        if i % 10 == 0:
            os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
            with open(args.cache, "w") as f:
                json.dump(cache, f, indent=1)

    cache["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    with open(args.cache, "w") as f:
        json.dump(cache, f, indent=1)
    print(f"wrote {args.cache}: {len(cache['species'])} species", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
