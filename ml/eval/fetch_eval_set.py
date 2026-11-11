#!/usr/bin/env python3
"""Phase 0 eval-set assembly from the iNaturalist public API.

Builds the reproducible "iNat slice" of the Phase 0 evaluation set:
  - a 200-species Hong Kong candidate list (the regional prior matrix), and
  - 150 research-grade photos, stratified over the 50 commonest HK
    butterfly + bird species, 3 distinct observations per species
    (distinct observers where possible, to limit per-observer bias).

No credentials required. Respects the 60 req/min anonymous limit.
Idempotent and resumable: existing files / manifest rows are kept.

Usage:
  python3 fetch_eval_set.py --out data/eval/inat
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from typing import Any

import requests

API = "https://api.inaturalist.org/v1"
PLACE_ID = 7613  # Hong Kong (place_type 12, admin)
TAXA = [("butterfly", 47224), ("bird", 3)]  # Papilionoidea, Aves
RATE_S = 1.15  # anonymous limit is 60 req/min
TIMEOUT = 45
UA = {"User-Agent": "wildlife-id-phase0-eval/0.1 (offline model evaluation; contact: owner)"}

N_CANDIDATES = 200  # regional candidate matrix size (charter: ~200)
N_EVAL_SPECIES = 50
PHOTOS_PER_SPECIES = 3
# Licenses acceptable for an internal benchmark set. Attribution is recorded
# per photo in the manifest; nothing here is redistributed.
PHOTO_LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"

class TerminalHTTPError(RuntimeError):
    """4xx that retrying cannot fix."""


_session = requests.Session()
_session.headers.update(UA)


def get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET with rate limiting and backoff on 429/5xx. 4xx (except 429) is terminal."""
    url = f"{API}/{path.lstrip('/')}"
    for attempt in range(5):
        try:
            r = _session.get(url, params=params, timeout=TIMEOUT)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise TerminalHTTPError(f"HTTP {r.status_code} for {url} {params}")
            if r.status_code == 429 or r.status_code >= 500:
                wait = 5 * (attempt + 1)
                print(f"  ! HTTP {r.status_code}, retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            r.raise_for_status()
            time.sleep(RATE_S)
            return r.json()
        except TerminalHTTPError:
            raise
        except requests.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"  ! {type(e).__name__}: {e}; retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url} {params}")


def species_counts(taxon_id: int) -> list[dict[str, Any]]:
    """All HK research-grade species for a taxon, most-observed first."""
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        d = get(
            "observations/species_counts",
            {
                "place_id": PLACE_ID,
                "taxon_id": taxon_id,
                "quality_grade": "research",
                "per_page": 500,
                "page": page,
            },
        )
        rows = d.get("results", [])
        out.extend(rows)
        if len(rows) < 500 or page >= 10:
            break
        page += 1
    return out


def pick_observations(taxon_id: int, want: int, max_month_attempts: int = 6) -> list[dict[str, Any]]:
    """Up to `want` research-grade observations spread across the calendar year.

    Sampling the *most recent* observations silently concentrates the whole eval
    set into the weeks before the run (first attempt: 149 photos, every one taken
    in the previous 21 days). Hong Kong butterflies and birds are strongly
    seasonal, so that would measure one season and make the month-prior
    experiment impossible. Instead: one observation per distinct month, months
    drawn in a seed-stable shuffled order, distinct observers where possible.
    """
    rng = random.Random(taxon_id)  # deterministic: the set is reproducible
    months = list(range(1, 13))
    rng.shuffle(months)
    chosen: list[dict[str, Any]] = []
    observers: set[str] = set()
    for month in months[:max_month_attempts]:
        if len(chosen) >= want:
            break
        d = get(
            "observations",
            {
                "place_id": PLACE_ID,
                "taxon_id": taxon_id,
                "quality_grade": "research",
                "photo_license": PHOTO_LICENSES,
                "month": month,
                "per_page": 30,
                "order_by": "observed_on",
                "order": "desc",
                "photos": "true",
            },
        )
        for obs in d.get("results", []):
            taxon = obs.get("taxon") or {}
            if taxon.get("id") != taxon_id or taxon.get("rank") != "species":
                continue  # exact species only
            if not [p for p in (obs.get("photos") or []) if p.get("id")]:
                continue
            user = ((obs.get("user") or {}).get("login") or "").lower()
            if user in observers:
                continue
            chosen.append(obs)
            observers.add(user)
            break  # one per month
    return chosen


def photo_url(photo: dict[str, Any]) -> str | None:
    """Derive the `large` rendition (~1024 px) from the observation payload.

    Do NOT call /v1/photos/{id}: that endpoint 404s for photos hosted on
    inaturalist-open-data, which is a large share of licensed observations.
    The observation's photo `url` is the static S3 path with a size segment, so
    the derivative we want is a string substitution.
    """
    url = photo.get("url") or ""
    if not url:
        return None
    return re.sub(r"/(square|small|medium|large|original)\.(jpe?g|png)$", r"/large.\2", url,
                  flags=re.IGNORECASE)


def download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        return
    for attempt in range(3):
        try:
            with _session.get(url, timeout=TIMEOUT, stream=True) as r:
                r.raise_for_status()
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                os.replace(tmp, dest)
            return
        except requests.RequestException as e:
            print(f"  ! photo {url}: {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


MANIFEST_COLUMNS = [
    "split", "obs_id", "photo_id", "taxon_id", "scientific_name", "common_name",
    "taxon_group", "license_code", "attribution", "observer", "observed_on",
    "latitude", "longitude", "photo_url", "file",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/inat", help="output root for the iNat slice")
    ap.add_argument("--candidates", default="data/species/hk_candidates_200.csv")
    ap.add_argument("--checklist", default="data/species/hk_checklist.csv")
    ap.add_argument("--refresh-lists", action="store_true")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    # ---- 1. species lists -------------------------------------------------
    if args.refresh_lists or not os.path.exists(args.checklist):
        rows = []
        for group, taxon_id in TAXA:
            counts = species_counts(taxon_id)
            print(f"{group}: {len(counts)} HK research-grade species", file=sys.stderr)
            for c in counts:
                t = c.get("taxon") or {}
                if t.get("rank") != "species":
                    continue
                rows.append({
                    "taxon_group": group,
                    "taxon_id": t["id"],
                    "scientific_name": t.get("name", ""),
                    "common_name": t.get("preferred_common_name", ""),
                    "observations": c.get("count", 0),
                })
        rows.sort(key=lambda r: -r["observations"])
        os.makedirs(os.path.dirname(args.checklist) or ".", exist_ok=True)
        with open(args.checklist, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.checklist} ({len(rows)} species)", file=sys.stderr)

    with open(args.checklist) as f:
        checklist = list(csv.DictReader(f))

    # Regional candidate matrix: top-N by observation count, both groups kept
    # so neither taxon is crowded out of the prior.
    per_group = N_CANDIDATES // len(TAXA)
    candidates: list[dict[str, str]] = []
    for group, _ in TAXA:
        group_rows = [r for r in checklist if r["taxon_group"] == group][:per_group]
        candidates.extend(group_rows)
    os.makedirs(os.path.dirname(args.candidates) or ".", exist_ok=True)
    with open(args.candidates, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        w.writeheader()
        w.writerows(candidates)
    print(f"wrote {args.candidates} ({len(candidates)} candidates)", file=sys.stderr)

    # ---- 2. eval species = commonest 50, spread across groups --------------
    eval_species: list[dict[str, str]] = []
    for group, _ in TAXA:
        eval_species.extend([r for r in checklist if r["taxon_group"] == group][: N_EVAL_SPECIES // 2])
    print(f"eval species: {len(eval_species)}", file=sys.stderr)

    # ---- 3. observations + photos ----------------------------------------
    manifest_path = os.path.join(args.out, "manifest.csv")
    done: set[int] = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            done = {int(r["obs_id"]) for r in csv.DictReader(f)}

    new_rows: list[dict[str, Any]] = []
    for i, sp in enumerate(eval_species, 1):
        tid = int(sp["taxon_id"])
        try:
            obs_list = pick_observations(tid, PHOTOS_PER_SPECIES)
        except RuntimeError as e:
            print(f"  ! {sp['scientific_name']}: {e}", file=sys.stderr)
            continue
        got = 0
        for obs in obs_list:
            if obs["id"] in done:
                got += 1
                continue
            photo = obs["photos"][0]
            try:
                url = photo_url(photo)
                if not url:
                    continue
                dest = os.path.join(img_dir, f"{obs['id']}_{photo['id']}.jpg")
                download(url, dest)
            except RuntimeError as e:
                print(f"  ! {sp['scientific_name']}: {e}", file=sys.stderr)
                continue
            geo = obs.get("geojson") or {}
            coords = (geo.get("coordinates") or [None, None])
            new_rows.append({
                "split": "inat",
                "obs_id": obs["id"],
                "photo_id": photo["id"],
                "taxon_id": tid,
                "scientific_name": sp["scientific_name"],
                "common_name": sp["common_name"],
                "taxon_group": sp["taxon_group"],
                "license_code": photo.get("license_code") or obs.get("license_code") or "",
                "attribution": obs.get("attribution") or "",
                "observer": (obs.get("user") or {}).get("login", ""),
                "observed_on": obs.get("observed_on") or "",
                "latitude": coords[1],
                "longitude": coords[0],
                "photo_url": url,
                "file": os.path.relpath(dest, args.out),
            })
            got += 1
        print(f"[{i:2d}/{len(eval_species)}] {sp['scientific_name']}: {got} photos", file=sys.stderr)

        # flush manifest every species so an interrupt loses nothing
        write_header = not os.path.exists(manifest_path)
        with open(manifest_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
        new_rows = []

    with open(manifest_path) as f:
        total = sum(1 for _ in csv.DictReader(f))
    print(f"done: {total} manifest rows -> {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
