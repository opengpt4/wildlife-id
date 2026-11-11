#!/usr/bin/env python3
"""Route A training set: commercially permissive only (CC0 + CC-BY), HK only.

Rules enforced here, not in the training script:
  - licence filter is `photo_license=cc0,cc-by` on the API call itself, so an
    NC or unlicensed photo cannot enter the set even by accident;
  - the 147 eval observations, every one of the 92 eval observers, and any
    observation within +/-3 days AND 500 m of an eval observation are excluded
    (near-duplicate session guard);
  - one photo per observation, distinct observers preferred, so 30 frames of one
    bird by one photographer cannot dominate a class;
  - per-photo attribution is written for every image: licence, creator, source
    URL. Route A is only defensible if we can prove what we trained on.

Outputs (data/train/):
  images/<obs>_<photo>.jpg
  manifest.csv       split-independent; observer column used for the split later
  attribution.csv    licence, creator, source URL per photo
  exclusion_log.json counts dropped by each rule

Usage:
  python3 ml/train/fetch_training_set.py --eval-manifest data/eval/inat/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

API = "https://api.inaturalist.org/v1"
PLACE_ID = 7613
RATE_S = 1.15
TIMEOUT = 60
UA = {"User-Agent": "wildlife-id-phase0-train/0.1 (offline model training; Route A permissive)"}
PERMISSIVE = "cc0,cc-by"
CAP_PER_SPECIES = 30
DAY_MS = 86_400_000
KM_DEG = 1 / 111.0  # 500 m in degrees, latitude

_session = requests.Session()
_session.headers.update(UA)


def get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{API}/{path.lstrip('/')}"
    for attempt in range(5):
        try:
            r = _session.get(url, params=params, timeout=TIMEOUT)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(f"HTTP {r.status_code} for {url}")
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            time.sleep(RATE_S)
            return r.json()
        except requests.RequestException as e:
            print(f"  ! {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def medium_url(photo: dict[str, Any]) -> str | None:
    url = photo.get("url") or ""
    if not url:
        return None
    return re.sub(r"/(square|small|medium|large|original)\.(jpe?g|png)$",
                  r"/medium.\2", url, flags=re.IGNORECASE)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import asin, cos, radians, sin, sqrt
    r = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-manifest", default="data/eval/inat/manifest.csv")
    ap.add_argument("--licence-mix", default="data/species/hk_candidate_licence_mix.json")
    ap.add_argument("--checklist", default="data/species/hk_checklist.csv")
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--min-permissive", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per-observer", type=int, default=1,
                    help="max photos from one observer per species")
    ap.add_argument("--pages", type=int, default=1,
                    help="observation result pages to scan per species")
    ap.add_argument("--topup", action="store_true",
                    help="only fetch species still below the per-species cap")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    # ---- exclusion sets ------------------------------------------------------
    eval_rows = list(csv.DictReader(open(args.eval_manifest)))
    eval_ids = {int(r["obs_id"]) for r in eval_rows if r["obs_id"]}
    eval_observers = {(r["observer"] or "").lower() for r in eval_rows}
    eval_sites = []
    for r in eval_rows:
        if r["latitude"] and r["longitude"] and r["observed_on"]:
            y, m, d = (int(x) for x in r["observed_on"].split("-")[:3])
            eval_sites.append({
                "species": r["scientific_name"],
                "lat": float(r["latitude"]), "lon": float(r["longitude"]),
                "date": __import__("datetime").date(y, m, d).toordinal(),
            })
    print(f"exclusions: {len(eval_ids)} obs, {len(eval_observers)} observers, "
          f"{len(eval_sites)} geo/date anchors", file=sys.stderr)

    mix = {r["scientific_name"]: r for r in json.load(open(args.licence_mix))}
    checklist = {r["scientific_name"]: r for r in csv.DictReader(open(args.checklist))}

    eligible = [(n, r) for n, r in mix.items() if r["permissive"] >= args.min_permissive]
    eligible.sort(key=lambda kv: -kv[1]["permissive"])
    print(f"eligible species (>= {args.min_permissive} permissive photos): {len(eligible)}",
          file=sys.stderr)

    if args.topup and os.path.exists(os.path.join(args.out, "manifest.csv")):
        have: dict[str, int] = {}
        for r in csv.DictReader(open(os.path.join(args.out, "manifest.csv"))):
            have[r["scientific_name"]] = have.get(r["scientific_name"], 0) + 1
        before = len(eligible)
        eligible = [(n, r) for n, r in eligible if have.get(n, 0) < CAP_PER_SPECIES]
        print(f"top-up mode: {len(eligible)}/{before} species below the {CAP_PER_SPECIES} cap",
              file=sys.stderr)

    manifest_path = os.path.join(args.out, "manifest.csv")
    attr_path = os.path.join(args.out, "attribution.csv")
    cols = ["obs_id", "photo_id", "taxon_id", "scientific_name", "common_name",
            "taxon_group", "observer", "observed_on", "latitude", "longitude",
            "license_code", "file", "url"]
    attr_cols = ["file", "scientific_name", "photo_id", "license_code", "creator",
                 "creator_name", "source_url", "attribution"]

    done_obs: set[int] = set()
    if os.path.exists(manifest_path):
        done_obs = {int(r["obs_id"]) for r in csv.DictReader(open(manifest_path))}

    dropped = {"eval_obs": 0, "eval_observer": 0, "near_duplicate": 0, "no_photo": 0}

    def guard_duplicate(species: str, lat: float, lon: float, day: int) -> bool:
        for s in eval_sites:
            if s["species"] != species:
                continue
            if abs(s["date"] - day) <= 3 and haversine_km(lat, lon, s["lat"], s["lon"]) <= 0.5:
                return True
        return False

    rows, attrs = [], []
    for i, (sci, rec) in enumerate(eligible, 1):
        tid = int(checklist[sci]["taxon_id"])
        results_all: list[dict[str, Any]] = []
        try:
            for page in range(1, args.pages + 1):
                d = get("observations", {
                    "place_id": PLACE_ID, "taxon_id": tid,
                    "quality_grade": "research", "photo_license": PERMISSIVE,
                    "per_page": 100, "page": page,
                    "order_by": "observed_on", "order": "desc",
                    "photos": "true",
                })
                results_all.extend(d.get("results", []))
                if len(d.get("results", [])) < 100:
                    break
        except RuntimeError as e:
            print(f"  ! {sci}: {e}", file=sys.stderr)
            continue

        picked = have_count = 0
        observer_counts: dict[str, int] = {}
        for obs in results_all:
            if picked >= CAP_PER_SPECIES:
                break
            taxon = obs.get("taxon") or {}
            if taxon.get("id") != tid or taxon.get("rank") != "species":
                continue
            photos = [p for p in (obs.get("photos") or []) if p.get("id")]
            if not photos:
                dropped["no_photo"] += 1
                continue
            if int(obs["id"]) in eval_ids:
                dropped["eval_obs"] += 1
                continue
            user = obs.get("user") or {}
            login = (user.get("login") or "").lower()
            if login in eval_observers:
                dropped["eval_observer"] += 1
                continue
            if observer_counts.get(login, 0) >= args.per_observer:
                continue
            geo = obs.get("geojson") or {}
            coords = geo.get("coordinates") or [None, None]
            lat, lon = coords[1], coords[0]
            on = obs.get("observed_on") or ""
            if lat is not None and lon is not None and on:
                y, m, dd = (int(x) for x in on.split("-")[:3])
                day = __import__("datetime").date(y, m, dd).toordinal()
                if guard_duplicate(sci, float(lat), float(lon), day):
                    dropped["near_duplicate"] += 1
                    continue
            if int(obs["id"]) in done_obs:
                picked += 1
                continue
            photo = photos[0]
            lic = (photo.get("license_code") or "").lower()
            if lic not in ("cc0", "cc-by"):
                continue  # belt and braces: the API filter should have caught this
            url = medium_url(photo)
            if not url:
                continue
            rows.append({
                "obs_id": obs["id"], "photo_id": photo["id"], "taxon_id": tid,
                "scientific_name": sci,
                "common_name": checklist[sci]["common_name"],
                "taxon_group": checklist[sci]["taxon_group"],
                "observer": login, "observed_on": on,
                "latitude": lat, "longitude": lon,
                "license_code": lic,
                "file": f"images/{obs['id']}_{photo['id']}.jpg",
                "url": url,
                "attribution": photo.get("attribution") or "",
                "creator_name": user.get("name") or "",
            })
            picked += 1
            observer_counts[login] = observer_counts.get(login, 0) + 1

        print(f"[{i:3d}/{len(eligible)}] {sci}: {picked} photos", file=sys.stderr)
        if rows:
            write_header = not os.path.exists(manifest_path)
            with open(manifest_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                w.writerows(rows)
            write_header = not os.path.exists(attr_path)
            with open(attr_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=attr_cols)
                if write_header:
                    w.writeheader()
                for r in rows:
                    w.writerow({
                        "file": r["file"], "scientific_name": r["scientific_name"],
                        "photo_id": r["photo_id"], "license_code": r["license_code"],
                        "creator": r["observer"], "creator_name": r["creator_name"],
                        "source_url": f"https://www.inaturalist.org/photos/{r['photo_id']}",
                        "attribution": r["attribution"],
                    })
            rows = []

    # ---- downloads -----------------------------------------------------------
    all_rows = list(csv.DictReader(open(manifest_path))) if os.path.exists(manifest_path) else []
    print(f"manifest: {len(all_rows)} photos; downloading to {img_dir}", file=sys.stderr)

    def fetch_one(row: dict[str, str]) -> tuple[str, str]:
        dest = os.path.join(args.out, row["file"])
        if os.path.exists(dest) and os.path.getsize(dest) > 8_000:
            return ("cached", row["file"])
        url = row.get("url") or ""
        if not url:
            return ("no_url", row["file"])
        try:
            with _session.get(url, timeout=TIMEOUT, stream=True) as r:
                r.raise_for_status()
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                os.replace(tmp, dest)
            return ("ok", row["file"])
        except requests.RequestException as e:
            return (f"fail:{type(e).__name__}", row["file"])

    stats: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for k, (status, _) in enumerate(pool.map(fetch_one, all_rows), 1):
            stats[status] = stats.get(status, 0) + 1
            if k % 250 == 0:
                print(f"  downloaded {k}/{len(all_rows)} {stats}", file=sys.stderr)
    print(f"download stats: {stats}", file=sys.stderr)

    with open(os.path.join(args.out, "exclusion_log.json"), "w") as f:
        json.dump({"dropped": dropped, "n_eval_obs": len(eval_ids),
                   "n_eval_observers": len(eval_observers),
                   "licence_filter": PERMISSIVE,
                   "cap_per_species": CAP_PER_SPECIES,
                   "species_eligible": len(eligible)}, f, indent=2)
    print(f"done: {len(all_rows)} photos, {len({r['scientific_name'] for r in all_rows})} species",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
