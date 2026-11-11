#!/usr/bin/env python3
"""Turn the owner's field-photo folder into a scored manifest.

Reads `owner_metadata.csv` (the single source of truth for labels), converts
HEIC via the macOS `sips` tool so no extra dependency is needed, and writes
`owner_manifest.csv` in the same shape as the iNat slice so `score.py` can read
it unchanged.

Photos whose species is not in the 200-species candidate list are **excluded from
the closed-set metric and reported separately** — a species the app cannot name
at all is a coverage limitation, not a model error, and mixing the two would
distort both numbers.

Usage:
  python3 ml/eval/prepare_owner_slice.py --dir data/eval/owner
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

MANIFEST_COLUMNS = [
    "split", "obs_id", "photo_id", "taxon_id", "scientific_name", "common_name",
    "taxon_group", "license_code", "attribution", "observer", "observed_on",
    "latitude", "longitude", "photo_url", "file",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".dng", ".tif", ".tiff"}


def load_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def convert_heic(src: str, out_dir: str) -> str:
    """macOS `sips` conversion: HEIC -> full-quality JPEG. No pip dependency."""
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.splitext(os.path.basename(src))[0] + ".jpg")
    if os.path.exists(dst):
        return dst
    fmt = ["-s", "format", "jpeg", "-s", "formatOptions", "best"]
    subprocess.run(["sips", *fmt, src, "--out", dst],
                   check=True, capture_output=True)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/eval/owner")
    ap.add_argument("--candidates", default="data/species/hk_candidates_200.csv")
    ap.add_argument("--checklist", default="data/species/hk_checklist.csv")
    args = ap.parse_args()

    meta_path = os.path.join(args.dir, "owner_metadata.csv")
    if not os.path.exists(meta_path):
        print(f"no {meta_path} yet — see {args.dir}/README.md", file=sys.stderr)
        print("owner slice: 0 photos", file=sys.stderr)
        return 0

    meta = [r for r in load_csv(meta_path) if (r.get("file") or "").strip()]
    # The template ships with example rows; never let them into a score.
    meta = [r for r in meta if "delete me" not in (r.get("notes") or "").lower()]

    candidates = {r["scientific_name"]: r for r in load_csv(args.candidates)}
    checklist = {r["scientific_name"]: r for r in load_csv(args.checklist)}

    converted_dir = os.path.join(args.dir, "converted")
    rows, skipped, out_of_set = [], [], []
    for m in meta:
        name = m["file"].strip()
        src = os.path.join(args.dir, name)
        if not os.path.exists(src):
            skipped.append((name, "file missing"))
            continue
        sci = (m.get("scientific_name") or "").strip()
        if sci not in checklist:
            skipped.append((name, f"unknown scientific_name '{sci}' (not in HK checklist)"))
            continue
        if sci not in candidates:
            out_of_set.append((name, sci))
            continue
        rel = name
        if os.path.splitext(name)[1].lower() in {".heic", ".heif"}:
            try:
                rel = os.path.relpath(convert_heic(src, converted_dir), args.dir)
            except subprocess.CalledProcessError as e:
                skipped.append((name, f"HEIC conversion failed: {e.stderr[:120]!r}"))
                continue
        rows.append({
            "split": "owner",
            "obs_id": "", "photo_id": "",
            "taxon_id": checklist[sci]["taxon_id"],
            "scientific_name": sci,
            "common_name": (m.get("common_name") or checklist[sci]["common_name"]).strip(),
            "taxon_group": (m.get("taxon_group") or checklist[sci]["taxon_group"]).strip(),
            "license_code": "owner-licensed",
            "attribution": "owner",
            "observer": "owner",
            "observed_on": (m.get("observed_on") or "").strip(),
            "latitude": (m.get("latitude") or "").strip(),
            "longitude": (m.get("longitude") or "").strip(),
            "photo_url": "",
            "file": rel,
        })

    out_path = os.path.join(args.dir, "owner_manifest.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"owner slice: {len(rows)} photos -> {out_path}", file=sys.stderr)
    if out_of_set:
        print(f"  {len(out_of_set)} photo(s) of species outside the 200-candidate list "
              f"(excluded from closed-set scoring, reported separately): {out_of_set[:5]}",
              file=sys.stderr)
    for name, why in skipped:
        print(f"  ! skipped {name}: {why}", file=sys.stderr)
    by_species: dict[str, int] = {}
    for r in rows:
        by_species[r["scientific_name"]] = by_species.get(r["scientific_name"], 0) + 1
    print(f"  species covered: {len(by_species)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
