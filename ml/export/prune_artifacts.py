#!/usr/bin/env python3
"""Prune TFLite artifacts that no current manifest refers to.

Every CI run drops five artifacts (~780 MB) into `data/eval/artifacts/`, which is
gitignored but not free. Keeping the whole history is how a working directory
becomes 40 GB without anyone noticing.

Dry run by default; `--delete` removes. Anything the current
`tflite_manifest.json` names, plus the JSON/NPZ side files, is never touched.

Usage:
  python ml/export/prune_artifacts.py                 # report only
  python ml/export/prune_artifacts.py --delete        # remove the strays
  python ml/export/prune_artifacts.py --keep-all      # keep every .tflite
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/eval/artifacts")
    ap.add_argument("--manifest", default=None,
                    help="defaults to <dir>/tflite_manifest.json")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--keep-all", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"no such directory: {args.dir}", file=sys.stderr)
        return 1

    manifest_path = args.manifest or os.path.join(args.dir, "tflite_manifest.json")
    referenced: set[str] = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        for value in manifest.values():
            if isinstance(value, dict) and "path" in value:
                referenced.add(os.path.basename(value["path"]))
        print(f"manifest {manifest_path}: {len(referenced)} referenced artifacts")
    else:
        print(f"no manifest at {manifest_path} — nothing will be treated as current",
              file=sys.stderr)

    keep_suffixes = (".json", ".npz", ".csv", ".txt", ".md")
    stale, kept, freed = [], [], 0
    for name in sorted(os.listdir(args.dir)):
        path = os.path.join(args.dir, name)
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        if name.endswith(keep_suffixes):
            kept.append((name, size, "side file"))
        elif name.endswith(".tflite"):
            if args.keep_all or name in referenced:
                kept.append((name, size, "current"))
            else:
                stale.append((name, size))
                freed += size

    print("\nkept:")
    for name, size, why in kept:
        print(f"  {human(size):>10}  {name}  ({why})")
    if stale:
        print("\nstale:")
        for name, size in stale:
            print(f"  {human(size):>10}  {name}")
        print(f"\n{len(stale)} files, {human(freed)} reclaimable")
    else:
        print("\nnothing stale — every .tflite is referenced by the manifest")

    if args.delete and stale:
        for name, _ in stale:
            os.remove(os.path.join(args.dir, name))
        print(f"deleted {len(stale)} files, freed {human(freed)}")
    elif stale:
        print("dry run — pass --delete to remove them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
