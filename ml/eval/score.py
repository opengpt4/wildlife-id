#!/usr/bin/env python3
"""Phase 0 fp32 baseline: BioCLIP ViT-B/16 zero-shot on the HK eval set.

Answers, in one pass and with no training data (so no split leakage):
  - S2 arm 0: does 384 px beat 224 px?  (charter assumes 384)
  - S2 floor:  zero-shot top-1 / top-5, per group, per species, common-20 slice
  - S5:        regional 200-species candidate list vs the full HK checklist

Output is JSON (machine) plus a markdown table (human). Every number carries the
config that produced it: model id, resolution, prompt template, candidate set,
manifest hash.

Usage:
  .venv/bin/python ml/eval/score.py --manifest data/eval/inat/manifest.csv \
      --resolutions 224,384 --out data/eval/results/fp32_inat.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from PIL import Image

import open_clip

# open_clip 3.x: the hf-hub schema goes in the model-name slot; `pretrained` is
# ignored on that path. The repo's own config defines the architecture.
MODEL_NAME = "hf-hub:imageomics/bioclip"
ARCH = "ViT-B-16"
PROMPT_TEMPLATES = {
    # BioCLIP's paper finds short scientific-name prompts hard to beat; both are
    # cheap to try, so try both rather than argue.
    "name": "{name}",
    "photo": "a photo of {name}",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: str, split: str | None) -> list[dict[str, str]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r.get("split") == split]
    return rows


def read_species(path: str) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def build_text_matrix(model, tokenizer, species: list[dict[str, str]], template: str,
                      device: str) -> torch.Tensor:
    names = [template.format(name=s["scientific_name"]) for s in species]
    tokens = tokenizer(names)
    with torch.no_grad():
        feats = model.encode_text(tokens.to(device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu()


def encode_images(model, preprocess, rows: list[dict[str, str]], root: str,
                  device: str, batch_size: int = 16) -> tuple[torch.Tensor, list[int]]:
    """Returns (N x D L2-normalised embeddings, indices of rows that decoded)."""
    embs, kept = [], []
    batch = []
    with torch.no_grad():
        for idx, row in enumerate(rows):
            path = os.path.join(root, row["file"])
            try:
                with Image.open(path) as im:
                    img = preprocess(im.convert("RGB"))
            except Exception as e:  # noqa: BLE001 - corrupt download is a data fact
                print(f"  ! unreadable {path}: {e}", file=sys.stderr)
                continue
            batch.append((idx, img))
            if len(batch) == batch_size:
                embs.append(_flush(model, batch, device))
                kept.extend(i for i, _ in batch)
                batch = []
        if batch:
            embs.append(_flush(model, batch, device))
            kept.extend(i for i, _ in batch)
    return torch.cat(embs).float().cpu(), kept


def _flush(model, batch, device: str) -> torch.Tensor:
    imgs = torch.stack([img for _, img in batch]).to(device)
    feats = model.encode_image(imgs)
    return (feats / feats.norm(dim=-1, keepdim=True)).cpu()


def wilson(hits: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval, in percent. Honest error bars on 60 photos."""
    if n == 0:
        return [0.0, 0.0]
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, centre - half) * 100, 1), round(min(1.0, centre + half) * 100, 1)]


def metrics(sim: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    """sim: N images x K candidates, y: candidate index of the true label."""
    n = sim.shape[0]
    if n == 0:
        return {"n": 0}
    top1 = sim.argmax(dim=1)
    top5 = sim.topk(min(5, sim.shape[1]), dim=1).indices
    hit1 = (top1 == y)
    hit5 = (top5 == y.unsqueeze(1)).any(dim=1)
    h1, h5 = int(hit1.sum()), int(hit5.sum())
    return {
        "n": int(n),
        "top1": round(100 * h1 / n, 2),
        "top1_ci95": wilson(h1, n),
        "top5": round(100 * h5 / n, 2),
        "top5_ci95": wilson(h5, n),
    }


def per_species(sim: torch.Tensor, y: torch.Tensor, species_ids: list[int]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cid in sorted(set(int(v) for v in y)):
        mask = y == cid
        m = metrics(sim[mask], y[mask])
        out[str(species_ids[cid])] = m
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="eval manifest CSV")
    ap.add_argument("--images-root", default=None,
                    help="root that manifest `file` is relative to (default: manifest's dir)")
    ap.add_argument("--candidates", default="data/species/hk_candidates_200.csv")
    ap.add_argument("--checklist", default="data/species/hk_checklist.csv")
    ap.add_argument("--resolutions", default="224,384")
    ap.add_argument("--prompts", default="name,photo")
    ap.add_argument("--out", default="data/eval/results/fp32.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    resolutions = [int(r) for r in args.resolutions.split(",") if r.strip()]
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    root = args.images_root or os.path.dirname(os.path.abspath(args.manifest))
    rows = read_manifest(args.manifest, split=None)
    if not rows:
        print("empty manifest", file=sys.stderr)
        return 1
    print(f"eval rows: {len(rows)}  device: {device}  resolutions: {resolutions}", file=sys.stderr)

    candidates = read_species(args.candidates)
    checklist = read_species(args.checklist)
    # Candidate sets, each mapped scientific_name -> index.
    sets = {
        "regional200": candidates,
        "full_checklist": checklist,
    }
    # Diagnostic closed-world set: only the species that appear in the eval set.
    eval_names = sorted({r["scientific_name"] for r in rows})
    sets["eval_species_only"] = [s for s in checklist if s["scientific_name"] in set(eval_names)]
    for name, s in sets.items():
        missing = [r["scientific_name"] for r in rows
                   if r["scientific_name"] not in {x["scientific_name"] for x in s}]
        if missing:
            print(f"  ! {name} is missing eval labels: {sorted(set(missing))}", file=sys.stderr)

    # The 20 commonest species present in the eval set -> gate G2's slice.
    obs_rank = {s["scientific_name"]: int(s["observations"]) for s in checklist}
    common20 = set(sorted(eval_names, key=lambda n: -obs_rank.get(n, 0))[:20])
    print(f"common-20 subset: {len(common20)} species", file=sys.stderr)

    results: dict = {
        "config": {
            "model": ARCH,
            "pretrained": MODEL_NAME,
            "manifest": os.path.abspath(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "n_images": len(rows),
            "device": device,
            "torch": torch.__version__,
            "open_clip": open_clip.__version__,
            "candidate_sizes": {k: len(v) for k, v in sets.items()},
            "gate_reference": {
                "top1_overall": 70, "top1_common20": 85, "top5_overall": 90,
                "int8_penalty_pts": 2.0, "latency_p50_ms": 1000, "artifact_mb": 120,
            },
        },
        "runs": [],
        "paired": [],
    }
    correct: dict[tuple[int, str, str], list[bool]] = {}
    pred_index: dict[tuple[int, str, str], list[int]] = {}

    for res in resolutions:
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, force_image_size=res
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        t0 = time.time()
        img_feats, kept = encode_images(model, preprocess, rows, root, device, args.batch_size)
        img_s = time.time() - t0
        kept_rows = [rows[i] for i in kept]
        y_names = [r["scientific_name"] for r in kept_rows]
        print(f"[{res}px] encoded {len(kept)} images in {img_s:.1f}s "
              f"({1000 * img_s / max(1, len(kept)):.0f} ms/image, host CPU/MPS)", file=sys.stderr)

        for prompt in prompts:
            for set_name, species in sets.items():
                text = build_text_matrix(model, tokenizer, species, PROMPT_TEMPLATES[prompt], device)
                sim = img_feats @ text.T
                index = {s["scientific_name"]: i for i, s in enumerate(species)}
                y = torch.tensor([index[n] for n in y_names])
                overall = metrics(sim, y)
                by_group: dict[str, dict] = {}
                for group in sorted({r["taxon_group"] for r in kept_rows}):
                    mask = torch.tensor([r["taxon_group"] == group for r in kept_rows])
                    by_group[group] = metrics(sim[mask], y[mask])
                common_mask = torch.tensor([n in common20 for n in y_names])
                common = metrics(sim[common_mask], y[common_mask])
                pred1 = sim.argmax(dim=1)
                hit1 = (pred1 == y).tolist()
                hit5 = (sim.topk(min(5, sim.shape[1]), dim=1).indices == y.unsqueeze(1)).any(dim=1).tolist()
                run = {
                    "resolution": res,
                    "prompt": prompt,
                    "candidate_set": set_name,
                    "n_candidates": len(species),
                    "overall": overall,
                    "by_group": by_group,
                    "common20": common,
                    "per_species": per_species(sim, y, [index[n] for n in y_names]),
                    # Per-image predictions: needed for paired tests and for a
                    # confusion analysis that is not an average of averages.
                    "predictions": [
                        {"file": r["file"], "true": r["scientific_name"],
                         "top1": species[int(pred1[i])]["scientific_name"],
                         "correct_top1": bool(hit1[i]),
                         "correct_top5": bool(hit5[i])}
                        for i, r in enumerate(kept_rows)
                    ],
                }
                correct[(res, prompt, set_name)] = hit1
                pred_index[(res, prompt, set_name)] = [int(v) for v in pred1]
                results["runs"].append(run)
                print(f"[{res}px {prompt:5s} {set_name:17s} K={len(species):4d}] "
                      f"top1 {overall['top1']:5.1f}  top5 {overall['top5']:5.1f}  "
                      f"common20 top1 {common.get('top1', 0):5.1f}", file=sys.stderr)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    # Paired comparison between resolutions. A 2-point gap on 150 photos is
    # inside the noise, so the *paired* delta (same images, same texts) gets a
    # bootstrap CI. This is what decides whether 384 px earns its 2.9x compute.
    rng = np.random.default_rng(0)
    for prompt in prompts:
        for set_name in sets:
            for a, b in zip(resolutions, resolutions[1:]):
                ka, kb = (a, prompt, set_name), (b, prompt, set_name)
                if ka not in correct or kb not in correct:
                    continue
                ha = np.asarray(correct[ka], dtype=bool)
                hb = np.asarray(correct[kb], dtype=bool)
                n = len(ha)
                delta = (hb.mean() - ha.mean()) * 100
                idx = rng.integers(0, n, size=(4000, n))
                boot = (hb[idx].mean(axis=1) - ha[idx].mean(axis=1)) * 100
                lo, hi = np.percentile(boot, [2.5, 97.5])
                entry = {
                    "prompt": prompt, "candidate_set": set_name,
                    "res_a": a, "res_b": b,
                    "top1_a": round(float(ha.mean()) * 100, 2),
                    "top1_b": round(float(hb.mean()) * 100, 2),
                    "delta_top1": round(float(delta), 2),
                    "delta_ci95": [round(float(lo), 2), round(float(hi), 2)],
                    "flips_worse": int((ha & ~hb).sum()),
                    "flips_better": int((hb & ~ha).sum()),
                    "n": n,
                }
                results["paired"].append(entry)
                print(f"[paired {a}->{b} {prompt:5s} {set_name:17s}] "
                      f"{delta:+.1f} pts top-1 (95% CI {lo:+.1f}..{hi:+.1f}), "
                      f"{entry['flips_better']} fixed / {entry['flips_worse']} broken",
                      file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w") as f:
        f.write(to_markdown(results))
    print(f"wrote {args.out} and {md_path}", file=sys.stderr)
    return 0


def to_markdown(results: dict) -> str:
    c = results["config"]
    g = c["gate_reference"]
    lines = [
        "# Phase 0 fp32 baseline\n",
        f"- model: `{c['model']}` / `{c['pretrained']}` (fp32, {c['device']})",
        f"- manifest: `{os.path.basename(c['manifest'])}` sha256 `{c['manifest_sha256'][:16]}…`, "
        f"{c['n_images']} photos",
        f"- candidates: " + ", ".join(f"{k}={v}" for k, v in c["candidate_sizes"].items()),
        f"- gates: top1≥{g['top1_overall']}%, top1(common20)≥{g['top1_common20']}%, "
        f"top5≥{g['top5_overall']}%\n",
        "| px | prompt | candidate set | K | top-1 % (95% CI) | top-5 % | common-20 top-1 % | vs gate G1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results["runs"]:
        o, cm = r["overall"], r["common20"]
        ci = o.get("top1_ci95", [0, 0])
        delta = o["top1"] - g["top1_overall"]
        lines.append(
            f"| {r['resolution']} | {r['prompt']} | {r['candidate_set']} | {r['n_candidates']} | "
            f"{o['top1']} ({ci[0]}–{ci[1]}) | {o['top5']} | {cm.get('top1', 0)} | "
            f"{'PASS' if delta >= 0 else 'FAIL'} {delta:+.1f} |"
        )
    if results.get("paired"):
        lines += ["\n## Paired resolution comparison (same images, same text embeddings)\n",
                  "| prompt | candidate set | A→B | top-1 A | top-1 B | delta | 95% CI | fixed/broken |",
                  "|---|---|---|---|---|---|---|---|"]
        for e in results["paired"]:
            ci = e["delta_ci95"]
            verdict = "distinguishable" if (ci[0] > 0 or ci[1] < 0) else "**inside noise**"
            lines.append(
                f"| {e['prompt']} | {e['candidate_set']} | {e['res_a']}→{e['res_b']} | "
                f"{e['top1_a']} | {e['top1_b']} | {e['delta_top1']:+.1f} | "
                f"{ci[0]:+.1f}..{ci[1]:+.1f} | {e['flips_better']}/{e['flips_worse']} | {verdict} |"
            )
    lines.append("\nZero-shot, no training data: these are a floor, not the shipped model. "
                 "G1/G2/G3 apply to the INT8 artifact + head; G4 compares it to this fp32 reference.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
