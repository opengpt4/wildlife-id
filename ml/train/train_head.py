#!/usr/bin/env python3
"""Train the Route A regional head on frozen BioCLIP features, then measure G2.

Design (deliberate, and worth reading before changing anything):

- The head is a 200-way linear layer over the frozen 224 px features. Its rows are
  **initialised from the text embeddings** and only the 180 species with training
  photos are updated. The 20 species with no permissive training data therefore
  keep their zero-shot weights inside the same classifier: the "per-species
  routing" the owner asked for is expressed as frozen rows, not as a second model
  with an uncalibrated score scale.
- Split is **by observer**, never by image: a photographer's frames of one species
  are near-duplicates and would leak across a random split.
- The eval manifest hash is asserted before anything is scored.

Outputs: data/train/head.pt, data/train/train_report.json, and eval numbers for
the frozen 147-photo set.

Usage:
  .venv/bin/python ml/train/train_head.py --manifest data/train/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
import score as S  # noqa: E402 - identical model, transform and candidate list

FROZEN_EVAL_SHA256 = None  # filled from the fp32 results file, asserted below


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(hits: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, centre - half) * 100, 1), round(min(1.0, centre + half) * 100, 1)]


def metrics(sim: np.ndarray, y: np.ndarray) -> dict:
    n = len(y)
    top1 = sim.argmax(1)
    top5 = np.argsort(-sim, axis=1)[:, :5]
    h1 = int((top1 == y).sum())
    h5 = int((top5 == y[:, None]).any(1).sum())
    return {"n": int(n), "top1": round(100 * h1 / n, 2), "top1_ci95": wilson(h1, n),
            "top5": round(100 * h5 / n, 2), "top5_ci95": wilson(h5, n)}


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 4000) -> dict:
    """Paired difference b - a in percentage points, with bootstrap CI."""
    rng = np.random.default_rng(0)
    n = len(a)
    d = (b.mean() - a.mean()) * 100
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = (b[idx].mean(1) - a[idx].mean(1)) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta_pts": round(float(d), 2),
            "ci95": [round(float(lo), 2), round(float(hi), 2)],
            "fixed": int((~a & b).sum()), "broken": int((a & ~b).sum()), "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/train/manifest.csv")
    ap.add_argument("--train-root", default="data/train")
    ap.add_argument("--eval-manifest", default="data/eval/inat/manifest.csv")
    ap.add_argument("--eval-root", default="data/eval/inat")
    ap.add_argument("--fp32-results", default="data/eval/results/fp32_inat.json")
    ap.add_argument("--candidates", default="data/species/hk_candidates_200.csv")
    ap.add_argument("--checklist", default="data/species/hk_checklist.csv")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out-dir", default="data/train")
    ap.add_argument("--seed", type=int, default=0,
                    help="training RNG seed (batch order); observer split stays seed 0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    species = S.read_species(args.candidates)
    index = {s["scientific_name"]: i for i, s in enumerate(species)}
    n_classes = len(species)

    # ---- frozen eval guarantee ----------------------------------------------
    fp32 = json.load(open(args.fp32_results))
    frozen = fp32["config"]["manifest_sha256"]
    eval_sha = sha256_file(args.eval_manifest)
    assert eval_sha == frozen, f"EVAL MANIFEST CHANGED: {eval_sha[:12]} != frozen {frozen[:12]}"
    print(f"eval manifest frozen OK: {eval_sha[:16]}…", file=sys.stderr)

    # ---- data ---------------------------------------------------------------
    rows = [r for r in csv.DictReader(open(args.manifest))
            if os.path.exists(os.path.join(args.train_root, r["file"]))]
    eval_rows = list(csv.DictReader(open(args.eval_manifest)))
    eval_ids = {int(r["obs_id"]) for r in eval_rows if r["obs_id"]}
    eval_obs = {(r["observer"] or "").lower() for r in eval_rows}
    leaked_ids = [r for r in rows if int(r["obs_id"]) in eval_ids]
    leaked_obs = [r for r in rows if r["observer"].lower() in eval_obs]
    if leaked_ids or leaked_obs:
        raise SystemExit(f"LEAKAGE: {len(leaked_ids)} eval obs, {len(leaked_obs)} eval observers "
                         f"present in the training manifest — refusing to train")
    print(f"train photos {len(rows)}, species {len({r['scientific_name'] for r in rows})}, "
          f"observers {len({r['observer'] for r in rows})} — no eval leakage", file=sys.stderr)

    covered = sorted({r["scientific_name"] for r in rows})
    uncovered = [s["scientific_name"] for s in species if s["scientific_name"] not in set(covered)]
    print(f"covered species {len(covered)}; zero-shot-only species {len(uncovered)}",
          file=sys.stderr)

    model, _, preprocess = S.open_clip.create_model_and_transforms(
        S.MODEL_NAME, force_image_size=224)
    model = model.to(device).eval()

    def feats(rows_, root):
        f, kept = S.encode_images(model, preprocess, rows_, root, device)
        return f, kept

    # Feature cache: extraction is 26 ms/image and the hyper-parameter grid
    # re-reads the same images. Keyed on the manifest hash.
    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "features_cache.npz")
    tr_hash = sha256_file(args.manifest)
    tr = None
    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=True) as z:
            if str(z["manifest_sha256"]) == tr_hash:
                x_tr = torch.tensor(z["x"], dtype=torch.float32)
                tr = [dict(zip([str(c) for c in z["cols"]], [str(v) for v in row]))
                      for row in z["rows"]]
                print(f"loaded cached features for {len(tr)} photos", file=sys.stderr)
    if tr is None:
        print("extracting training features …", file=sys.stderr)
        x_tr, kept_tr = feats(rows, args.train_root)
        tr = [rows[i] for i in kept_tr]
        np.savez(cache_path, x=x_tr.numpy(),
                 files=np.array([r["file"] for r in tr]),
                 manifest_sha256=np.array(tr_hash),
                 cols=np.array(list(tr[0].keys())),
                 rows=np.array([list(r.values()) for r in tr]))
    y_tr = np.array([index[r["scientific_name"]] for r in tr])
    obs_tr = np.array([r["observer"].lower() for r in tr])

    # observer-level split
    rng = np.random.default_rng(0)
    observers = np.array(sorted(set(obs_tr)))
    rng.shuffle(observers)
    n_val = max(1, int(round(len(observers) * args.val_frac)))
    val_obs = set(observers[:n_val].tolist())
    val_mask = np.array([o in val_obs for o in obs_tr])
    train_mask = ~val_mask
    print(f"split by observer: {len(observers)-n_val} train observers "
          f"({int(train_mask.sum())} photos), {n_val} val observers "
          f"({int(val_mask.sum())} photos)", file=sys.stderr)

    # ---- classifier: text-embedding initialised, only covered rows trainable --
    with np.load("data/eval/artifacts/text_embeddings_200.npz") as z:
        names = [str(s) for s in z["scientific_name"]]
        text_np = z["embeddings"].astype(np.float32)
    assert names == [s["scientific_name"] for s in species], "candidate order mismatch"
    W0 = torch.tensor(text_np, dtype=torch.float32)          # (200, 512)
    covered_idx = torch.tensor([index[n] for n in covered], dtype=torch.long)
    W = torch.nn.Parameter(W0.clone())
    scale = torch.nn.Parameter(torch.tensor(20.0))
    trainable = torch.zeros(n_classes, dtype=torch.bool)
    trainable[covered_idx] = True
    trainable = trainable.unsqueeze(1)

    xt = torch.tensor(x_tr.numpy(), dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    opt = torch.optim.AdamW([W, scale], lr=args.lr, weight_decay=0.0)
    lossf = torch.nn.CrossEntropyLoss()

    xtr, ytr = xt[train_mask], yt[train_mask]
    xva, yva = xt[val_mask], yt[val_mask]

    best = {"val_top1": -1.0, "epoch": -1, "W": W.detach().clone(), "scale": float(scale)}
    for epoch in range(1, args.epochs + 1):
        W.grad = None
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 256):
            b = perm[i:i + 256]
            logits = (xtr[b] @ W.T) * scale
            loss = lossf(logits, ytr[b]) + args.wd * (W[covered_idx] ** 2).sum()
            loss.backward()
            with torch.no_grad():
                if W.grad is not None:
                    W.grad[~trainable.expand_as(W)] = 0.0
            opt.step()
            with torch.no_grad():  # frozen rows never move
                W[~trainable.expand_as(W)] = W0[~trainable.expand_as(W)]
        if epoch % 20 == 0 or epoch == args.epochs:
            with torch.no_grad():
                val_sim = (xva @ W.T) * scale
                vtop1 = float((val_sim.argmax(1) == yva).float().mean()) * 100
            if vtop1 > best["val_top1"]:
                best = {"val_top1": vtop1, "epoch": epoch,
                        "W": W.detach().clone(), "scale": float(scale)}
            print(f"  epoch {epoch:4d} val top-1 {vtop1:5.2f}%  (best {best['val_top1']:.2f})",
                  file=sys.stderr)

    Wf = best["W"].numpy()
    sf = best["scale"]

    # ---- zero-shot on validation, paired ------------------------------------
    def zeroshot_sim(x: torch.Tensor) -> np.ndarray:
        zs = torch.tensor(text_np, dtype=torch.float32)
        return (x @ zs.T).numpy() * 100  # scale irrelevant for top-k, kept positive

    zs_val = zeroshot_sim(xva)
    head_val = (xva @ best["W"].T).numpy() * sf
    val_report = {
        "zeroshot": metrics(zs_val, yva.numpy()),
        "head": metrics(head_val, yva.numpy()),
        "paired": paired(zs_val.argmax(1) == yva.numpy(), head_val.argmax(1) == yva.numpy()),
    }
    print(f"VAL zeroshot {val_report['zeroshot']['top1']}%  head {val_report['head']['top1']}%  "
          f"delta {val_report['paired']['delta_pts']:+.2f} {val_report['paired']['ci95']}",
          file=sys.stderr)

    # ---- evaluate on the frozen eval set ------------------------------------
    print("extracting eval features …", file=sys.stderr)
    x_ev, kept_ev = feats(eval_rows, args.eval_root)
    ev = [eval_rows[i] for i in kept_ev]
    y_ev = np.array([index[r["scientific_name"]] for r in ev])
    common20 = set(sorted({r["scientific_name"] for r in ev},
                          key=lambda n: -int(next(s["observations"] for s in
                                                  S.read_species(args.checklist)
                                                  if s["scientific_name"] == n)))[:20])

    head_sim = (x_ev @ best["W"].T).numpy() * sf
    zs_sim = zeroshot_sim(x_ev)
    head_rep = metrics(head_sim, y_ev)
    zs_rep = metrics(zs_sim, y_ev)

    cmask = np.array([r["scientific_name"] in common20 for r in ev])
    head_cm = metrics(head_sim[cmask], y_ev[cmask])
    zs_cm = metrics(zs_sim[cmask], y_ev[cmask])

    head_hit = head_sim.argmax(1) == y_ev
    zs_hit = zs_sim.argmax(1) == y_ev

    # Which eval species have no training data at all?
    uncovered_eval = sorted({r["scientific_name"] for r in ev} & set(uncovered))

    report = {
        "config": {
            "train_photos": int(len(tr)), "species_covered": len(covered),
            "species_zeroshot_only": len(uncovered),
            "train_observers": len(observers) - n_val, "val_observers": n_val,
            "best_epoch": best["epoch"], "scale": round(float(sf), 3),
            "eval_manifest_sha256": eval_sha, "frozen": True,
            "resolution": 224,
            "train_manifest_sha256": sha256_file(args.manifest),
            "preprocessing": "open_clip transform: shorter-side 224 bicubic, centre crop, CLIP mean/std",
            "licence": "CC0 + CC-BY only (Route A)",
        },
        "validation": val_report,
        "eval_zeroshot": zs_rep,
        "eval_head": head_rep,
        "eval_paired_full": paired(zs_hit, head_hit),
        "eval_common20_zeroshot": zs_cm,
        "eval_common20_head": head_cm,
        "eval_common20_paired": paired(zs_hit[cmask], head_hit[cmask]),
        "uncovered_eval_species": uncovered_eval,
        "gate_g2": {
            "target": 85.0,
            "measured": head_cm["top1"],
            "status": "PASS" if head_cm["top1"] >= 85.0 else "FAIL",
        },
        "gate_g1": {"target": 70.0, "measured": head_rep["top1"],
                    "status": "PASS" if head_rep["top1"] >= 70.0 else "FAIL"},
        "falsification_rule": "below 80% top-1 overall => head is not worth its complexity, revert to zero-shot",
        "prediction_sample": [
            {"file": r["file"], "true": r["scientific_name"],
             "top1": species[int(head_sim[i].argmax())]["scientific_name"],
             "correct": bool(head_hit[i])}
            for i, r in enumerate(ev[:20])
        ],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"W": best["W"], "scale": float(sf)}, os.path.join(args.out_dir, "head.pt"))
    with open(os.path.join(args.out_dir, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEVAL full: zeroshot {zs_rep['top1']}%  head {head_rep['top1']}%  "
          f"delta {report['eval_paired_full']['delta_pts']:+.2f} "
          f"{report['eval_paired_full']['ci95']}", file=sys.stderr)
    print(f"EVAL common-20: zeroshot {zs_cm['top1']}%  head {head_cm['top1']}%  "
          f"delta {report['eval_common20_paired']['delta_pts']:+.2f} "
          f"{report['eval_common20_paired']['ci95']}", file=sys.stderr)
    print(f"G2 {report['gate_g2']['status']} ({head_cm['top1']}% vs 85%)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
