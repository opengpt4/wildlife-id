# Phase 0 fp32 baseline

- model: `ViT-B-16` / `hf-hub:imageomics/bioclip` (fp32, mps)
- manifest: `manifest.csv` sha256 `66bc3387f2a12fff…`, 149 photos
- candidates: regional200=200, full_checklist=828, eval_species_only=50
- gates: top1≥70%, top1(common20)≥85%, top5≥90%

| px | prompt | candidate set | K | top-1 % (95% CI) | top-5 % | common-20 top-1 % | vs gate G1 |
|---|---|---|---|---|---|---|---|
| 224 | name | regional200 | 200 | 46.98 (39.1–55.0) | 73.15 | 38.98 | FAIL -23.0 |
| 224 | name | full_checklist | 828 | 40.27 (32.7–48.3) | 64.43 | 28.81 | FAIL -29.7 |
| 224 | name | eval_species_only | 50 | 55.7 (47.7–63.4) | 81.88 | 50.85 | FAIL -14.3 |
| 224 | photo | regional200 | 200 | 71.14 (63.4–77.8) | 91.28 | 66.1 | PASS +1.1 |
| 224 | photo | full_checklist | 828 | 62.42 (54.4–69.8) | 83.22 | 55.93 | FAIL -7.6 |
| 224 | photo | eval_species_only | 50 | 81.21 (74.2–86.7) | 96.64 | 79.66 | PASS +11.2 |
| 384 | name | regional200 | 200 | 51.68 (43.7–59.6) | 73.15 | 45.76 | FAIL -18.3 |
| 384 | name | full_checklist | 828 | 42.95 (35.3–51.0) | 68.46 | 35.59 | FAIL -27.0 |
| 384 | name | eval_species_only | 50 | 61.07 (53.1–68.5) | 85.91 | 59.32 | FAIL -8.9 |
| 384 | photo | regional200 | 200 | 73.15 (65.5–79.6) | 91.95 | 79.66 | PASS +3.2 |
| 384 | photo | full_checklist | 828 | 60.4 (52.4–67.9) | 88.59 | 61.02 | FAIL -9.6 |
| 384 | photo | eval_species_only | 50 | 82.55 (75.7–87.8) | 96.64 | 84.75 | PASS +12.5 |

Zero-shot, no training data: these are a floor, not the shipped model. G1/G2/G3 apply to the INT8 artifact + head; G4 compares it to this fp32 reference.
