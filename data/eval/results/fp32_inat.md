# Phase 0 fp32 baseline

- model: `ViT-B-16` / `hf-hub:imageomics/bioclip` (fp32, mps)
- manifest: `manifest.csv` sha256 `b029f55c7495fbf3…`, 147 photos
- candidates: regional200=200, full_checklist=828, eval_species_only=49
- gates: top1≥70%, top1(common20)≥85%, top5≥90%

| px | prompt | candidate set | K | top-1 % (95% CI) | top-5 % | common-20 top-1 % | vs gate G1 |
|---|---|---|---|---|---|---|---|
| 224 | name | regional200 | 200 | 53.06 (45.0–60.9) | 74.15 | 38.33 | FAIL -16.9 |
| 224 | name | full_checklist | 828 | 46.26 (38.4–54.3) | 70.07 | 30.0 | FAIL -23.7 |
| 224 | name | eval_species_only | 49 | 63.95 (55.9–71.3) | 84.35 | 51.67 | FAIL -6.0 |
| 224 | photo | regional200 | 200 | 77.55 (70.2–83.5) | 91.84 | 75.0 | PASS +7.5 |
| 224 | photo | full_checklist | 828 | 68.03 (60.1–75.0) | 87.07 | 63.33 | FAIL -2.0 |
| 224 | photo | eval_species_only | 49 | 86.39 (79.9–91.0) | 95.24 | 86.67 | PASS +16.4 |
| 384 | name | regional200 | 200 | 52.38 (44.3–60.3) | 70.75 | 41.67 | FAIL -17.6 |
| 384 | name | full_checklist | 828 | 43.54 (35.8–51.6) | 66.67 | 30.0 | FAIL -26.5 |
| 384 | name | eval_species_only | 49 | 63.95 (55.9–71.3) | 80.95 | 56.67 | FAIL -6.0 |
| 384 | photo | regional200 | 200 | 73.47 (65.8–79.9) | 93.2 | 70.0 | PASS +3.5 |
| 384 | photo | full_checklist | 828 | 63.27 (55.2–70.6) | 87.07 | 55.0 | FAIL -6.7 |
| 384 | photo | eval_species_only | 49 | 85.03 (78.4–89.9) | 95.24 | 86.67 | PASS +15.0 |

## Paired resolution comparison (same images, same text embeddings)

| prompt | candidate set | A→B | top-1 A | top-1 B | delta | 95% CI | fixed/broken |
|---|---|---|---|---|---|---|---|
| name | regional200 | 224→384 | 53.06 | 52.38 | -0.7 | -6.1..+4.8 | 8/9 | **inside noise** |
| name | full_checklist | 224→384 | 46.26 | 43.54 | -2.7 | -8.2..+3.4 | 7/11 | **inside noise** |
| name | eval_species_only | 224→384 | 63.95 | 63.95 | +0.0 | -5.4..+5.4 | 8/8 | **inside noise** |
| photo | regional200 | 224→384 | 77.55 | 73.47 | -4.1 | -10.2..+2.0 | 8/14 | **inside noise** |
| photo | full_checklist | 224→384 | 68.03 | 63.27 | -4.8 | -11.6..+1.4 | 9/16 | **inside noise** |
| photo | eval_species_only | 224→384 | 86.39 | 85.03 | -1.4 | -6.1..+2.7 | 4/6 | **inside noise** |

Zero-shot, no training data: these are a floor, not the shipped model. G1/G2/G3 apply to the INT8 artifact + head; G4 compares it to this fp32 reference.
