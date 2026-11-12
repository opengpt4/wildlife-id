# Phase 0 — de-risking plan

**Status:** Phase 0 measured, blocked on hardware · **Owner:** Atlas ·
**Started:** 2026-11-11 · **Last updated:** 2026-11-12

> **This plan is the historical record of what Phase 0 set out to test.** Where it
> disagrees with reality, `docs/phase0-results.md` and `AGENTS.md` win. Two
> supersessions to read first: the shipped resolution is **224 px**, not 384
> (ADR 0001 §2), and the shipped quantisation is **dynamic-range INT8**, i.e. int8
> weights with fp32 activations, not full-integer INT8 (ADR 0001 §3a). §2's gate
> definitions still stand — including "INT8" being read as "the int8 artifact we
> ship".

**Latest (2026-11-12):** S3, S5 and the quantisation arm of S2 complete; the artifact
is built in CI and scored on real TFLite. G1 PASS (77.55% on the shipping artifact),
G3 PASS (91.16%), **G2 FAIL (73.3%)**, **G4 PASS (−1.36 pts)**, **G7 PASS (88.1 MB)**,
G5/G6 blocked on a device. Full-integer INT8 failed at chance and is not a viable
artifact. The month arm of the regional prior was measured and does not help (§3e).
Charter decision 2 amended to **224 px** (owner-approved 2026-11-11) and to
dynamic-range INT8 (2026-11-12); both are now written into `AGENTS.md`, so charter,
ADR and plan agree. Numbers in
`docs/phase0-results.md`. **Headline: G1 PASS (77.5%), G3 PASS (91.8%), G2 FAIL
(75.0%).** 384 px loses to 224 px (−4.1 pts). A Route A head was trained and
measured (3,903 photos): **G2 still FAIL (78.3%)**, no gain over zero-shot across
three seeds — the falsification rule fired, so the recommendation is to ship
zero-shot and carry 85% as an unmet v2 target (`docs/phase0-g2-training-data-proposal.md`
§10).
**Rule for this document:** every claim below is either a measurement (with a
number and a device) or is marked *estimate* / *assumption*. Nothing here is a
verdict until a number exists.

## 1. Why Phase 0 exists

The charter fixes the architecture (offline-first, BioCLIP vision tower + regional
linear head, TFLite INT8, 384 px, ~200-species regional candidate set) **before**
any of it is measured. Phase 0 exists to find out whether those choices survive
contact with real photos and a real phone. If a gate fails we change the
architecture now, not after the app is built.

**No app code and no feature work until the gates in §3 have numbers.**

## 2. Falsifiable gates (approved by the owner, 2026-11-11)

All gates apply to the *shipped* artifact: the int8 vision tower at **224 px**
(no head — it was tested and dropped, `docs/phase0-results.md` §8).
Green requires **all** of them.

| # | Metric | Gate | Measured on |
|---|--------|------|-------------|
| G1 | top-1 accuracy, 200-photo eval set | ≥ 70% | iNat slice + owner slice, reported separately |
| G2 | top-1 accuracy, 20 commonest HK species | ≥ 85% | 60-photo common subset |
| G3 | top-5 accuracy | ≥ 90% | full eval set |
| G4 | INT8 penalty vs fp32 | ≤ 2.0 points top-1 | identical eval set, identical preprocessing |
| G5 | latency, warm, mid-range Android (SD695 / Pixel 6a class, ≥4 GB) | ≤ 1000 ms p50 | on-device, GPU delegate and CPU reported separately |
| G6 | latency, iPhone (deferred to a later phase) | ≤ 500 ms p50 | — |
| G7 | model artifact size (vision tower + head, INT8) | ≤ 120 MB | exported file, not fp32 |

G6 is explicitly deferred: no Xcode and no Simulator on this machine, so iOS is
unmeasurable today. It is recorded as an accepted open risk, not silently dropped.

Results are reported as numbers with units, per device, with the exact artifact
name and its SHA-256. "Looking good" is not a result.

## 3. Environment constraints (recorded for reproducibility)

Phase 0 runs in a **sandboxed staging directory outside the git repository**,
`/Users/Keung/.dsh/im/staging/wildlife-id-phase0/`, which mirrors the project
layout (`docs/`, `ml/`, `data/`). Reason: the agent session that runs Phase 0 has
a file sandbox rooted at its own workspace and cannot write anywhere under
`/Users/Keung/wildlife-id`; sandbox escalation is unavailable to it. The owner
moves the finished artifacts into the repository at the end of Phase 0.

Consequences, all of which are also true of the future CI job:

- All paths in these documents are **relative to the project root**, so the move
  is a no-op for links and scripts.
- The Python venv lives at `.venv/` in the staging root (gitignored) and is
  **not** part of the deliverable; the reproducible part is
  `ml/eval/requirements.txt` plus the fetch and scoring scripts.
- No network credentials are used. The iNat API is accessed anonymously at
  ≤ 60 requests/min.
- No iOS device or simulator is available; no Android device is currently
  attached. G5/G6 are therefore blocked on hardware and run last.

## 4. Evaluation set

Two slices, never averaged together.

### 4.1 iNat slice — 150 photos (automated, reproducible)

Source: iNaturalist public API, `place_id=7613` (Hong Kong), `quality_grade=research`,
photo licence restricted to CC0 / CC-BY / CC-BY-NC / CC-BY-SA / CC-BY-NC-SA.
Built by `ml/eval/fetch_eval_set.py`; manifest at `data/eval/inat/manifest.csv`
records observation id, photo id, licence, attribution, observer, date and
coordinates for every image.

- **50 species × 3 photos**, chosen as the 50 commonest HK species by
  research-grade observation count, 25 butterflies + 25 birds. This guarantees
  that gate G2 (20 commonest species) has 60 photos behind it.
- **One photo per distinct calendar month**, months drawn in a seeded shuffled
  order, so the set spans seasons rather than the fortnight before the run. The
  first attempt (v0) sampled the 3 most recent observations per species and
  produced 149 photos all taken within the previous 21 days — that measured one
  November, not a year, and would have made the month-prior experiment
  impossible. v0's manifest is kept at `data/eval/results/manifest_v0_november.csv`
  as a record of the mistake.
- One *distinct observation* per photo, distinct observers preferred, one photo
  per observation. This blocks the classic leak where five frames of one bird by
  one photographer land on both sides of a split.
- The **common subset** for G2 is the 20 species with the largest research-grade
  counts in HK, scored as its own slice.
- Images are the iNat `large` rendition (~1024 px), which is what the app will
  have after its own downscale to 384 px.

### 4.2 Owner slice — 50 field photos (manual, the honest one)

`data/eval/owner/` — same protocol, scored identically and always reported
**separately**. It is a placeholder until the owner drops photos in; the README
in that directory specifies naming and metadata.

### 4.3 Known bias — stated now, not discovered later

**iNat research-grade photos are curated close-ups**: they are accepted by two
identifiers, usually cropped, well lit, shot with a macro or a long lens, and
often framed the way field guides frame them. The app will be handed a
one-handed phone shot of a butterfly three metres away, backlit, motion-blurred,
partly occluded by a leaf, in a 12 MP frame the subject occupies 5% of.

Therefore:

- The iNat slice measures **the model**, not the product.
- The owner slice is the **honest number** and the one that decides go/no-go on
  accuracy. G1/G2/G3 are reported for both; where they disagree, the owner slice
  wins and the gap is itself a Phase 0 finding (it is the "curation gap").
- A separate crop/distance stress test is a **Phase 1** item (subject-box
  degradation curve), not a Phase 0 gate — noted so it is not forgotten.

### 4.4 Inference protocol

Reset shorter side to 384, centre crop 384×384, bicubic, no augmentation, no
test-time augmentation. fp32 and INT8 use byte-identical preprocessed inputs.
Closed-set scoring over the 200-species regional candidate list
(`data/species/hk_candidates_200.csv`) unless a spike says otherwise; top-1 and
top-5 by cosine similarity against precomputed text embeddings.

## 5. The five spikes

Ordered by dependency and by what can run without hardware.

### S3 — iNat data availability (no device, no GPU)
*Question:* can a defensible HK butterfly + bird eval set and a 200-species prior
be assembled from public data, and how thick is coverage per species?
*Method:* fetch script above; report species counts per group, photos per species,
licence mix, observer diversity, date spread.
*Pass:* ≥ 50 species with ≥ 3 usable research-grade photos each; a checklist large
enough for the prior to have something to narrow (measured: **828** HK
research-grade species — 562 birds, 266 butterflies — so "≥ 700" is the real bar;
an earlier draft said 1,500 and was simply wrong about HK's checklist size).
*Output:* `data/eval/inat/manifest.csv`, `data/species/hk_checklist.csv`.
*Result — PASS:* 828 HK research-grade species (562 birds / 266 butterflies); 200
candidates, thinnest with 346 observations; 147 photos / 49 species / all 12
months / 92 observers / 138 CC-BY-NC + 7 CC-BY + 2 CC0. Details in
`docs/phase0-results.md` §1.

### S2 — quantization accuracy delta (no device)
*Question:* what does INT8 cost on this eval set?
*Method (staged so the plan cannot stall):*
0. **Resolution arm** — fp32 zero-shot at both 224 and 384 px. BioCLIP is trained
   at 224, so the charter's 384 is an untested interpolation costing 2.9× compute.
   If 384 does not beat 224 by ≥ 2 points top-1, Config B (224) is adopted and the
   charter is amended with the number behind it (see ADR 0001 §2).
1. **fp32 reference** — BioCLIP ViT-B/16, zero-shot against species-name text
   embeddings. No training data touches the eval set, so this number has no leak.
   It is a **floor**, not the product: a trained linear head can only help.
2. **INT8** — exported artifact scored on the identical manifest.
3. Linear head (logistic regression on frozen fp32 features) trained on iNat
   observations **disjoint from the eval observations** — only if the zero-shot
   floor is close enough to the gates that a head could plausibly close the gap.
*Pass:* G4 (≤ 2.0 points) and G1/G2/G3 at INT8.

*Partial result:* fp32 zero-shot at 224 px, K=200 → **77.5% top-1 [70.2–83.5],
91.8% top-5, 75.0% common-20**. G1 and G3 pass, **G2 fails by 10 points**, and a
LOO linear probe on three photos per species does not close it (61.9%) — the head
needs a real training set, so data acquisition is the critical path to G2.

*G4/G7 (measured 2026-11-11 on an ONNX Runtime INT8 proxy, 224 px):* INT8 costs
**−0.68 points top-1** (paired, CI −2.72…+1.36) and lands at **87.8 MB**, both
inside their gates — but on an artifact that quantises weights only. The TFLite
full-integer build is **blocked by the conversion toolchain on this macOS host**
(tensorflow + onnx cannot coexist in one process: import-order abort/deadlock).
G4/G7 are "passing on a proxy" and close only when the TFLite artifact is built in
a Linux container and re-scored (`ml/export/eval_tflite.py` is written for that).

### S5 — does the regional prior lift accuracy? (no device)
*Question:* does restricting to 200 regional species beat ranking over the full HK
checklist (~N species, text embeddings are cheap, no photos needed)?
*Method:* score the same 150 photos twice, once over 200 candidates, once over the
full checklist; the delta in top-1/top-5 is the prior's contribution. Month-based
narrowing is a second, cheaper arm (*estimate*: small; worth 5 lines of code).
*Pass (decision rule):* if the lift is **< 5 points top-1**, the GPS/month candidate
narrowing is not worth the product complexity in v1 — simplify to a static regional
list and say so. If ≥ 5 points, keep it and it becomes a spec.

*Result — PASS:* K=200 vs K=828 is **77.5% vs 68.0%, a +9.5 point lift**. The
regional prior stays. Caveat: this measures candidate-set size, not GPS or month
filtering; the month arm is still to be built, and the now-stratified eval set is
what makes it measurable.

### S1 — INT8 latency on target device (needs hardware)
*Question:* does the exported INT8 tower hit G5 on a mid-range Android?
*Method:* standalone benchmark harness — not the app — that loads the `.tflite`,
runs 5 warm-up + 30 timed passes on a fixed 384 px image, reports p50/p95, with
XNNPACK CPU and the GPU delegate measured separately, battery not charging, warm
device. Harness must run unchanged on any attached device.
*Pass:* p50 ≤ 1000 ms. Report both backends.

### S4 — RN camera + fast-tflite compatibility (needs hardware)
*Question:* do `react-native-vision-camera` and `react-native-fast-tflite` compose?
*Method:* a throwaway spike app; only if S1 passes, because it costs a device
build. Deferred to last deliberately.

## 6. Decision rules if a gate fails

- **G5 fails on CPU but passes on GPU delegate** → ship GPU-gated; 224 px is now
  the shipped resolution (ADR 0001 §2, owner-approved), so the fallback is the
  GPU delegate plus a distilled student, not a resolution drop.
- **G2 was funded, measured, and failed.** Route A head: 78.3% common-20 at the
  validation-selected configuration, 0–3 points over zero-shot, larger seed
  variance than effect. Recommendation: ship zero-shot, document the common-20 bar
  at 75%, keep 85% as an unmet v2 target. Do not re-open without materially more
  data or a different method (fine-tune the tower; two-stage classification).
- **G4 fails (INT8 too lossy)** → try full-integer per-channel with a 300-photo HK
  calibration set; then try fp16; then a distilled student model (Phase 1).
- **G1/G2 fail on zero-shot fp32** → the linear head is mandatory, not optional;
  G1/G2 then apply to head+INT8 and the training-set curation becomes the critical
  path.
- **G1 fails on the owner slice only** → the model is fine and the *capture UX* is
  the problem (crop guidance, distance prompt, burst capture); that is a finding
  worth more than the model numbers.
- **S5 lift < 5 points** → no GPS/month prior in v1.

## 7. Out of scope for Phase 0

App code, UI, server, sync, the linear head except as described, iOS builds,
species outside Hong Kong, and the VLM "why" text.

## 8. Open inputs (not blockers)

1. Owner's actual phone model and min Android version (G5 uses the SD695 class
   until known).
2. Owner slice photos (50) and the metadata in its README.
3. Whether observation history syncs across devices in v1 — affects the data
   model, not Phase 0.
4. Launch language set — I would default to Traditional Chinese + English because
   strings are cheap and HK is bilingual; revisit at Phase 1.
