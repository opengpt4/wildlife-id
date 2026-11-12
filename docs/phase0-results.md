# Phase 0 results — first numbers

**Date:** 2026-11-11; G4/G7 on real TFLite artifacts 2026-11-12 · **Author:** Atlas ·
**Status:** S3 complete, S2 measured to a verdict — **full-integer INT8 fails G4
(−78.23 pts, §3c) and dynamic-range INT8 passes it (−1.36 pts, §3d)**; G7 passes at
88.1 MB. S1/S4 blocked on hardware, and G5 is now the gate that decides shipping.

> **Charter note:** decision 2 (384 px) has been amended to 224 px with owner
> approval. The edit to `AGENTS.md` is pending the parent agent; until it lands,
> ADR 0001 §2 is authoritative for the shipped input resolution. This document
> does not treat the unamended charter as blocking.

Everything below is measured on this machine unless it says *estimate*. Raw data:
`data/eval/results/fp32_inat.json` (+ `.md`), `linear_probe.json`,
`manifest_v0_november.csv`.

## 1. What was actually run

| Item | Value |
|---|---|
| Model | `imageomics/bioclip`, open_clip `ViT-B-16`, 512-d embeddings, fp32 |
| Weights | `hf-hub:imageomics/bioclip`, cached locally; not committed |
| Eval set | 147 photos, 49 species, **all 12 calendar months**, 92 distinct observers, 2024-04 → 2026-11 |
| Licences | 138 CC-BY-NC, 7 CC-BY, 2 CC0 (attribution per photo in the manifest) |
| Candidate set | 200 species (100 butterflies + 100 birds), thinnest species 346 HK observations |
| Full checklist | **828** HK research-grade species (562 birds, 266 butterflies) |
| Protocol | shorter side → N, centre crop, bicubic, cosine vs text embeddings, no augmentation |
| Host | Apple Silicon, MPS, torch 2.5.1, open_clip 3.3.0 |
| Manifest sha256 | `fp32_inat.json → config.manifest_sha256` |

Two sets were built. The first (v0) sampled the most recent observations per
species and produced 149 photos **all taken in the previous 21 days** — one
November, not a year. It is kept as `data/eval/results/manifest_v0_november.csv`
because it produced a *different conclusion about resolution* (below), and that
is exactly the kind of error the eval set exists to catch.

## 2. Results — fp32 zero-shot

Closed-set, cosine similarity against precomputed text embeddings.

| px | prompt | candidate set | K | top-1 % [95% CI] | top-5 % | common-20 top-1 % |
|---|---|---|---|---|---|---|
| 224 | `a photo of {name}` | regional 200 | 200 | **77.5** [70.2–83.5] | 91.8 | 75.0 |
| 224 | `a photo of {name}` | full checklist | 828 | 68.0 [60.1–75.0] | 87.1 | 63.3 |
| 224 | `a photo of {name}` | eval species only | 49 | 86.4 [79.9–91.0] | 95.2 | 86.7 |
| 384 † | `a photo of {name}` | regional 200 | 200 | 73.5 [65.8–79.9] | 93.2 | 70.0 |
| 384 | `a photo of {name}` | full checklist | 828 | 63.3 [55.2–70.6] | 87.1 | 55.0 |
| 224 | `{name}` | regional 200 | 200 | 53.1 [45.0–60.9] | 74.2 | 38.3 |
| 384 | `{name}` | regional 200 | 200 | 52.4 [44.3–60.3] | 70.8 | 41.7 |

† 384 px is the **documented fallback**, kept with its numbers because the
resolution decision can be revisited (ADR 0001 §2). It is not the shipped
configuration as of this document.

By group at 224 / `a photo of` / K=200: butterflies **80.0%**, birds **75.0%** top-1
(n=75 and n=72 — not distinguishable from each other).

Paired comparison, same 147 images, same text embeddings, 4000-sample bootstrap:

| prompt | candidate set | 224 → 384 top-1 | delta | 95% CI | fixed / broken |
|---|---|---|---|---|---|
| `a photo of` | regional 200 | 77.5 → 73.5 | **−4.1** | −10.2 … +2.0 | 8 / 14 |
| `a photo of` | full checklist | 68.0 → 63.3 | −4.8 | −11.6 … +1.4 | 9 / 16 |
| `{name}` | regional 200 | 53.1 → 52.4 | −0.7 | −6.1 … +4.8 | 8 / 9 |

## 3. Gates

| Gate | Target | Measured | Status |
|---|---|---|---|
| G1 top-1 overall | ≥ 70% | **77.5%** [70.2–83.5] | **PASS** (lower CI bound sits on the gate) |
| G2 top-1 common-20 | ≥ 85% | **75.0%** | **FAIL**, −10.0 pts |
| G3 top-5 | ≥ 90% | **91.8%** | **PASS** |
| G4 INT8 vs fp32 | ≤ 2.0 pts | **−1.36 pts** [−3.40…0.00] for dynamic-range INT8, 88.1 MB (§3d). Full-integer INT8 is at chance: −78.23 pts (§3c) | **PASS**, but only for the int8-*weights* artifact — full-integer INT8 is unusable |
| G5 latency p50 | ≤ 1000 ms | not measured | no Android device attached |
| G6 iPhone latency | ≤ 500 ms | deferred | no Xcode/Simulator on this machine |
| G7 artifact | ≤ 120 MB | **88.1 MB** dynamic-range INT8 (§3d, sha256 `6ecf8cb9…`) vs 344.9 MB fp32; float16 is 172.5 MB and breaches it | **PASS** — measured on the actual artifact |

These are fp32 **zero-shot** numbers over a closed 200-set — a floor, not the
product. G1/G3 are cleared by zero-shot alone; G2 is not.

## 3b. INT8 — G4 and G7 (measured on a proxy artifact)

**The TFLite conversion path is blocked on this host.** `tensorflow 2.18/2.21` +
`onnx2tf 1.28.8` cannot be imported together: importing `onnx` after `tensorflow`
aborts the process with `libc++abi: terminating due to uncaught exception of type
std::__1::system_error: mutex lock failed: Invalid argument`, and importing
`tensorflow` after `onnx` deadlocks (killed after 4 min). `ai-edge-torch` needs
TensorFlow + torch in one environment, which is the same collision. These are
environment failures on the conversion toolchain, not statements about the model:
the ONNX export is numerically identical to PyTorch (cosine 1.000000 on all five
sampled images).

So G4/G7 were measured through **ONNX Runtime dynamic INT8** (per-channel int8
weights, fp32 activations, `onnxruntime.quantization.quantize_dynamic`), at 224 px,
on the same 147 photos, the same PIL preprocessing, and the same prompt strings —
whose sha256 is asserted at load time by both scorer runs.

| Artifact | size | top-1 % [95% CI] | top-5 % | common-20 | host p50 |
|---|---|---|---|---|---|
| fp32 ONNX | 345.0 MB | 78.91 [71.6–84.7] | 92.52 | 76.67 | 20.3 ms |
| INT8 ONNX (dynamic) | **87.8 MB** | **78.23** [70.9–84.1] | 91.16 | 75.00 | 16.7 ms |

Paired, same 147 images: **−0.68 points top-1** (95% CI −2.72…+1.36); 2 photos
broken, 1 fixed. By group: butterflies unchanged (78.67 both), birds 79.17 → 77.78.

**Reading this honestly:** the accuracy cost of quantisation looks negligible and
the size lands at 88 MB, both inside the gates — but this quantises *weights*
only, with activations still fp32, which is the friendliest possible INT8. A
full-integer TFLite artifact also quantises activations and the input tensor, and
that is where ViT attention usually loses its points. **G4 and G7 are therefore
"passing on a proxy", not closed.** They close when the TFLite artifact is built in
a Linux container and re-scored with `ml/export/eval_tflite.py`, which is already
written to run that check and to refuse to score if the prompt strings moved.

One incidental data point: swapping the *preprocessing implementation* (open_clip's
torch transform → my PIL transform) moved top-1 from 77.55% to 78.91%, i.e. **1.4
points**. Implementation details are worth more than several of the effects we are
trying to measure, which is why every comparison here is paired.

## 3c. G4 and G7 on the real TFLite artifact — full-integer INT8 **FAILS**

Built in GitHub Actions (ubuntu 26.04, python 3.11, `tensorflow-cpu 2.21.0`,
`onnx2tf 1.28.8`, run 43124693365), full-integer INT8 — weights, activations and
input tensor — calibrated on 300 HK photos whose loaded value range was measured at
build time as **[-1.792, 2.146]**. Scored with `ml/export/eval_tflite.py` on the
same 147 photos, eval manifest sha256
`b029f55c7495fbf3…`; the prompt list sha256 is asserted at load, so the strings
cannot have moved between the fp32 reference and these runs.

| Artifact | size | top-1 % [95% CI] | top-5 % | common-20 % | sha256 |
|---|---|---|---|---|---|
| TFLite fp32 | 344.9 MB | **78.91** [71.6–84.7] | 92.52 | 76.67 | `4484462e64f5e897…` |
| TFLite INT8, int8 I/O | **88.6 MB** | **0.68** [0.1–3.8] | 4.76 | 0.00 | `97a8ebe9a0a91a7a…` |
| TFLite INT8, float I/O (control) | **88.6 MB** | **0.68** [0.1–3.8] | 4.76 | 0.00 | `76ef1f9dfda509f7…` |

Raw output: `data/eval/results/tflite_{fp32,int8,int8_floatio}.json`.

**G7 passes and is now closed.** 88.6 MB ≤ 120 MB, measured on the artifact that
would ship, not on a proxy.

**G4 fails against a ≤ 2.0-point budget: 0.68% versus 78.91%, i.e. −78.23 points.**
The model is at chance (1/200 = 0.5%).

**The fp32 artifact is the control that makes the failure attributable.** It
reproduces the ONNX fp32 proxy exactly — 78.91 / 92.52 / 76.67 on the same 147
photos, same preprocessing, same text embeddings, different runtime and file
format. So the conversion pipeline, the prompt strings and the preprocessing are
all doing what the reference does. What differs is quantisation, and nothing else.

**It is not the input boundary.** The third artifact keeps the same int8 weights
and activations with float input and output tensors, so no input clipping can
occur; it scores the same 0.68%, not better. The int8 graph itself is what loses
the model. (Its earlier build scored 0.00% — the converter is not deterministic
across builds, but every run of every variant is at chance, which is the point.)

**Where it loses.** Embedding cosine against fp32 on the same images is 0.10–0.31,
while two *different* fp32 images sit at 0.73 — the output is not collapsed to a
constant, it is noise. Early patch-embed activations still match fp32 at 0.98–0.99;
the error compounds through the blocks and is 0.137 by the last LayerNorm. That is
the failure ADR 0001 §5 predicted in advance: LayerNorm and attention are where a
ViT's per-tensor dynamic range exceeds what one int8 scale can carry.

**One quirk worth recording,** because it would silently wreck any model shipped
with int8 I/O: the converter quantised the input tensor to `[0.185, 4.12]`
(zero-point −12) even though the calibration arrays span `[-1.792, 2.146]`. That
range clips **59.4% of all pixel values** in the eval set, and every image has
values below its floor. It is not the cause of this failure — the float-I/O
control rules that out — but a pipeline that trusted the input range would ship a
model that is broken in a way no accuracy number would explain.

**Decision, per §8's rule and ADR 0001 §3.** This is recorded as a **TFLite-side
failure**: the artefact of a post-training full-integer conversion, not evidence
that BioCLIP cannot identify HK species, and not a reason to retrain the regional
head (which §8 already found worthless). The zero-shot configuration is unchanged.
The response is the ADR's pre-agreed ladder, in order: (a) selective or
per-channel quantisation with LayerNorm and softmax left in fp32; (b) float16
weights with int8 activations; (c) float16 throughout, ~172 MB, which **fails G7**
and therefore needs an owner decision; (d) a distilled student, Phase 1.

**Methodology lesson, stated plainly.** The dynamic-range proxy said −0.68 points
and the real artifact said −78.23. A proxy that quantises weights but not
activations is not a preview of full-integer INT8; it is a measurement of a
different experiment that happens to share a name. §3b's caveat was correct on the
substance and far too gentle on the magnitude.

## 3d. The quantisation ladder, measured — dynamic-range INT8 clears G4 and G7

The ADR's ladder assumed TFLite could keep LayerNorm and softmax in fp32 while
quantising everything else. **It cannot.** The converter exposes three modes —
dynamic range (int8 weights, float activations), float16, and full integer — and
no flag exempts a given op. So "selective INT8" is not reachable without QAT or
graph surgery, and the honest move was to measure the two modes that can ship.

Built in run 43132640820, same 300-photo calibration, same 147-photo eval set
(manifest sha256 `b029f55c…`). Paired against the fp32 TFLite artifact, 4000-sample
bootstrap over the same images:

| Artifact | mode | size | top-1 % | Δ vs fp32 [95% CI] | broken / fixed | G4 | G7 |
|---|---|---|---|---|---|---|---|
| fp32 | — | 344.9 MB | 78.91 | — | — | — | — |
| **INT8 dynamic** | int8 weights, fp32 activations | **88.1 MB** | **77.55** | **−1.36** [−3.40, 0.00] | 2 / 0 | **PASS** | **PASS** |
| float16 | fp16 weights, fp32 activations | 172.5 MB | 78.91 | +0.00 [+0.00, +0.00] | 0 / 0 | PASS | **FAIL** (172.5 > 120) |
| INT8 full-integer | int8 weights + activations | 88.6 MB | 0.68 | −78.23 [−84.35, −71.43] | 115 / 0 | FAIL | pass |
| INT8 full-integer, float I/O (control) | as above, float boundary | 88.6 MB | 0.68 | −78.23 [−84.35, −71.43] | 115 / 0 | FAIL | pass |

All five artifacts come from one build; raw output in
`data/eval/results/tflite_{fp32,int8_dynamic,fp16,int8,int8_floatio}.json`.

**What this settles.** Full-integer INT8 is not a viable artifact for this model
post-training, and no amount of retrying the converter will change that. Dynamic
range — int8 weights, float activations, 88.1 MB — costs **1.36 points**; its CI
(−3.40…0.00) does not exclude a loss worse than the 2.0-point budget, so this is a
pass with a thin margin, not a comfortable one. It **clears G4 and G7 on the
artifact that would ship**. float16 is numerically free (0.00 points,
byte-identical predictions) and 52 MB too large; that is a size decision for the
owner, not an accuracy one.

**What it costs.** Dynamic range keeps activations in fp32, so on device it trades
the memory traffic that full-integer INT8 saves. ADR 0001 §3 originally rejected it
for exactly that reason — a judgement made before the alternative was known to be
worthless. G5, not G4, is now the gate that decides whether this ships, and the
ADR's own latency arithmetic (1.2–2.9 s CPU at 224 px for *full* int8) suggests
the margin is thin. **G4 passing here is not evidence that G5 passes.**

**Reproducibility note.** The same commit produced int8 artifacts with different
bytes across builds (`97a8ebe9…` then `fec8fc14…`, same size, same accuracy class),
and the float-I/O control likewise (`d91e6807…`, `76ef1f9d…`). The conversion is not
bit-reproducible. Before a region pack becomes the versioned unit, the ONNX export
and the converter both need a determinism check, or packs must be identified by
score and not by hash alone.

## 4. Findings

**F1 — 384 px does not earn its 2.9× compute; 224 px wins on top-1.**
At K=200 the paired delta is −4.1 points top-1 in favour of 224, with 14 photos
broken and 8 fixed by going to 384. The CI includes zero, so this is "no evidence
384 helps, point estimate says it costs" — and the v0 November-only set said the
opposite (+2.1 for 384), which is why a single-season eval set cannot decide an
architecture. Top-5 does improve slightly at 384 (93.2 vs 91.8), i.e. 384 spreads
probability mass without improving the head of the ranking.
*Consequence:* ADR 0001 §2's pre-agreed rule fires — adopt 224, which also gives
the latency spike 2.9× of headroom against a budget already predicted to fail on
CPU. This contradicts charter decision 2 and needs the owner's sign-off.

**F2 — the prompt template is worth +24 points, bigger than any other lever here.**
`a photo of Papilio polytes` 77.5% vs `Papilio polytes` 53.1%. That is a larger
effect than resolution, and larger than the regional prior. The region pack must
ship the exact prompt template alongside the text embeddings; a pack generated
with bare binomials is a different, much worse model.

**F3 — the regional prior is worth +9.5 points and clears the S5 decision rule.**
K=200 vs K=828: 77.5% vs 68.0% top-1. Threshold was ≥ 5 points to keep GPS/month
candidate narrowing in v1; +9.5 means it stays. Note this measures *candidate set
size*, not GPS or month filtering — the month arm is not yet built, and the
month-stratified set is now in place to test it.

**F4 — G2 fails at zero-shot, and a head trained on three photos per species does
not fix it.** A leave-one-out ridge probe on the frozen 224 px features reaches
61.9% top-1 / 56.7% common-20 — *worse* than zero-shot, because 151 of the 200
classes have no training examples at all and three examples per class cannot fit a
200-way linear head. Diagnostic only, not a gate result (`linear_probe.json`).
*Consequence:* the linear head is not a cheap patch; it requires a real training
set (thousands of HK observations, ~20–50 per species) and that data acquisition
is now the critical path for G2.

**F5 — the "curated close-up" bias is real but overstated.** Spot-checking the
misclassifications found genuine field photos inside the iNat slice: a butterfly
occupying ~5% of a cluttered foliage frame, and another shot into a backlit tree
canopy where the subject is a few pixels. Research-grade means "two identifiers
agreed", not "well framed". So this slice is closer to the product's real input
than the plan assumed — the owner slice is still **pending** (the folder exists,
the photos do not), so the honest field-condition number does not exist yet, but
the expected gap between the two slices is smaller than the plan claimed. Worst confusions are congeners and
look-alikes, which is the correct failure mode:
`Papilio polytes → Papilio xuthus`, `Pseudozizeeria maha → Zizina otis`,
`Euploea midamus → Euploea core`, `Passer montanus → Psittacula eupatria` (×2).

**F6 — 147 photos cannot resolve the gates (±6.7 points).** At 77.5% the 95% CI is ±6.7 points,
so "70% target" and "77.5% measured" are barely distinguishable, and the
resolution delta of 4 points is inside noise. If G1/G2 are going to decide a
go/no-go, the set needs to be ~400–600 photos, or the decision must be made on
the paired deltas rather than the absolute level. Stating this now rather than
discovering it at the decision meeting.

**F7 — post-training full-integer INT8 destroys this ViT; weight-only quantisation
does not.** 78.91% fp32 → 0.68% full-integer (−78.23 points) → 77.55% with int8
weights and fp32 activations (−1.36 points), all on the same 147 photos and the
same graph (§3c, §3d). The failure is in the activations, not the weights, and it
is not an input-boundary artefact (the float-I/O control is equally bad). This is
the LayerNorm/attention dynamic-range problem ADR 0001 §5 predicted; the
consequence is that the quantisation mode is decided by measurement, not by the
ADR's original preference, and that **G5 latency — not accuracy — is the remaining
risk on the shipping artifact.**

## 5. What is next, in order

1. ~~Owner sign-off on dropping 384 → 224~~ — approved 2026-11-11; ADR 0001 amended.
2. ~~TFLite build in a Linux container~~ — done 2026-11-12 via GitHub Actions.
   Full-integer INT8 **fails G4 at −78.23 points** (§3c); the conversion toolchain
   is fine, the quantisation is not.
3. ~~Quantisation ladder~~ — done 2026-11-12 (§3d). Dynamic-range INT8 (int8
   weights, fp32 activations, 88.1 MB) costs **−1.36 points** and clears G4 + G7;
   float16 is numerically free but 172.5 MB and breaches G7; full-integer is
   unusable. **ADR 0001 §3 needs amending**: it rejected dynamic range before the
   alternative was known to be worthless.
4. **Training-data pipeline for the head (G2)** — fetch 20–50 iNat observations
   per candidate species with a *train* split disjoint by observation id from this
   eval set, then logistic regression on frozen features. Note §8: this route
   already failed at 3,903 photos and needs materially more data, not more tuning.
5. **Device benchmark (G5)** — needs an Android phone; harness is written to run
   unchanged on whatever gets attached.
6. **Owner slice** — 50 field photos; instructions in `data/eval/owner/README.md`.

## 6. Not measured, explicitly

iOS anything, on-device latency (G5/G6), the month/GPS prior conditions (only
candidate-set size), the owner slice, and any species outside Hong Kong. INT8
accuracy and artifact size were measured on 2026-11-12 (§3c) and are no longer on
this list.

## 7. Methodology notes — the traps worth writing down

**Preprocessing is part of the model definition.** The same weights, the same
photos, the same text embeddings produced **77.55%** top-1 through open_clip's
torchvision transform and **78.91%** through an equivalent PIL transform — a
**1.36 point** swing from the resize/crop implementation alone (torchvision
resizes with antialiasing and rounds differently; PIL bicubic does not). That is
larger than the resolution effect we measured (4.1 points, CI including zero) and
larger than the INT8 penalty (−0.68 points).

Therefore, as a standing rule:

1. **Every comparison is paired** — same images, same preprocessing, same text
   embeddings, one variable changed — and reported with a bootstrap CI on the
   paired difference. Unpaired absolute numbers are not evidence.
2. **Every results table names its preprocessing.** "77.5%" without the transform
   behind it is not reproducible, and two numbers from different pipelines must
   never be subtracted from each other.
3. **Sub-2-point differences on 147 photos are not interpretable**, whether or not
   they look like progress. The CI on a single 77.5% measurement is ±6.7 points.
4. The eval set's preprocessing is: shorter side → 224 bicubic, centre crop,
   CLIP mean/std, no augmentation. The device pipeline must reproduce exactly
   this, and a mismatch will show up as an accuracy drop that looks like a model
   regression.

## 8. G2 — Route A head trained and measured: **FAIL**

Funded 2026-11-11 (Route A, CC0 + CC-BY only). Training set: **3,903 photos, 179
species, 129 observers**, median 20 photos/species (min 8, max 40), 785 CC0 + 3,118
CC-BY, per-photo attribution in `data/train/attribution.csv`. The two numbers that
matter: **G2 is 78.33%, not 85%**, and by the pre-agreed falsification rule
(< 80% top-1) the head does not earn its complexity.

### Protocol (fixed before looking at eval)

- Features frozen at 224 px, fp32, same transform as the fp32 reference.
- Classifier is a 200-way linear layer initialised from the text embeddings; only
  the 179 covered species' rows train, so the 21 uncovered species keep zero-shot
  weights inside the same scorer (the per-species routing, expressed as frozen rows).
- Split **by observer**: 110 train observers (3,750 photos), 19 validation
  observers (153 photos). No eval observation, observer, or near-duplicate session
  is present — asserted in code, not assumed.
- Regularisation chosen **only on the validation split**: grid over wd ∈ {1e-4,
  1e-3, 1e-2} → val top-1 78.08 / **78.54** / 68.98 → selected wd = 1e-3.
- Then, once, scored on the frozen eval manifest (sha256 `b029f55c…` asserted).

### Result at the validation-selected configuration, three seeds

| seed | eval full top-1 | Δ vs zero-shot | common-20 top-1 | Δ vs zero-shot | paired CI (common-20) |
|---|---|---|---|---|---|
| 0 | 77.55% | +0.00 | 75.00% | +0.00 | −13.3…+13.3 |
| 1 | 78.23% | +0.68 | 75.00% | +0.00 | −13.3…+13.3 |
| 2 | 77.55% | +0.00 | 73.33% | −1.67 | −13.3…+11.7 |

Zero-shot reference on the same 147 photos: 77.55% full, 75.00% common-20.

**The head buys nothing measurable.** The paired deltas straddle zero in every
seed, and the spread between seeds (±2.7 points on common-20) is larger than the
effect. G2 stays **FAIL**; G1 and G3 are unchanged.

### The trap worth recording — one configuration "passed"

With **wd = 1e-4** the same pipeline scores **83.67% full / 85.00% common-20**,
i.e. exactly on the gate. That number is *not* the result, for two reasons:

1. Validation **preferred wd = 1e-3** (78.54 vs 78.08). Choosing 1e-4 because it
   scores better *on the eval set* is selecting on the test set — the one thing
   this harness exists to prevent.
2. Averaged over the three tuned configurations the head sits at ~76% common-20;
   85.0% is the top of a 12-point spread driven by a hyperparameter, not a
   reproducible capability.

It is written down here rather than deleted, because a future reader will find
that number in `data/train/` scratch runs and needs to know why it is not the
headline. **A single configuration landing exactly on a threshold, chosen after
seeing the threshold, is a coincidence with good PR.**

### Consequence — what I recommend

Per the owner's falsification rule:

1. **Ship zero-shot.** It is simpler, has no training set to defend, no licence
   surface beyond the text embeddings, and performs the same.
2. **Document the common-20 bar at 75%** as measured (zero-shot), and carry 85% as
   an unmet v2 target rather than pretending it was cleared.
3. The head is not abandoned, it is **not funded further without new information**:
   either materially more data (the budget's 5,566 were not reachable — Route A
   delivered 3,903 because permissive photos are thinner per observer than raw
   counts suggested), or a different approach (fine-tuning the tower, not just a
   linear probe; or two-stage coarse-then-species).
