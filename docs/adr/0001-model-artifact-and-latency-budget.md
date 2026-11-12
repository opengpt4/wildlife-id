# ADR 0001 — On-device model artifact and latency budget

- **Status:** Accepted; amended 2026-11-11 (224 px, owner-approved) and 2026-11-12
  (quantisation mode, §3a)
- **Amendment:** charter decision 2 (384 px) is superseded by §2 — parent is editing the charter;
  until that edit lands this ADR is authoritative for the shipped input resolution.
  §3's full-integer INT8 requirement is superseded by §3a.
- **Date:** 2026-11-11
- **Deciders:** Owner (Keung) approved the budget; Atlas authored
- **Driver:** Charter decisions 1 (offline-first), 2 (BioCLIP vision tower + regional
  linear head, TFLite INT8, 384 px) and gate G5/G7 in `docs/phase0-plan.md`.

## Context

The charter fixes the model family, the export format, the input resolution and a
120 MB artifact ceiling before anything is measured. Two of those choices are
coupled and cannot be judged independently: **resolution decides compute, compute
decides which device class can hit 1000 ms, and the INT8 ceiling decides which
backbone fits in 120 MB.** This ADR records the concrete artifact, the budget
breakdown, and what has to be true for the choice to hold.

The constraint that matters: on-device latency is dominated by the vision encoder.
Everything else in the pipeline is noise by comparison.

## Decision

### 1. Model: `imageomics/bioclip`, ViT-B/16 (open_clip), vision tower only

- 86.2 M parameters in the vision tower. At INT8 that is **≈ 86 MB**, leaving
  ~34 MB of headroom under G7 for the head (200 × 768 floats ≈ 0.6 MB), the
  preprocessing tables and pack metadata.
- The **text tower is not shipped to the device.** Text embeddings for the
  candidate species are precomputed offline and travel in the region pack
  (200 × 768 fp32 ≈ 0.6 MB), exactly as the charter specifies. This is what makes
  a 120 MB budget survivable at all.
- Region pack = vision tower + text-embedding matrix + species metadata + linear
  head, versioned and checksummed as one unit.

### 2. Resolution: measure 224 and 384 at fp32 before committing to 384

BioCLIP is trained at 224 px; 384 is an interpolation the model has never seen.
Charter decision 2 specifies 384 on the assumption that it helps small subjects.
That assumption is cheap to test and expensive to keep if wrong: **384 costs
2.9× the compute of 224** (patch count scales as (384/224)²).

- Config **A = 384 px** — the charter default.
- Config **B = 224 px** — the documented fallback, 2.9× cheaper.
- `ml/eval/score.py` scores both at fp32 on the eval set. **If A does not beat B
  by ≥ 2 points top-1, we adopt B** and buy back latency headroom; the charter's
  384 is then amended with a number behind it rather than an opinion.

**Outcome (measured 2026-11-11, 147 photos, `docs/phase0-results.md` §2):** A did
not beat B. Paired delta at K=200 was **−4.1 points top-1 in favour of 224**
(95% CI −10.2…+2.0; 8 photos fixed, 14 broken by going to 384). The v0
single-season set had shown +2.1 for 384, which is precisely why the resolution
call needed a seasonally stratified set and a paired test.

**Resolution decision: adopt Config B (224 px). Approved by the owner 2026-11-11.**

**The decision is not "224 is more accurate than 384".** It is:

1. **No detectable accuracy difference.** The paired delta is −4.1 points top-1
   with a 95% CI of −10.2…+2.0, i.e. the set cannot resolve the difference in
   either direction. Claiming 224 is "better" would be reading noise as signal.
2. **224 is the model's native training resolution.** BioCLIP was trained at 224;
   384 is an interpolation the encoder has never seen.
3. **224 buys ~2.9× latency headroom** against a G5 that §5's arithmetic already
   predicts will fail at 384 on CPU-only mid-range hardware.

So the decision rests on cost and evidence, not on a 4.1-point delta that the
eval set cannot resolve. A sensible reader could object that we are trading a
possible real gain for an unmeasured one; the counter is that the latency budget
is measured in seconds of overrun, while the accuracy budget is 147 photos wide.

**Sample size caveat, stated plainly:** 147 photos gives **±6.7 points** at the
95% level around a 77.5% estimate. The absolute number is coarse. What carries
the argument is the *paired* test — same 147 images, same text embeddings, only
the input resolution changed — not the absolute difference between two runs.

**384 px is retained as the documented fallback** with its measured numbers
(top-1 73.5%, top-5 93.2% at K=200, `docs/phase0-results.md` §2). Revisit 224 if:
the eval set grows to materially more than ~400 photos and 384 shows a paired
gain whose CI excludes zero; or the owner slice (once it exists) systematically
disagrees with the iNat slice; or a target device has enough NPU headroom that the
compute is free anyway.

**G4 must be measured at 224**, because 224 is the shipped artifact — an INT8
penalty measured at a resolution we do not ship would answer the wrong question.

### 3. Export: PyTorch → ONNX (opset 17) → TFLite, full-integer INT8

- ~~Static, per-channel weight quantization; activation ranges calibrated on
  **300 HK observation photos**, disjoint from the eval set. Dynamic-range INT8 is
  not acceptable for a ViT: quantizing activations at runtime loses more than it
  saves on attention blocks.~~ **Superseded 2026-11-12 — see §3a.
- Fallback ladder if G4 (≤ 2.0 points) fails, in order: (a) per-channel full-int8
  with a larger, better-stratified calibration set; (b) `float16` weights +
  int8 activations hybrid; (c) TFLite `float16` entirely (~172 MB — **fails G7**,
  so this requires an owner decision); (d) distilled student model, Phase 1.
- Backend ladder if TFLite cannot represent the graph faithfully
  (LayerNorm / GELU / softmax are the risk ops): ExecuTorch, then ONNX Runtime
  Mobile. iOS later gets Core ML. Switching backends is a Phase 1 decision and
  must not be taken silently.

### 3a. Amendment 2026-11-12 — the shipped artifact is dynamic-range INT8

The requirement above was written before full-integer INT8 had been measured, on
the assumption that activation quantisation would cost a few accuracy points. It
costs all of them. Measured on the real artifacts, 147-photo eval set, paired
(`docs/phase0-results.md` §3c, §3d):

| Mode | size | top-1 | Δ vs fp32 |
|---|---|---|---|
| full-integer INT8 (int8 I/O) | 88.6 MB | 0.68% | **−78.23** |
| full-integer INT8 (float I/O) | 88.6 MB | 0.00% | **−78.91** |
| **dynamic range (int8 weights)** | **88.1 MB** | **77.55%** | **−1.36** [−3.40, 0.00] |
| float16 | 172.5 MB | 78.91% | +0.00 |

**Decision: the shipped artifact is dynamic-range INT8** — int8 weights, fp32
activations, 88.1 MB. G4 (−1.36 pts) and G7 (88.1 MB) both pass on it.

Two consequences, stated plainly rather than discovered later:

1. **This is a latency risk, not an accuracy one.** Keeping activations in fp32
   gives up exactly the memory-traffic saving that motivated full-integer INT8, and
   §5's arithmetic was already predicting failure on CPU-only hardware at 224 px.
   G5 becomes the gate that decides whether the product ships, and it needs a
   device.
2. **The fallback ladder above was mis-specified.** (a) "per-channel full-int8 with
   better calibration" and (b) "float16 weights + int8 activations" are not
   expressible in the TFLite converter; it offers dynamic range, float16 and full
   integer, and no per-op exemption. Selective quantisation without QAT is not an
   option this toolchain has. (c) float16 at 172.5 MB is measured and breaches G7
   by 52 MB — an owner decision if accuracy must be bit-exact. (d) distillation is
   unchanged as the Phase 1 escape hatch.

### 4. Latency budget (G5: p50 ≤ 1000 ms, SD695 / Pixel 6a class)

| Stage | Budget (ms) | Basis |
|---|---|---|
| Decode + downscale 12 MP → 384² (or 224²) | 120 | *estimate*: camera JPEG/HEIC hardware path + scaling |
| Normalize / pack | 15 | *estimate* |
| **Vision encoder (INT8)** | **750** | remainder minus overhead and slack |
| Head + softmax + top-k (200 classes) | 5 | dot product, negligible |
| Copy to UI / JSON | 30 | *estimate* |
| Slack | 80 | 8% |

Model load is **excluded** from per-photo latency: load once at app start, keep
resident (~90 MB weights + ~50 MB activations fits a 4 GB device). Cold start is
reported separately as time-to-first-identification, which is what the user
actually feels.

### 5. Predicted failure — stated before measurement so it cannot be spun later

These are *estimates* from published FLOPs, not measurements. They exist to size
the risk, and the spike exists to replace them.

| Config | GFLOPs / image | Est. CPU INT8 (SD695) | Est. GPU delegate | Verdict |
|---|---|---|---|---|
| ViT-B/16 @ 224 | ~17.6 | 1.2–2.9 s | 0.3–0.6 s | GPU: pass · CPU: fail |
| ViT-B/16 @ 384 | ~51.7 | 3.4–8.6 s | 0.9–1.7 s | GPU: borderline · CPU: fail |
| ViT-L/14 @ 224 | ~80.6 | — | — | rejected |

Assumptions behind the CPU column: XNNPACK INT8 sustained 6–15 GFLOP/s on a
4-core A78/A55 mix. The estimate is deliberately wide because ViT attention is
memory-bound and poorly served by CPU kernels.

**Consequence:** G5 at 384 px is predicted to fail on CPU and to be borderline on
GPU. This is the single most likely Phase 0 failure, and it is why Config B (224)
and the GPU-delegate path are both pre-agreed fallbacks rather than emergency
improvisation.

## Options considered and rejected

| Option | Why rejected |
|---|---|
| **BioCLIP 2 (`imageomics/bioclip-2`)**, ViT-L/14 | 303.7 M vision params → ~304 MB INT8. Blows G7 (120 MB) by 2.5× before any head or pack. Also ~80.6 GFLOPs at 224 — 4.6× ViT-B, so G5 would fail on any mid-range device. |
| **BioCLIP 2.5 (`imageomics/bioclip-2.5-vith14`)** | Same size/latency class as BioCLIP 2. Not evaluated further for the same reason. |
| ViT-L/14 at 224 on the server only | Legitimate for the escalation path, but it cannot be the core loop: charter decision 1 forbids a network dependency for identification. |
| MobileNet / EfficientNet-based classifier | ~10× faster and ~5 MB, but it is a closed-world trained classifier, not an open-vocabulary embedder; loses the text-embedding mechanism that makes region packs cheap and the "200 species" set swappable. Reconsider only if ViT-B cannot hit G5 *and* a distilled student is wanted. |
| Server-first identification | Directly contradicts charter decision 1. |
| 10k-species global zero-shot on device | Contradicts charter decision 3 and inflates the text-embedding pack, which is not the bottleneck but for the wrong reasons — accuracy is. |

## Consequences

- We are betting one encoder pass per photo against a 750 ms budget, on hardware
  we do not own yet. That is the whole risk of the product in one line.
- The region pack is the unit of versioning and download; model changes ship as
  packs, not app releases. Good for iteration, and it means a bad model can be
  rolled back without an app-store review (Phase 1 work).
- INT8 is load-bearing for G7, so the quantization spike cannot be skipped or
  postponed — an fp32 artifact is 345 MB and simply does not ship.
- Everything downstream (camera capture resolution, burst strategy, crop guidance)
  should be designed against the measured latency, not the budgeted one.

## Revisit triggers

1. G5 fails at 224 px on GPU too → the vision encoder must be distilled
   (Phase 1 becomes "train a student", a different project shape).
2. G4 fails after the export ladder → switch backend (ExecuTorch / ORT Mobile) or
   accept fp16 with an explicit G7 re-negotiation by the owner.
3. A device with a real NPU appears in the target set → re-run S1 before designing
   around it; NPU delegates change the arithmetic completely.
4. The 224-vs-384 result contradicts charter decision 2 → amend the charter.

## Verification

Numbers from spikes S1 (latency) and S2 (quantization accuracy) in
`docs/phase0-results.md`, against the artifact whose SHA-256 is recorded there.
Until those exist, this ADR is a proposal with a budget, not a result.
