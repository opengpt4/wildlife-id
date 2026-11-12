# wildlife-id — project charter

Mobile app (Android + iOS) for identifying wildlife from photos, focused on
**butterflies and birds in Hong Kong**. This file is the standing brief for every
agent working in this repository. Keep it current; it is the handover document.

## Product

Users photograph a butterfly or bird (camera or photo library), get an
identification with honest uncertainty, and keep a personal observation history
(life list, trip lists).

Prior art on this machine: `~/bird-photo-manager` — an Electron desktop app with
LM Studio / Gemini species ID, canvas-based quality rating, and sidecar JSON
metadata. Reuse its quality-gate logic and identification response shape; do not
reuse its desktop architecture.

## Architecture decisions

1. **Offline-first.** The app must identify photos with no network. Server is an
   enhancement, never a dependency for the core loop.
2. **Primary model: BioCLIP vision tower, zero-shot over the regional candidate
   set.** Text embeddings are precomputed offline, so per-photo cost on device is
   one encoder pass plus a dot product against a ~200-species matrix. Export to
   TFLite at **224 px** (amended 2026-11-11: the paired 224-vs-384 confidence
   interval includes zero, 224 is the native training resolution, and it buys
   ~2.9× latency headroom).

   **Quantisation: int8 weights with fp32 activations (amended 2026-11-12).**
   Full-integer INT8 — the mode ADR 0001 originally required — was built and
   measured at **0.68% top-1, i.e. chance** (−78.23 points); a float-I/O control
   ruled out the input boundary, and the error compounds through the transformer
   blocks exactly where LayerNorm/attention lose dynamic range. The TFLite
   converter offers no per-op exemption, so keeping LayerNorm in fp32 needs QAT.
   The shipped artifact is therefore **dynamic-range INT8, 88.1 MB, −1.36 points**
   [−3.40, 0.00] against fp32. See ADR 0001 §3a and `docs/phase0-results.md`
   §3c/§3d. Do not reintroduce full-integer INT8 without a QAT story.

   **The regional linear head was tested and dropped (2026-11-11).** A head trained
   on 3,903 permissively licensed HK photos (179 species, observer-split, leakage
   guards asserted) scored 77.55% top-1 against zero-shot's 77.55% — no gain, with
   seed spread (±2.7 pts) exceeding the effect. See `docs/phase0-results.md` §8.
   Consequence: **75% common-species accuracy is the measured bar; 85% is an unmet
   v2 target, not a Phase 0 deliverable.** Revisit only with a materially larger,
   better-balanced training set — not with different hyperparameters on this data.
3. **Regional candidate set is the main accuracy lever.** Narrowing the candidate
   list before ranking is what buys accuracy; global zero-shot over 10k species is
   not the product. Measured (2026-11-12): **size pays, month does not** — 200
   species versus the full 828-species checklist is worth **+9.5 points**, while a
   month-of-year filter costs up to −3.4 points and still leaves 169–192 of 200
   candidates, because 192/200 HK species are recorded in every month
   (`docs/phase0-results.md` §3e, F8). **Do not build month filtering into v1 on
   this evidence.** GPS/range and habitat are untested and are the next candidate;
   revisiting month needs a phenology source that is not iNaturalist effort.
4. **VLM is for explanation, not classification.** Show BioCLIP probabilities as
   uncertainty; use a VLM only for the "why" text and second opinions.
5. **Server is optional.** FastAPI + PyTorch + object storage + Postgres. Used
   for low-confidence escalation, rare species, and model updates via downloadable
   region packs.
6. **Local-first storage.** SQLite with UUIDs, soft deletes and a sync log. The
   server is a mirror, never the source of truth.
7. **Mobile stack:** React Native (bare, not Expo) + `react-native-vision-camera`
   + `react-native-fast-tflite`, to reuse the TypeScript work from
   `bird-photo-manager`.
8. **Artifacts are built in CI, not on the dev Mac (2026-11-12).** `tensorflow`
   and `onnx` cannot coexist in one process on macOS, so TFLite conversion runs in
   GitHub Actions (`.github/workflows/build-tflite.yml`, ~21 min, five artifacts
   per run) and is scored locally with `ml/export/eval_tflite.py`. The repo is
   public because private-repo runners have 2 vCPU / 8 GB and OOM during
   conversion; public runners have 4 vCPU / 16 GB. **Conversion is not
   bit-reproducible** — identify an artifact by its score and manifest, not by
   hash alone.

## Repository layout

```
app/      React Native application          (empty — no feature work yet)
server/   FastAPI service                   (empty — optional escalation path)
ml/       model export, quantization, evaluation harness
data/     species lists, benchmark photo sets, iNat ingest output
docs/     ADRs, phase 0 results, data model, decisions log
```

## Working agreements

- **De-risk before building.** No feature work until Phase 0 assumptions are
  measured. Accuracy and latency numbers beat opinions.
- **Numbers over adjectives.** Report top-1/top-5 and milliseconds, on named
  devices, with the exact model artifact.
- **Write decisions down.** Anything reversible in a day: just do it. Anything
  that changes the architecture: add an ADR in `docs/` first.
- **Never fake confidence.** If the model is unsure, the UI says so.
- **No secrets in the repo.** API keys live in the OS keychain or `.env`
  (gitignored).

## Status

**Phase 0 is measured, not finished.** Five of seven gates have a verdict; G5 and
G6 need hardware that is not attached, and no feature work has started — `app/`
and `server/` are empty on purpose.

Shipping candidate: BioCLIP ViT-B/16 vision tower, 224 px, dynamic-range INT8,
**88.1 MB**, zero-shot over the 200-species HK candidate set.

Numbers below are the **shipping artifact** (dynamic-range INT8); where the fp32
reference differs it is given in brackets, because the gate was originally set
against fp32.

| Gate | Target | Measured | Status |
|---|---|---|---|
| G1 top-1 | ≥ 70% | 77.55% (fp32: 78.91%) | pass |
| G2 common-20 | ≥ 85% | 73.3% (fp32: 75.0%) | **fail** — v2 target, needs data not tuning |
| G3 top-5 | ≥ 90% | 91.16% | pass |
| G4 INT8 vs fp32 | ≤ 2.0 pts | −1.36 pts [−3.40, 0.00] | pass, thin margin |
| G5 latency p50 | ≤ 1000 ms | not measured | **blocked: no Android device** |
| G6 iPhone latency | ≤ 500 ms | not run | deferred, no Xcode here |
| G7 artifact | ≤ 120 MB | 88.1 MB | pass |

The five spike questions, status:

1. **INT8 latency on target devices — not answered.** Needs a phone (G5). This is
   the only remaining gate that can invalidate the architecture: weight-only
   quantisation keeps activations in fp32, so the memory-traffic saving that made
   the budget plausible is gone. ADR 0001 §5's arithmetic already predicted CPU
   failure at 224 px.
2. **Quantisation delta — answered.** Full-integer INT8 is unusable (0.68% top-1,
   −78.23 pts); dynamic-range INT8 costs −1.36 pts at 88.1 MB; float16 is
   numerically free but 172.5 MB, which breaches G7. `docs/phase0-results.md`
   §3c/§3d.
3. **iNaturalist availability — answered.** 828 research-grade HK species in the
   checklist (562 birds, 266 butterflies); a 200-species candidate list whose
   thinnest member still has 346 HK observations; 3,903 permissively licensed
   photos for the head experiment.
4. **RN camera + fast-tflite compatibility — not started.** Also blocked on the
   device; `react-native-fast-tflite` is still an untested choice.
5. **Regional prior — answered: size yes, month no.** +9.5 points top-1 (77.5% at
   K=200 vs 68.0% at K=828); the month arm measured negative (−0.0 to −3.4 points,
   §3e), so it is not a v1 feature. Separately, the prompt template is worth +24
   points, so the region pack must ship the exact template with the embeddings.

Where the numbers live: `docs/phase0-results.md` (all gates, methodology traps,
§8 for the head), `docs/adr/0001-model-artifact-and-latency-budget.md` (artifact and
latency budget, §3a for quantisation), `data/eval/results/*.json` (raw scores),
`.github/workflows/build-tflite.yml` (how the artifacts are built).

## Open questions for the owner

- **Which device, and when?** G5 needs one Android phone of the SD695 / Pixel 6a
  class; every remaining engineering decision is downstream of that number.
- **88.1 MB dynamic-range INT8, or renegotiate G7 for 172.5 MB float16?** float16
  reproduces fp32 predictions exactly, which is the only reason to consider it.
- Target minimum OS versions and the primary test device.
- Whether identification history syncs across devices in v1.
- Language: Traditional Chinese + English, or English only at launch.
