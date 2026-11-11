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
   TFLite INT8 at **224 px** (amended 2026-11-11: the paired 224-vs-384 confidence
   interval includes zero, 224 is the native training resolution, and it buys
   ~2.9× latency headroom).

   **The regional linear head was tested and dropped (2026-11-11).** A head trained
   on 3,903 permissively licensed HK photos (179 species, observer-split, leakage
   guards asserted) scored 77.55% top-1 against zero-shot's 77.55% — no gain, with
   seed spread (±2.7 pts) exceeding the effect. See `docs/phase0-results.md` §8.
   Consequence: **75% common-species accuracy is the measured bar; 85% is an unmet
   v2 target, not a Phase 0 deliverable.** Revisit only with a materially larger,
   better-balanced training set — not with different hyperparameters on this data.
3. **Regional candidate set is the main accuracy lever.** GPS + month + habitat
   narrows the candidate list before ranking. Global zero-shot over 10k species
   is not the product.
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

## Repository layout

```
app/      React Native application
server/   FastAPI service (optional escalation path)
ml/       model export, quantization, evaluation harness
data/     species lists, benchmark photo sets, iNat ingest output
docs/     ADRs, API contract, data model, decisions log
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

Phase 0 — not started. The five spike questions:

1. BioCLIP INT8 latency on target devices.
2. Quantization accuracy delta (fp32 vs INT8) on a 200-photo eval set.
3. iNaturalist data availability for HK butterflies and birds.
4. RN camera + fast-tflite compatibility.
5. How much does the regional prior actually lift top-1 accuracy?

## Open questions for the owner

- Target minimum OS versions and the primary test device.
- Whether identification history syncs across devices in v1.
- Language: Traditional Chinese + English, or English only at launch.
