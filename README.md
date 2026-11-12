# wildlife-id

Offline-first identification of **Hong Kong butterflies and birds** from a phone
photo: BioCLIP vision tower → cosine against precomputed text embeddings → top-k
species, with honest uncertainty. Android + iOS, React Native, no network needed
for the core loop.

Phase 0 is measurement, not product code. `app/` and `server/` are empty on
purpose — see `AGENTS.md`, which is the standing brief and the document to read
first.

## Where the project stands

| Gate | Target | Measured | |
|---|---|---|---|
| G1 top-1 | ≥ 70% | 77.55% | pass |
| G2 common-20 | ≥ 85% | 73.3% | fail — v2 target |
| G3 top-5 | ≥ 90% | 91.16% | pass |
| G4 INT8 vs fp32 | ≤ 2.0 pts | −1.36 pts | pass, thin margin |
| G5 latency p50 | ≤ 1000 ms | — | **blocked: no device** |
| G6 iPhone latency | ≤ 500 ms | — | deferred |
| G7 artifact | ≤ 120 MB | 88.1 MB | pass |

Shipping candidate: BioCLIP ViT-B/16, 224 px, dynamic-range INT8 (int8 weights,
fp32 activations), 88.1 MB, zero-shot over 200 HK species. Full-integer INT8 measured
at chance and is not a viable artifact — `docs/phase0-results.md` §3c.

## Reproduce a number

Artifacts cannot be built on macOS (`tensorflow` and `onnx` collide in one
process), so the conversion runs in GitHub Actions:

```bash
gh workflow run build-tflite.yml -f resolution=224     # ~21 min, five artifacts
gh run download <run-id> -n bioclip-vitb16-224-tflite -D data/eval/artifacts

python3.11 -m venv .venv-tflite && .venv-tflite/bin/pip install tensorflow==2.21.0 pillow numpy
.venv-tflite/bin/python ml/export/eval_tflite.py \
    --tflite data/eval/artifacts/vision_tower_224_int8_dynamic.tflite \
    --label int8_dynamic --artifact-dir data/eval/artifacts \
    --out data/eval/results/tflite_int8_dynamic.json
```

The scorer refuses to run if the prompt strings moved, because a prompt change is
a model change (`docs/phase0-results.md` §4, F2: the template is worth 24 points).

## Layout

```
AGENTS.md   standing brief: product, decisions, status, open questions
app/        React Native application (empty)
server/     FastAPI escalation service (empty)
ml/         export, quantisation, evaluation harness
data/       species lists, eval manifests, iNat ingest output
docs/       phase 0 results, ADRs
```

## Data and licensing

Eval photos are iNaturalist observations, downloaded locally and **not
redistributed** (`data/eval/inat/images/` is gitignored); per-photo attribution and
licence codes are in `data/eval/inat/manifest.csv`. Training photos for the
abandoned head are likewise local-only, with `data/train/attribution.csv` recording
provenance. No API keys are stored in the repo.

## Decisions

- `docs/phase0-results.md` — every gate, the methodology traps, and what failed
- `docs/adr/0001-model-artifact-and-latency-budget.md` — artifact, resolution,
  quantisation mode, latency budget
