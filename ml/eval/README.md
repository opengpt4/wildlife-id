# ml/eval — Phase 0 evaluation harness

Host-side (laptop) evaluation for the offline model. No app code, no device
required. Everything here produces numbers that go in `docs/phase0-results.md`.

## What each script does

| Script | Purpose | Runtime |
|---|---|---|
| `fetch_eval_set.py` | Builds the 150-photo iNat slice + the 200-species candidate list + the full HK checklist from the public iNat API. Idempotent, resumable, rate-limited to 60 req/min. | ~6 min, network |
| `prepare_owner_slice.py` | Turns `data/eval/owner/owner_metadata.csv` into a manifest `score.py` can read; converts HEIC via macOS `sips`. | seconds |
| `score.py` | BioCLIP ViT-B/16 fp32 zero-shot over the eval set at 224 and 384 px, against multiple candidate sets. Emits JSON + markdown. | ~1–2 min on MPS |

## Reproduce from scratch

```bash
python3.11 -m venv .venv                     # OUTSIDE the repo tree
.venv/bin/pip install -r ml/eval/requirements.txt
export HF_HOME="$PWD/.cache/hf" PIP_CACHE_DIR="$PWD/.cache/pip"

python3 ml/eval/fetch_eval_set.py --out data/eval/inat
.venv/bin/python ml/eval/score.py \
    --manifest data/eval/inat/manifest.csv \
    --resolutions 224,384 --prompts name,photo \
    --out data/eval/results/fp32_inat.json
```

Weights are not committed: `hf-hub:imageomics/bioclip` is fetched on first run
(≈ 600 MB, cached under `$HF_HOME`).

## Protocol (must match the device pipeline)

- shorter side → target, centre crop to target², bicubic, no augmentation
- closed-set cosine similarity against **precomputed text embeddings**, which is
  what ships in the region pack
- top-1 and top-5, with 95% Wilson intervals — 150 photos cannot support more
  than ~±7 points of precision, and the report must say so

## Non-obvious things that cost time (recorded so nobody rediscovers them)

1. **`/v1/photos/{id}` returns 404** for photos hosted on
   `inaturalist-open-data.s3.amazonaws.com`, which is most licensed observations.
   Do not use it. The observation payload's photo `url` is
   `.../photos/<id>/square.jpg`; substitute the size segment to get `large.jpg`
   (~1024 px). That is what `photo_url()` does.
2. **open_clip 3.x dropped `hf_hub:`** (underscore) as a pretrained tag and uses
   the `hf-hub:` schema as the *model name*:
   `create_model_and_transforms('hf-hub:imageomics/bioclip', force_image_size=384)`.
   The `pretrained=` argument is ignored on that path.
3. **384 px is an interpolation.** BioCLIP is trained at 224. `score.py` measures
   both; the charter's 384 must earn its 2.9× compute or be dropped (ADR 0001 §2).
4. `large.jpg` can be 0.5–1 MB; the 150-photo slice is ~100 MB on disk. Gitignored.
5. **TFLite conversion does not work on macOS here.** `tensorflow` + `onnx2tf` in
   one environment: importing `onnx` after `tensorflow` aborts with a libc++
   mutex error; importing `tensorflow` after `onnx` deadlocks. Export ONNX on the
   Mac (`ml/export/export_vision_tower.py`, numerically identical to torch), then
   convert in a Linux container. `ml/export/eval_tflite.py` scores whatever
   `.tflite` comes out and asserts the prompt hashes before reporting a number.
6. The INT8 proxy numbers in `docs/phase0-results.md` §3b come from
   `ml/export/eval_onnx_int8.py` (ONNX Runtime dynamic quantisation). Useful, but
   weights-only: it does not quantise activations or the input tensor.

## Reading the output

`data/eval/results/fp32_inat.md` is the human table. `fp32_inat.json` keeps
per-species and per-group breakdowns for the confusion analysis. Every number is
tied to `manifest_sha256` and the candidate-set sizes, so a number quoted out of
context can always be traced back.

**Zero-shot is a floor, not the shipped model.** There is no training data in it,
so there is no leakage and no optimism. A trained linear head can only raise it.
