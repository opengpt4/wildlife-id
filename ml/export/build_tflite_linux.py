#!/usr/bin/env python3
"""Linux-only: ONNX -> TFLite fp32 and full-integer INT8, plus a manifest.

Why a separate script and not a flag on the exporter: this is the half of the
pipeline that cannot run on the macOS dev host (tensorflow + onnx collide in one
process there). It is written for the GitHub Actions workflow in
`.github/workflows/build-tflite.yml` and for any Linux box with the same packages.

INT8 here is *full integer*: weights and activations, with the input tensor
quantised, calibrated on real HK photos listed in `calibration_manifest.csv`.
That is deliberately stricter than the ONNX dynamic-quantisation proxy used for
the G4 estimate locally, and it is why the artifact must be re-scored before G4/G7
are called closed.

Usage:
  python ml/export/build_tflite_linux.py \
      --onnx data/eval/artifacts/vision_tower_224.onnx \
      --calibration ml/export/calibration_manifest.csv \
      --out-dir data/eval/artifacts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image(path_or_url: str, size: int) -> np.ndarray:
    if path_or_url.startswith("http"):
        with urllib.request.urlopen(path_or_url, timeout=60) as r:
            data = r.read()
        im = Image.open(io.BytesIO(data))
    else:
        im = Image.open(path_or_url)
    with im as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = size / min(w, h)
        im = im.resize((max(size, round(w * scale)), max(size, round(h * scale))),
                       Image.BICUBIC)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return ((arr - CLIP_MEAN) / CLIP_STD).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out-dir", default="data/eval/artifacts")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--calibration-limit", type=int, default=300)
    args = ap.parse_args()

    import onnx2tf
    import tensorflow as tf

    os.makedirs(args.out_dir, exist_ok=True)
    saved_model_dir = os.path.join(args.out_dir, "saved_model")

    # ---- ONNX -> SavedModel (onnx2tf) ---------------------------------------
    print("converting ONNX -> SavedModel …", file=sys.stderr)
    onnx2tf.convert(
        input_onnx_file_path=args.onnx,
        output_folder_path=saved_model_dir,
        non_verbose=True,
        # keep NCHW input: our preprocessing and the ONNX graph are NCHW
        keep_ncw_or_nchw_or_ncdhw_input_names=["image"],
    )

    # ---- calibration set -----------------------------------------------------
    with open(args.calibration) as f:
        cal_rows = [r for r in csv.DictReader(f)][: args.calibration_limit]
    print(f"calibration images: {len(cal_rows)}", file=sys.stderr)
    cal_arrays = []
    for r in cal_rows:
        src = r.get("url") or r.get("file") or ""
        if not src:
            continue
        try:
            x = load_image(src, args.resolution)
            cal_arrays.append(np.transpose(x, (2, 0, 1))[None, ...])  # 1,C,H,W
        except Exception as e:  # noqa: BLE001
            print(f"  ! calibration skip {src}: {e}", file=sys.stderr)
    if not cal_arrays:
        raise SystemExit("no calibration images could be loaded — refusing to quantise")
    print(f"calibration arrays: {len(cal_arrays)}", file=sys.stderr)

    # ---- TFLite conversions ---------------------------------------------------
    def convert(quantize: bool) -> str:
        conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if quantize:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]

            def rep():
                for arr in cal_arrays:
                    yield [arr]

            conv.representative_dataset = rep
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
        tflite = conv.convert()
        name = "vision_tower_224_int8.tflite" if quantize else "vision_tower_224_fp32.tflite"
        path = os.path.join(args.out_dir, name)
        with open(path, "wb") as f:
            f.write(tflite)
        return path

    paths = {}
    for label, quant in (("fp32", False), ("int8", True)):
        print(f"converting to TFLite {label} …", file=sys.stderr)
        paths[label] = convert(quant)

    # ---- prompt integrity must travel with the artifact -----------------------
    prompts_json = os.path.join(args.out_dir, "prompts.json")
    prompt_hash = None
    if os.path.exists(prompts_json):
        prompt_hash = json.load(open(prompts_json)).get("prompt_list_sha256")

    manifest = {
        "source_onnx": os.path.abspath(args.onnx),
        "source_onnx_sha256": sha256_file(args.onnx),
        "resolution": args.resolution,
        "prompt_list_sha256": prompt_hash,
        "calibration_images": len(cal_arrays),
        "quantisation": "full-integer int8 (weights + activations + input), calibrated",
        "gelu": "erf via Abramowitz & Stegun 7.1.26 with sign(), applied in the "
                "exporter so the graph contains no tf.Erf (max abs err ~1.4e-7)",
        "toolchain": {
            "tensorflow": tf.__version__,
            "onnx2tf": getattr(onnx2tf, "__version__", "unknown"),
        },
    }
    for label, path in paths.items():
        manifest[label] = {
            "path": path,
            "sha256": sha256_file(path),
            "size_mb": round(os.path.getsize(path) / 1e6, 1),
        }
        print(f"{label}: {manifest[label]['size_mb']} MB  sha256 {manifest[label]['sha256'][:16]}…",
              file=sys.stderr)

    with open(os.path.join(args.out_dir, "tflite_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote tflite_manifest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
