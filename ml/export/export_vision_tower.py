#!/usr/bin/env python3
"""Export the BioCLIP vision tower to ONNX at 224 px, and the candidate text
embeddings with a *hash-verified* prompt template.

Why the hashing: G4 compares the INT8 artifact against the fp32 reference on the
same eval set. If the prompt template changed between the two runs (a stray edit,
a different default), the comparison would silently measure a different model.
So the prompt strings are materialised once, hashed, and every downstream stage
asserts the hash.

Outputs (data/eval/artifacts/):
  vision_tower_224.onnx        fp32 vision tower, input 1x3x224x224, output 1x512
  text_embeddings_200.npz      L2-normalised text embeddings + species list
  prompts.json                 the exact prompt strings, template, and sha256s
  export_manifest.json         model revision, param count, hashes, sizes

Usage (torch venv):
  .venv/bin/python ml/export/export_vision_tower.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
import score as S  # noqa: E402 - same constants as the fp32 scorer, on purpose


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class VisionTower(torch.nn.Module):
    """model.visual(x) -> 512-d embedding (pre-normalisation)."""

    def __init__(self, visual: torch.nn.Module):
        super().__init__()
        self.visual = visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def main() -> int:
    # ---- TFLite-safe GELU -----------------------------------------------------
    # TFLite has no ERF kernel, so exact GELU (Div -> Erf -> Add -> Mul -> Mul)
    # cannot convert. onnx2tf's pseudo-erf workaround computes sign(x) as
    # x / (|x| + 1e-11); under int8 quantisation that denominator rounds to zero
    # for a zero input and the DIV kernel aborts ("data[i] != 0 was not true").
    # Applying the approximation here instead, with torch.sign in place of x/|x|
    # (ONNX Sign -> tfl.sign), keeps every op one TFLite actually has.
    # Formula: Abramowitz & Stegun 7.1.26, max abs error ~1.5e-7 — three orders of
    # magnitude below int8 quantisation noise, so it cannot move a G4 number.
    def _erf_as(x: torch.Tensor) -> torch.Tensor:
        a1, a2, a3, a4, a5, p = (
            0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911)
        ax = x.abs()
        t = 1.0 / (1.0 + p * ax)
        poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
        return torch.sign(x) * (1.0 - poly * torch.exp(-ax * ax))

    def _gelu_tflite(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        return 0.5 * x * (1.0 + _erf_as(x * 0.7071067811865476))

    torch.nn.functional.gelu = _gelu_tflite

    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--candidates", default="data/species/hk_candidates_200.csv")
    ap.add_argument("--prompt", default="photo", choices=list(S.PROMPT_TEMPLATES))
    ap.add_argument("--out-dir", default="data/eval/artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu"  # export on CPU: deterministic and avoids MPS op gaps

    model, _, _ = S.open_clip.create_model_and_transforms(
        S.MODEL_NAME, force_image_size=args.resolution)
    model = model.to(device).eval()
    tokenizer = S.open_clip.get_tokenizer(S.MODEL_NAME)

    species = S.read_species(args.candidates)
    template = S.PROMPT_TEMPLATES[args.prompt]
    prompts = [template.format(name=s["scientific_name"]) for s in species]

    # ---- text embeddings, materialised next to the ONNX so they cannot drift --
    with torch.no_grad():
        tokens = tokenizer(prompts)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    emb_np = emb.float().numpy()
    npz_path = os.path.join(args.out_dir, "text_embeddings_200.npz")
    np.savez(npz_path,
             embeddings=emb_np,
             scientific_name=np.array([s["scientific_name"] for s in species]),
             prompt=np.array(prompts),
             taxon_id=np.array([s["taxon_id"] for s in species]))

    prompt_hash = sha256_text("\n".join(prompts))
    prompts_doc = {
        "template": template,
        "n_prompts": len(prompts),
        "prompt_list_sha256": prompt_hash,
        "first_three": prompts[:3],
        "last_one": prompts[-1],
        "source": "ml/eval/score.py PROMPT_TEMPLATES (single source of truth)",
    }
    with open(os.path.join(args.out_dir, "prompts.json"), "w") as f:
        json.dump(prompts_doc, f, indent=2)

    # ---- vision tower -> ONNX -------------------------------------------------
    # torch fuses attention into aten::_native_multi_head_attention, which has no
    # ONNX symbolic in opset 17. Disable the fast path so the graph is expressible.
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass
    tower = VisionTower(model.visual).eval()
    dummy = torch.zeros(1, 3, args.resolution, args.resolution)
    onnx_path = os.path.join(args.out_dir, "vision_tower_224.onnx")
    with torch.no_grad():
        torch.onnx.export(
            tower, dummy, onnx_path,
            input_names=["image"], output_names=["embedding"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )

    n_params = sum(p.numel() for p in model.visual.parameters())
    int8_est_mb = n_params / 1e6  # 1 byte per param
    manifest = {
        "model": S.ARCH,
        "checkpoint": S.MODEL_NAME,
        "resolution": args.resolution,
        "vision_params": int(n_params),
        "fp32_size_mb": round(os.path.getsize(onnx_path) / 1e6, 1),
        "int8_theoretical_size_mb": round(int8_est_mb, 1),
        "onnx_sha256": sha256_file(onnx_path),
        "text_embeddings_sha256": sha256_file(npz_path),
        "prompt_list_sha256": prompt_hash,
        "prompt_template": template,
        "n_candidates": len(species),
        "embed_dim": int(emb_np.shape[1]),
    }
    with open(os.path.join(args.out_dir, "export_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"prompt #0  = {prompts[0]!r}")
    print(f"prompt hash= {prompt_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
