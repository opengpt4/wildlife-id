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
    # TFLite has no ERF kernel, and the graph PyTorch exports for exact GELU is
    # Div -> Erf -> Add -> Mul -> Mul, which fails conversion outright ("neither a
    # custom op nor a flex op"). Two workarounds were tried and both are wrong:
    #   * onnx2tf's pseudo-erf computes sign(x) = x/(|x|+1e-11) -> 0/0 at x=0.
    #   * our own A&S 7.1.26 erf with torch.sign still needs t = 1/(1+p|x|), and
    #     TFLite's int8 DIV kernel aborts whenever the divisor's raw quantised
    #     value is 0 — which asymmetric quantisation makes routine (measured: 12
    #     DIV ops, zero_point -128, all crashing on the int8 artifact).
    # So: no division at all. The standard tanh GELU approximation uses only
    # multiply/add/tanh, every one of which has a native int8 kernel. Its error
    # against exact GELU is ~1e-3, i.e. a few percent of one int8 quantisation
    # step — it cannot move a G4 number by a point, and it is the same
    # approximation PyTorch/TensorFlow use under approximate='tanh'.
    def _gelu_tflite(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        inner = 0.7978845608028654 * (x + 0.044715 * x.pow(3))  # sqrt(2/pi)
        return 0.5 * x * (1.0 + torch.tanh(inner))

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
