"""Fold the alpha/beta residual gates of a (mid-training) gated checkpoint into a plain Qwen2.

The grown model (modeling_gated_qwen2.py) carries a learnable scalar gate on each inserted
layer's two residual branches:

    h = x + alpha * Attention(Norm(x)),     y = h + beta * MLP(Norm(h))

Because each gate multiplies a whole residual branch whose last op is a linear projection
(o_proj for attention, down_proj for the MLP), the gate folds into that projection EXACTLY:

    o_proj.weight   <- alpha * o_proj.weight
    down_proj.weight <- beta  * down_proj.weight

then the gate is deleted. This is an algebraic identity -- the folded model computes the
SAME function (its loss is unchanged at the fold step). The result is a standard
Qwen2ForCausalLM: no gates, no extra params, no remote code.

Use it to fold MID-TRAINING (not only at deploy): train gated for a while so the gates fade
the inserted layers in smoothly, then fold and continue training the plain model uniformly
(normal weight decay + LR on the now-full o_proj/down_proj). To keep the Adam moments across
the fold, run fold_optimizer.py on the same checkpoint afterwards.

Usage:
    python fold_model.py <gated_ckpt_dir> <out_dir>

    # then (optional) carry the optimizer moments, and continue training:
    python fold_optimizer.py <gated_ckpt_dir> <out_dir>
    python train.py --model <out_dir> --prior-steps <global_step> ... \
                    --init-optimizer-from <out_dir>

Note: <gated_ckpt_dir> must be loadable with trust_remote_code (it needs
modeling_gated_qwen2.py). If a training checkpoint lacks that file, copy it in from the
grown-model dir first.
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


@torch.no_grad()
def fold_model(gated_ckpt, out_dir):
    model = AutoModelForCausalLM.from_pretrained(
        gated_ckpt, trust_remote_code=True, torch_dtype=torch.float32
    )
    n_folded = 0
    for layer in model.model.layers:
        for parent, name in ((layer.self_attn, "o_proj"), (layer.mlp, "down_proj")):
            mod = getattr(parent, name)
            if hasattr(mod, "gate") and hasattr(mod, "base"):       # a GatedLinear
                mod.base.weight.data.mul_(mod.gate.data)            # exact: W <- gate * W
                setattr(parent, name, mod.base)                     # swap in the plain Linear
                n_folded += 1

    # Rebuild a clean plain config (drop the gated-only fields so it loads as a stock Qwen2).
    cfg = Qwen2Config(**{k: v for k, v in model.config.to_dict().items()
                         if k not in ("gated_layers", "auto_map", "architectures")})
    cfg.architectures = ["Qwen2ForCausalLM"]
    plain = Qwen2ForCausalLM(cfg)
    plain.load_state_dict(model.state_dict(), strict=True)          # keys are now plain (no .base/.gate)

    os.makedirs(out_dir, exist_ok=True)
    plain.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(gated_ckpt, trust_remote_code=True).save_pretrained(out_dir)
    print(f"folded {n_folded} gates -> plain Qwen2 saved to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python fold_model.py <gated_ckpt_dir> <out_dir>")
    fold_model(sys.argv[1], sys.argv[2])
