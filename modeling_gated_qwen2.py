"""Qwen2 with per-branch zero-init residual gates (the alpha/beta of Sec 4.3).

A grown (deeper) model installs each INSERTED layer dormant but pre-aimed:

    h = x + alpha * Attention(Norm(x)),   alpha_init = 0
    y = h + beta  * MLP(Norm(h)),         beta_init  = 0

so at step 0 every inserted layer is the identity and the deep model computes
EXACTLY the same function as the source -- no loss spike. Training lifts the
gates off 0 and fades in the (already on-trajectory) interpolated weights.

Implementation note. We do NOT reimplement Qwen2DecoderLayer.forward (that
signature drifts across transformers versions). Instead we exploit the fact
that each residual branch ends in a linear projection with no bias:

    attention branch ends in  self_attn.o_proj   (bias=False)
    mlp branch       ends in  mlp.down_proj      (bias=False)

so `alpha * Attention(...)` == scaling o_proj's OUTPUT by alpha, and likewise
for beta/down_proj. We wrap those two projections in a tiny GatedLinear that
multiplies by a learnable scalar. The decoder forward is untouched.

At deployment the gate folds into the projection weight exactly
(`o_proj.weight *= alpha`) and disappears -- see fold_gates() in grow_model.py.
The result is a plain Qwen2 with no extra params and no inference overhead.
"""
import torch
import torch.nn as nn
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model


class GatedQwen2Config(Qwen2Config):
    model_type = "gated_qwen2"

    def __init__(self, gated_layers=None, **kwargs):
        # gated_layers: indices of the INSERTED layers carrying zero-init gates.
        # Original (kept) layers are NOT wrapped and keep their full residual.
        self.gated_layers = list(gated_layers) if gated_layers else []
        super().__init__(**kwargs)


class GatedLinear(nn.Module):
    """Wraps a Linear so its output is scaled by a learnable scalar `gate`."""

    def __init__(self, base: nn.Linear, init: float = 0.0):
        super().__init__()
        self.base = base
        self.gate = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        return self.gate * self.base(x)


def install_gates(model, gated_layers):
    """Wrap o_proj/down_proj of the given decoder-layer indices with a 0-init gate."""
    for i in gated_layers:
        layer = model.layers[i]
        if not isinstance(layer.self_attn.o_proj, GatedLinear):
            layer.self_attn.o_proj = GatedLinear(layer.self_attn.o_proj, 0.0)
        if not isinstance(layer.mlp.down_proj, GatedLinear):
            layer.mlp.down_proj = GatedLinear(layer.mlp.down_proj, 0.0)


class GatedQwen2Model(Qwen2Model):
    config_class = GatedQwen2Config

    def __init__(self, config):
        super().__init__(config)
        install_gates(self, getattr(config, "gated_layers", []))


class GatedQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = GatedQwen2Config

    def __init__(self, config):
        super().__init__(config)
        install_gates(self.model, getattr(config, "gated_layers", []))
