"""Fold a gated checkpoint's AdamW moments to match the folded (plain) model.

Companion to fold_model.py. When you fold alpha/beta into the weights mid-training
(o_proj.weight <- alpha*o_proj.weight, down_proj.weight <- beta*down_proj.weight) and want to
CONTINUE training without a cold optimizer, the Adam moments must be carried across the fold:

  * the gate scalars disappear  -> drop their moments;
  * o_proj/down_proj are RE-PARAMETERISED  W' = alpha*W_base. The gradient in the new
    parameterisation is g' = g_base/alpha, so the moments transform as
        exp_avg    (m) ->  m / alpha
        exp_avg_sq (v) ->  v / alpha^2
    (and beta for down_proj). This makes the loaded moments consistent with the post-fold
    gradients. Adam's update is scale-invariant, so even without this rescale the FIRST step
    would be ~correct -- but rescaling keeps the EMA on the right scale through the transition.
  * every other parameter (q/k/v_proj, norms, embeddings, lm_head, and any UNgated original
    layer's o_proj/down_proj) is copied unchanged.

It writes `<folded_dir>/optimizer.pt` over the FOLDED model's [decay, no-decay] param groups,
so the existing `train.py --init-optimizer-from <folded_dir>` (momentum_transfer.transfer_moments)
picks it up by name and warm-starts the continued run.

Inputs / output:
    --source : INPUT  the gated checkpoint (has optimizer.pt + the gate values to rescale by)
    --folded : INPUT  the folded model dir from fold_model.py (read for its param structure)
               OUTPUT  optimizer.pt is WRITTEN into this same dir, ready for --init-optimizer-from

Usage:
    python fold_model.py     --source <gated_ckpt> --out    <folded_dir>   # 1) fold the weights
    python fold_optimizer.py --source <gated_ckpt> --folded <folded_dir>   # 2) fold the moments
    python train.py --model <folded_dir> --prior-steps <global_step> ... \
                    --init-optimizer-from <folded_dir>                      # 3) continue, moments warm

Assumes the gated run used --gate-lr-mult 1.0 (gates share the no-decay group) OR >1.0
(gates in a separate 3rd group); both are detected from optimizer.pt automatically.
"""
import argparse
import os
import re

import torch
from transformers import AutoModelForCausalLM, Qwen2ForCausalLM
from transformers.trainer_pt_utils import get_parameter_names

try:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
except Exception:                                            # pragma: no cover - older transformers
    from torch.nn import LayerNorm
    ALL_LAYERNORM_LAYERS = [LayerNorm]

GATE_EPS = 1e-8        # |gate| below this: skip transfer (folded weight ~0) -> that param starts fresh


def decay_names(model):
    """HF default decay set (2-D weights, excluding norms + biases), minus the .gate scalars --
    identical to train.py's get_decay_parameter_names."""
    names = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    return {n for n in names if "bias" not in n and not n.endswith(".gate")}


def _build_groups(model, n_groups):
    """Rebuild the param groups in the SAME order the trainer used, so optimizer.pt loads."""
    decay = decay_names(model)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    g0 = [p for n, p in named if n in decay]                          # decayed weights
    if n_groups == 3:                                                # gate-lr-mult > 1: gates split off
        g1 = [p for n, p in named if n not in decay and not n.endswith(".gate")]
        g2 = [p for n, p in named if n.endswith(".gate")]
        return [{"params": g0}, {"params": g1}, {"params": g2}]
    g1 = [p for n, p in named if n not in decay]                      # no-decay (incl. gates if mult==1)
    return [{"params": g0}, {"params": g1}]


@torch.no_grad()
def load_moments(gated_ckpt):
    """Return {param_name -> {exp_avg, exp_avg_sq, step}} for the gated checkpoint, plus the
    per-layer (alpha, beta) gate values."""
    opt_path = os.path.join(gated_ckpt, "optimizer.pt")
    if not os.path.exists(opt_path):
        raise SystemExit(f"no optimizer.pt in {gated_ckpt}")

    gm = AutoModelForCausalLM.from_pretrained(gated_ckpt, trust_remote_code=True,
                                              torch_dtype=torch.float32)
    gm.eval()
    sd = torch.load(opt_path, map_location="cpu")
    opt = torch.optim.AdamW(_build_groups(gm, len(sd["param_groups"])), lr=1e-3)
    opt.load_state_dict(sd)

    id2name = {id(p): n for n, p in gm.named_parameters()}
    M = {id2name[id(p)]: st for p, st in opt.state.items() if id(p) in id2name}

    gates = {}                                                       # layer idx -> (alpha, beta)
    for i, layer in enumerate(gm.model.layers):
        a = getattr(layer.self_attn.o_proj, "gate", None)
        b = getattr(layer.mlp.down_proj, "gate", None)
        gates[i] = (None if a is None else float(a),
                    None if b is None else float(b))
    del gm, opt
    return M, gates


@torch.no_grad()
def fold_optimizer(gated_ckpt, folded_dir):
    M, gates = load_moments(gated_ckpt)
    folded = Qwen2ForCausalLM.from_pretrained(folded_dir, torch_dtype=torch.float32)
    folded.eval()

    opt = torch.optim.AdamW(_build_groups(folded, 2), lr=1e-3)       # plain model: [decay, no-decay]
    layer_re = re.compile(r"model\.layers\.(\d+)\.")

    n_copy = n_scaled = n_skip = 0
    for name, p in folded.named_parameters():
        src_key, sm, sv = name, 1.0, 1.0                            # default: copy unchanged
        m = layer_re.match(name)
        if m and (name.endswith("self_attn.o_proj.weight") or name.endswith("mlp.down_proj.weight")):
            li = int(m.group(1))
            is_o = name.endswith("self_attn.o_proj.weight")
            gate = gates.get(li, (None, None))[0 if is_o else 1]
            if gate is not None:                                    # this layer was gated -> rescale
                if abs(gate) < GATE_EPS:
                    n_skip += 1
                    continue
                src_key = name.replace(".weight", ".base.weight")   # gated stored it under .base.weight
                sm, sv = 1.0 / gate, 1.0 / (gate * gate)

        st = M.get(src_key)
        if st is None or st["exp_avg"].shape != p.shape:
            continue
        opt.state[p] = {
            "step": (st["step"].clone() if torch.is_tensor(st["step"])
                     else torch.tensor(float(st["step"]))),
            "exp_avg": st["exp_avg"].clone().mul_(sm),
            "exp_avg_sq": st["exp_avg_sq"].clone().mul_(sv),
        }
        n_scaled += sm != 1.0
        n_copy += sm == 1.0

    out_path = os.path.join(folded_dir, "optimizer.pt")
    torch.save(opt.state_dict(), out_path)
    print(f"folded optimizer -> {out_path}")
    print(f"  copied {n_copy} params | rescaled {n_scaled} o_proj/down_proj | "
          f"skipped {n_skip} (|gate|<{GATE_EPS})")
    print(f"  resume with: train.py --model {folded_dir} --init-optimizer-from {folded_dir} ...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fold a gated checkpoint's Adam moments to match the folded plain model.")
    ap.add_argument("--source", required=True,
                    help="INPUT: gated checkpoint (its optimizer.pt + gate values)")
    ap.add_argument("--folded", required=True,
                    help="INPUT+OUTPUT: folded model dir from fold_model.py; optimizer.pt is written here")
    args = ap.parse_args()
    fold_optimizer(args.source, args.folded)
