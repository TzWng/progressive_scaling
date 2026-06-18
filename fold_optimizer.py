"""Fold a gated checkpoint's AdamW moments to match the folded (plain) model.

Companion to fold_model.py. When you fold alpha/beta into the weights mid-training
(o_proj.weight <- alpha*o_proj.weight, down_proj.weight <- beta*down_proj.weight) and want to
CONTINUE training without a cold optimizer, the Adam moments must be carried across the fold:

  * the gate scalars disappear  -> drop their moments;
  * o_proj/down_proj are RE-PARAMETERISED  W' = alpha*W_base. The gradient in the new
    parameterisation is g' = g_base/alpha, so the moments transform as
        exp_avg    (m) ->  m / alpha
        exp_avg_sq (v) ->  v / alpha^2
    (and beta for down_proj). This keeps the loaded moments on the right scale post-fold.
  * every other parameter (q/k/v_proj, norms, embeddings, lm_head, and any UNgated original
    layer's o_proj/down_proj) is copied unchanged.

It writes `<folded_dir>/optimizer.pt` over the FOLDED model's param groups so the existing
`train.py --init-optimizer-from <folded_dir>` (momentum_transfer.transfer_moments) picks it up
by name and warm-starts the continued run.

Mapping the saved (index-keyed) optimizer state back to parameter NAMES needs the trainer's
exact decay / no-decay split. We do NOT hand-roll that rule (it varies across transformers
versions); we call the SAME function train.py uses -- HF's Trainer.get_decay_parameter_names --
and apply train.py's only customisation: the .gate scalars are no-decay. The one structural
distinction this tool makes is therefore alpha/beta (gates) vs everything else.

Inputs / output:
    --source : INPUT  the gated checkpoint (its optimizer.pt + gate values)
    --folded : INPUT  the folded model dir from fold_model.py (read for its param structure)
               OUTPUT optimizer.pt is WRITTEN into this same dir, for --init-optimizer-from

Usage:
    python fold_model.py     --source <gated_ckpt> --out    <folded_dir>
    python fold_optimizer.py --source <gated_ckpt> --folded <folded_dir>
    python train.py --model <folded_dir> --prior-steps <global_step> ... \
                    --init-optimizer-from <folded_dir>
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
    """train.py's decay set, EXACTLY: the installed HF Trainer's get_decay_parameter_names, minus
    the .gate scalars. Calling the real function (instead of a hand-rolled rule) keeps us in
    lockstep with whatever split the trainer used to SAVE optimizer.pt, across versions. The only
    custom distinction is gates (alpha/beta) -> no-decay."""
    try:
        from transformers import Trainer
        base = list(Trainer.get_decay_parameter_names(None, model))   # self unused -> None is fine
    except Exception:                                                 # pragma: no cover - fallback
        base = [n for n in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "bias" not in n]
    return {n for n in base if not n.endswith(".gate")}


def _grouped_named(model, n_groups):
    """Params as (name, param) lists per group, in the SAME order the HF trainer builds them:
    [decay], [no-decay] (+ [gates] when the gated run used --gate-lr-mult > 1)."""
    decay = decay_names(model)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    g0 = [(n, p) for n, p in named if n in decay]
    if n_groups == 3:                                        # gate-lr-mult > 1: gates split off
        g1 = [(n, p) for n, p in named if n not in decay and not n.endswith(".gate")]
        g2 = [(n, p) for n, p in named if n.endswith(".gate")]
        return [g0, g1, g2]
    g1 = [(n, p) for n, p in named if n not in decay]        # no-decay (incl. gates if mult == 1)
    return [g0, g1]


@torch.no_grad()
def load_moments(gated_ckpt):
    """Return {param_name -> state} for the gated checkpoint + per-layer (alpha, beta) gates."""
    opt_path = os.path.join(gated_ckpt, "optimizer.pt")
    if not os.path.exists(opt_path):
        raise SystemExit(f"no optimizer.pt in {gated_ckpt}")

    gm = AutoModelForCausalLM.from_pretrained(gated_ckpt, trust_remote_code=True,
                                              torch_dtype=torch.float32)
    gm.eval()
    sd = torch.load(opt_path, map_location="cpu")
    sizes = [len(g["params"]) for g in sd["param_groups"]]
    groups = _grouped_named(gm, len(sizes))
    got = [len(g) for g in groups]
    if got != sizes:
        raise SystemExit(
            f"reconstructed param-group sizes {got} != saved {sizes}. The decay split could not be "
            f"reproduced from Trainer.get_decay_parameter_names -- check the transformers version.")
    flat = [x for g in groups for x in g]                    # index -> (name, param)
    M = {flat[i][0]: st for i, st in sd["state"].items() if i < len(flat)}

    gates = {}                                               # layer idx -> (alpha, beta)
    for i, layer in enumerate(gm.model.layers):
        a = getattr(layer.self_attn.o_proj, "gate", None)
        b = getattr(layer.mlp.down_proj, "gate", None)
        gates[i] = (None if a is None else float(a),
                    None if b is None else float(b))
    del gm
    return M, gates, sizes


@torch.no_grad()
def fold_optimizer(gated_ckpt, folded_dir):
    M, gates, sizes = load_moments(gated_ckpt)
    folded = Qwen2ForCausalLM.from_pretrained(folded_dir, torch_dtype=torch.float32)
    folded.eval()

    # Folded model has no gates -> plain [decay, no-decay]; same decay function transfer_moments uses.
    folded_groups = _grouped_named(folded, 2)
    opt = torch.optim.AdamW([{"params": [p for _, p in grp]} for grp in folded_groups], lr=1e-3)

    layer_re = re.compile(r"model\.layers\.(\d+)\.")
    n_copy = n_scaled = n_skip = 0
    for name, p in folded.named_parameters():
        src_key, sm, sv = name, 1.0, 1.0                     # default: copy unchanged
        m = layer_re.match(name)
        if m and (name.endswith("self_attn.o_proj.weight") or name.endswith("mlp.down_proj.weight")):
            li = int(m.group(1))
            is_o = name.endswith("self_attn.o_proj.weight")
            gate = gates.get(li, (None, None))[0 if is_o else 1]
            if gate is not None:                             # this layer was gated -> rescale
                if abs(gate) < GATE_EPS:
                    n_skip += 1
                    continue
                src_key = name.replace(".weight", ".base.weight")
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
        n_scaled += int(sm != 1.0)
        n_copy += int(sm == 1.0)

    out_path = os.path.join(folded_dir, "optimizer.pt")
    torch.save(opt.state_dict(), out_path)
    print(f"folded optimizer -> {out_path}")
    print(f"  source optimizer groups {sizes} | copied {n_copy} | "
          f"rescaled {n_scaled} o_proj/down_proj | skipped {n_skip} (|gate|<{GATE_EPS})")
    print(f"  resume: train.py --model {folded_dir} --init-optimizer-from {folded_dir} ...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fold a gated checkpoint's Adam moments to match the folded plain model.")
    ap.add_argument("--source", required=True,
                    help="INPUT: gated checkpoint (its optimizer.pt + gate values)")
    ap.add_argument("--folded", required=True,
                    help="INPUT+OUTPUT: folded model dir from fold_model.py; optimizer.pt written here")
    args = ap.parse_args()
    fold_optimizer(args.source, args.folded)
