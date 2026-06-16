"""Grow a trained Qwen2 model DEEPER by interpolating layers (SVD geodesic).

Implements the method in layer_extension_algorithm.pdf:

  * insert m-1 new layers in each gap between real layers  ->  L' = m(L-1)+1,
    then copy-last to reach target_L (boundary layers we cannot interpolate);
  * each inserted layer's 7 weight matrices (q,k,v,o,gate,up,down) are rebuilt
    from the two neighbours: FRAMES (U,V) by geodesic (shortest-arc) interpolation,
    SPECTRUM (singular values) by log/geometric-mean interpolation; biases and
    RMSNorm scales by plain linear interpolation;
  * every ADDED layer (inserted + copy-last) is installed with zero-init alpha/beta
    residual gates (modeling_gated_qwen2.py) so the deep model is function-preserving
    at step 0 -- its initial loss equals the source model's loss exactly.

Usage:
    python grow_model.py --source $BASE/ckpt_grow_p1_2B/checkpoint-7500 \
                         --out    $BASE/model24_grown --m 2 --target-L 24
Then verify the printed step-0 loss gap is ~0 before training.

Deploy (fold gates away -> plain Qwen2, no extra params):
    python grow_model.py --fold $BASE/model24_grown --out $BASE/model24_folded
"""
import argparse
import math
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modeling_gated_qwen2 import (
    GatedQwen2Config,
    GatedQwen2ForCausalLM,
    GatedLinear,
)

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------
def geodesic(A, B, t):
    """Frame a fraction t along the shortest rotation from col(A) to col(B).

    A, B: (n, k) matrices with orthonormal columns (frames). Returns (n, k).
    This is the appendix `geodesic()` of the proposal, done in fp64 for safety.
    """
    A = A.double()
    B = B.double()
    X, c, Yt = torch.linalg.svd(A.T @ B)          # principal-angle cosines c
    th = torch.arccos(torch.clamp(c, 0.0, 1.0))   # principal angles
    A_aligned = A @ X
    sin_th = torch.sin(th)
    safe = torch.where(sin_th > 1e-7, sin_th, torch.full_like(sin_th, float("inf")))
    Q = (B @ Yt.T - A_aligned * c) / safe         # perpendicular directions
    G = A_aligned * torch.cos(t * th) + Q * torch.sin(t * th)
    return torch.linalg.qr(G)[0][:, : A.shape[1]]  # re-orthonormalize float drift


def interp_matrix(Wl, Wr, t):
    """Interpolate a single weight matrix: geodesic frames + log-mean spectrum."""
    Wl64, Wr64 = Wl.double(), Wr.double()
    Ul, sl, Vlh = torch.linalg.svd(Wl64, full_matrices=False)
    Ur, sr, Vrh = torch.linalg.svd(Wr64, full_matrices=False)
    U = geodesic(Ul, Ur, t)
    V = geodesic(Vlh.T, Vrh.T, t)
    s = torch.exp((1.0 - t) * torch.log(sl) + t * torch.log(sr))  # log/geom mean
    W = (U * s) @ V.T
    return W.to(Wl.dtype)


@torch.no_grad()
def interp_layer_state(sd_l, sd_r, t):
    """Build an inserted layer's state_dict from two neighbour state_dicts.

    2-D tensors (the 7 weight matrices) -> SVD interpolation.
    1-D tensors (q/k/v biases, RMSNorm scales) -> linear interpolation.
    """
    out = {}
    for k, vl in sd_l.items():
        vr = sd_r[k]
        if vl.ndim == 2:
            out[k] = interp_matrix(vl, vr, t)
        else:
            out[k] = (1.0 - t) * vl + t * vr
    return out


# ---------------------------------------------------------------------------
# Build the grown model
# ---------------------------------------------------------------------------
@torch.no_grad()
def grow(source_path, m, target_L):
    src = AutoModelForCausalLM.from_pretrained(source_path, torch_dtype=torch.float32)
    src.eval()
    L = src.config.num_hidden_layers
    inserted_per_gap = m - 1
    interp_L = m * (L - 1) + 1
    print(f"source layers L={L} | m={m} | interpolated L'={interp_L} | target={target_L}")
    if target_L < interp_L:
        raise SystemExit(f"target_L={target_L} < interpolated {interp_L}; raise target or lower m")

    # --- plan the new stack: (kind, payload) per new layer ---
    plan = []  # 'orig' -> source idx ; 'interp' -> (l, t) ; 'copy' -> source idx
    for l in range(L - 1):
        plan.append(("orig", l))
        for j in range(1, m):
            plan.append(("interp", (l, j / m)))
    plan.append(("orig", L - 1))
    while len(plan) < target_L:                      # copy-last boundary layers (Sec 4.5)
        plan.append(("copy", L - 1))
    assert len(plan) == target_L

    gated_layers = [i for i, (kind, _) in enumerate(plan) if kind != "orig"]
    print(f"added (gated, zero-init) layers: {gated_layers}")
    print(f"  interpolated: {[i for i,(k,_) in enumerate(plan) if k=='interp']}")
    print(f"  copy-last   : {[i for i,(k,_) in enumerate(plan) if k=='copy']}")

    # --- new gated config + empty model ---
    # Drop the special keys so GatedQwen2Config keeps model_type="gated_qwen2"
    # (otherwise the source's "qwen2" would override it and break auto_map routing).
    cfg_dict = src.config.to_dict()
    for k in ("model_type", "architectures", "auto_map",
              "transformers_version", "_name_or_path"):
        cfg_dict.pop(k, None)
    cfg = GatedQwen2Config(**cfg_dict)
    cfg.num_hidden_layers = target_L
    cfg.gated_layers = gated_layers
    cfg.architectures = ["GatedQwen2ForCausalLM"]
    cfg.auto_map = {
        "AutoConfig": "modeling_gated_qwen2.GatedQwen2Config",
        "AutoModelForCausalLM": "modeling_gated_qwen2.GatedQwen2ForCausalLM",
    }
    new = GatedQwen2ForCausalLM(cfg)
    new.eval()

    # embeddings / final norm / lm_head straight from source
    new.model.embed_tokens.load_state_dict(src.model.embed_tokens.state_dict())
    new.model.norm.load_state_dict(src.model.norm.state_dict())
    if not cfg.tie_word_embeddings:
        new.lm_head.load_state_dict(src.lm_head.state_dict())
    new.tie_weights()

    src_layers = src.model.layers
    for i, (kind, payload) in enumerate(plan):
        dst = new.model.layers[i]
        if kind in ("orig", "copy"):
            src_sd = src_layers[payload].state_dict()
            _load_layer(dst, src_sd)            # exact weights; gate (if any) stays 0
        else:
            l, t = payload
            sd = interp_layer_state(src_layers[l].state_dict(),
                                    src_layers[l + 1].state_dict(), t)
            _load_layer(dst, sd)
    return src, new, plan


def _load_layer(dst_layer, plain_sd):
    """Load a *plain* (ungated) layer state_dict into dst, which may have
    GatedLinear-wrapped o_proj/down_proj. Remaps `<x>.weight` -> `<x>.base.weight`
    for the wrapped projections; leaves the gate parameter untouched (init 0)."""
    target = dict(dst_layer.state_dict())
    remapped = {}
    for k, v in plain_sd.items():
        if k not in target and k.replace(".weight", ".base.weight") in target:
            remapped[k.replace(".weight", ".base.weight")] = v
        else:
            remapped[k] = v
    # keep existing gate params (not present in plain_sd) at their init value
    for k, v in target.items():
        remapped.setdefault(k, v)
    dst_layer.load_state_dict(remapped, strict=True)


# ---------------------------------------------------------------------------
# Function-preservation check
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_preservation(src, new, vocab, seq=64, bsz=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (bsz, seq), generator=g)
    src_loss = src(input_ids=ids, labels=ids).loss.item()
    new_loss = new(input_ids=ids, labels=ids).loss.item()
    print(f"\n[function preservation] source loss = {src_loss:.6f} | "
          f"grown loss = {new_loss:.6f} | gap = {abs(src_loss-new_loss):.2e}")
    if abs(src_loss - new_loss) > 1e-3:
        print("  WARNING: gap is not ~0 -- gates/wiring are wrong, DO NOT train yet.")
    else:
        print("  OK: deep model computes the same function at step 0.")


# ---------------------------------------------------------------------------
# Deploy: fold gates -> plain Qwen2
# ---------------------------------------------------------------------------
@torch.no_grad()
def fold_gates(grown_path, out_path):
    """Fold every learned alpha/beta into its projection weight and emit a
    standard Qwen2ForCausalLM (no gates, no extra params, no remote code)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    model = AutoModelForCausalLM.from_pretrained(grown_path, trust_remote_code=True,
                                                 torch_dtype=torch.float32)
    for layer in model.model.layers:
        for proj_name in ("o_proj", "down_proj"):
            mod = getattr(layer.self_attn if proj_name == "o_proj" else layer.mlp, proj_name)
            if isinstance(mod, GatedLinear):
                mod.base.weight.mul_(mod.gate)          # exact: alpha * W
                setattr(layer.self_attn if proj_name == "o_proj" else layer.mlp,
                        proj_name, mod.base)
    cfg = Qwen2Config(**{k: v for k, v in model.config.to_dict().items()
                         if k not in ("gated_layers", "auto_map", "architectures")})
    cfg.architectures = ["Qwen2ForCausalLM"]
    plain = Qwen2ForCausalLM(cfg)
    plain.load_state_dict(model.state_dict(), strict=True)
    os.makedirs(out_path, exist_ok=True)
    plain.save_pretrained(out_path)
    AutoTokenizer.from_pretrained(grown_path).save_pretrained(out_path)
    print(f"folded plain Qwen2 saved to {out_path}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", help="trained L-layer model dir / checkpoint to grow from")
    p.add_argument("--out", required=True, help="output dir for the grown (or folded) model")
    p.add_argument("--m", type=int, default=2, help="subdivision factor (m=2 inserts 1 per gap)")
    p.add_argument("--target-L", type=int, default=24, help="final number of layers")
    p.add_argument("--fold", default=None,
                   help="deploy mode: path to a trained grown model to fold gates into a plain Qwen2")
    args = p.parse_args()

    if args.fold:
        fold_gates(args.fold, args.out)
        return

    src, new, plan = grow(args.source, args.m, args.target_L)
    check_preservation(src, new, vocab=new.config.vocab_size)

    os.makedirs(args.out, exist_ok=True)
    new.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.source).save_pretrained(args.out)
    shutil.copy(os.path.join(HERE, "modeling_gated_qwen2.py"),
                os.path.join(args.out, "modeling_gated_qwen2.py"))
    print(f"\ngrown model saved to {args.out}  (load with trust_remote_code=True)")


if __name__ == "__main__":
    main()
