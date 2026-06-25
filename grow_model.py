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


def slerp_cols(A, B, t):
    """Spherical interpolation of each PAIRED column A[:,i] -> B[:,i] by fraction t.
    A, B: (n, k) with unit columns. Returns (n, k) with re-normalized unit columns."""
    dots = (A * B).sum(0).clamp(-1.0, 1.0)         # cos angle per column
    th = torch.arccos(dots)
    sin = torch.sin(th)
    small = sin < 1e-6
    wa = torch.where(small, torch.full_like(th, 1.0 - t), torch.sin((1.0 - t) * th) / sin)
    wb = torch.where(small, torch.full_like(th, t),       torch.sin(t * th) / sin)
    G = A * wa + B * wb
    return G / G.norm(dim=0, keepdim=True).clamp_min(1e-12)


def _interp_geodesic(Wl, Wr, t):
    """DEFAULT: independent Stiefel geodesics on U and V (the original method)."""
    Wl64, Wr64 = Wl.double(), Wr.double()
    Ul, sl, Vlh = torch.linalg.svd(Wl64, full_matrices=False)
    Ur, sr, Vrh = torch.linalg.svd(Wr64, full_matrices=False)
    U = geodesic(Ul, Ur, t)
    V = geodesic(Vlh.T, Vrh.T, t)
    s = torch.exp((1.0 - t) * torch.log(sl) + t * torch.log(sr))   # log / geometric mean
    return ((U * s) @ V.T).to(Wl.dtype)


def _interp_paired(Wl, Wr, t):
    """OPT-IN (--uv-align): slerp PAIRED singular triplets, keeping U and V CONSISTENT.

    Triplets are paired by singular-value order and each triplet's joint sign is aligned
    (flip u_i and v_i together), so the reassembly reduces to Wl at t=0 and Wr at t=1 --
    endpoint-consistent, unlike _interp_geodesic which rotates U and V separately and
    scrambles the U<->V correspondence (right spectrum, random singular directions)."""
    Ul, sl, Vlh = torch.linalg.svd(Wl.double(), full_matrices=False)
    Ur, sr, Vrh = torch.linalg.svd(Wr.double(), full_matrices=False)
    Vl, Vr = Vlh.T, Vrh.T
    sign = torch.sign((Ul * Ur).sum(0))            # align each triplet's sign (u & v together)
    sign[sign == 0] = 1.0
    Ur, Vr = Ur * sign, Vr * sign
    U = slerp_cols(Ul, Ur, t)
    V = slerp_cols(Vl, Vr, t)
    s = torch.exp((1.0 - t) * torch.log(sl) + t * torch.log(sr))
    return ((U * s) @ V.T).to(Wl.dtype)


def interp_matrix(Wl, Wr, t, uv_align=False):
    """Default = original geodesic; uv_align=True = U,V-consistent paired slerp."""
    return _interp_paired(Wl, Wr, t) if uv_align else _interp_geodesic(Wl, Wr, t)


@torch.no_grad()
def interp_layer_state(sd_l, sd_r, t, uv_align=False):
    """Build an inserted layer's state_dict from two neighbour state_dicts.

    2-D tensors (the 7 weight matrices) -> SVD interpolation.
    1-D tensors (q/k/v biases, RMSNorm scales) -> linear interpolation.
    """
    out = {}
    for k, vl in sd_l.items():
        vr = sd_r[k]
        if vl.ndim == 2:
            out[k] = interp_matrix(vl, vr, t, uv_align=uv_align)
        else:
            out[k] = (1.0 - t) * vl + t * vr
    return out


# ---------------------------------------------------------------------------
# Build the grown model
# ---------------------------------------------------------------------------
@torch.no_grad()
def grow(source_path, total_insert, per_gap, gated=True, uv_align=False):
    """Insert `total_insert` new layers, up to `per_gap` (>=2) per gap, filling gaps from
    the LAST gap (between layers L-2 and L-1) backward toward the front."""
    if per_gap < 1:
        raise SystemExit(f"per_gap={per_gap} must be >= 1")
    src = AutoModelForCausalLM.from_pretrained(source_path, torch_dtype=torch.float32)
    src.eval()
    L = src.config.num_hidden_layers

    # Interpolation can fill at most (L-1) gaps x per_gap layers. Anything beyond that is
    # handled at the BOUNDARY by copy-last layers -- you cannot interpolate past the last
    # layer (Finding 3), so the overflow is copies of it.
    capacity = (L - 1) * per_gap
    n_interp = min(total_insert, capacity)
    n_copy = total_insert - n_interp

    # Distribute the interpolated inserts EVENLY across the (L-1) gaps; the remainder
    # goes to the BACK gaps (deeper gaps get one extra). Stays <= per_gap because
    # n_interp <= capacity = (L-1)*per_gap. E.g. 12 over 5 gaps -> [2,2,2,3,3].
    n_gaps = L - 1
    base, rem = divmod(n_interp, n_gaps)
    fill = {l: base + (1 if l >= n_gaps - rem else 0) for l in range(n_gaps)}
    fill = {l: c for l, c in fill.items() if c > 0}   # drop empty gaps

    # A gap with c inserts is subdivided into c+1 equal steps: t_j = j/(c+1), j=1..c.
    plan = []  # 'orig' -> source idx ; 'interp' -> (l, t) ; 'copy' -> source idx
    for l in range(L - 1):
        plan.append(("orig", l))
        c = fill.get(l, 0)
        for j in range(1, c + 1):
            plan.append(("interp", (l, j / (c + 1))))
    plan.append(("orig", L - 1))
    for _ in range(n_copy):                          # copy-last boundary layers (Finding 3)
        plan.append(("copy", L - 1))
    target_L = L + total_insert
    assert len(plan) == target_L
    print(f"L={L} -> {target_L} | insert {total_insert}: {n_interp} interpolated "
          f"(<= {per_gap}/gap, from back) + {n_copy} copy-last")
    print(f"  gaps filled (gap between l and l+1 -> #inserts): {dict(sorted(fill.items()))}")
    if n_copy:
        print(f"  NOTE: interpolation capacity {capacity} exceeded; {n_copy} copy-last appended.")

    gated_layers = [i for i, (kind, _) in enumerate(plan) if kind != "orig"] if gated else []
    print(f"added layers ({'gated zero-init' if gated else 'DIRECT, no gate'}): "
          f"{[i for i, (k, _) in enumerate(plan) if k != 'orig']}")
    print(f"  interpolated: {[i for i,(k,_) in enumerate(plan) if k=='interp']}")
    print(f"  copy-last   : {[i for i,(k,_) in enumerate(plan) if k=='copy']}")

    # --- new config + empty model ---
    # Drop the special keys so a custom model_type isn't overridden by the source's,
    # and so per-layer fields (layer_types) regenerate at the new depth.
    cfg_dict = src.config.to_dict()
    for k in ("model_type", "architectures", "auto_map",
              "transformers_version", "_name_or_path", "layer_types"):
        cfg_dict.pop(k, None)
    cfg_dict["num_hidden_layers"] = target_L
    if gated:
        cfg = GatedQwen2Config(**cfg_dict)
        cfg.gated_layers = gated_layers
        cfg.architectures = ["GatedQwen2ForCausalLM"]
        cfg.auto_map = {
            "AutoConfig": "modeling_gated_qwen2.GatedQwen2Config",
            "AutoModelForCausalLM": "modeling_gated_qwen2.GatedQwen2ForCausalLM",
        }
        new = GatedQwen2ForCausalLM(cfg)
    else:
        # direct insert (Sec 4.4 ablation): a plain Qwen2, inserted layers active from
        # step 0 -- no function preservation, but the new capacity engages immediately.
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(**cfg_dict)
        cfg.architectures = ["Qwen2ForCausalLM"]
        new = Qwen2ForCausalLM(cfg)
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
                                    src_layers[l + 1].state_dict(), t, uv_align=uv_align)
            _load_layer(dst, sd)
    return src, new, plan


@torch.no_grad()
def stack_grow(source_path, total_insert):
    """G_stack baseline (Du et al. 2024): plain depthwise stacking -- the grown layer i takes
    the weights of source layer (i mod L). Pure copy-paste, no interpolation, no gates; the
    deeper model is the L-layer block tiled. This is ONLY the core G_stack init operator (for
    a fair comparison), not their LR / growth-timing recipe."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    src = AutoModelForCausalLM.from_pretrained(source_path, torch_dtype=torch.float32)
    src.eval()
    L = src.config.num_hidden_layers
    target_L = L + total_insert
    cfg_dict = src.config.to_dict()
    for k in ("model_type", "architectures", "auto_map",
              "transformers_version", "_name_or_path", "layer_types"):
        cfg_dict.pop(k, None)
    cfg_dict["num_hidden_layers"] = target_L
    cfg = Qwen2Config(**cfg_dict)
    cfg.architectures = ["Qwen2ForCausalLM"]
    new = Qwen2ForCausalLM(cfg)
    new.eval()
    new.model.embed_tokens.load_state_dict(src.model.embed_tokens.state_dict())
    new.model.norm.load_state_dict(src.model.norm.state_dict())
    if not cfg.tie_word_embeddings:
        new.lm_head.load_state_dict(src.lm_head.state_dict())
    new.tie_weights()
    for i in range(target_L):
        new.model.layers[i].load_state_dict(src.model.layers[i % L].state_dict())
    print(f"G_stack: L={L} -> {target_L} | grown layer i <- source layer (i mod {L})")
    print(f"  layer mapping: {[i % L for i in range(target_L)]}")
    return src, new, None


@torch.no_grad()
def gstack_scale_grow(source_path, total_insert, eps=1.0, cap=3.0, skip_first=1):
    """G_stack-scale (SVD): G_stack skeleton (FRAMES U,V copied verbatim -- the only Finding-3-safe
    thing to do with the unpredictable rotation DIRECTION) + an EXTRAPOLATED spectrum. We SVD every
    base layer, fit EACH singular value's own log-trend slope b_j across the base block (Finding 2:
    the spectrum's scale -- and a small amount of its shape -- drift smoothly/monotonically with
    depth, the PREDICTABLE part), then extrapolate across tiles: tile k's singular values become
    s_j * exp(eps * b_j * L * k), capped at +/- `cap` in log so they cannot blow up. Reassemble with
    the copied frames. Frames untouched -> no frame prediction (Finding-3 safe); eps=0 -> G_stack."""
    src = AutoModelForCausalLM.from_pretrained(source_path, torch_dtype=torch.float32).eval()
    L = src.config.num_hidden_layers
    if L - skip_first < 2:
        raise SystemExit(f"gstack-scale needs >= 2 layers after skip_first={skip_first} to fit the trend")
    target_L = L + total_insert
    sds = [src.model.layers[l].state_dict() for l in range(L)]
    keys2d = [k for k, v in sds[0].items() if v.ndim == 2]
    ls = torch.arange(L, dtype=torch.float64)
    fl = ls[skip_first:]                                                        # layers used to fit the trend

    # SVD every base layer per key; fit per-singular-value log-trend slope b_j (skipping early layers)
    spec = {}
    for k in keys2d:
        Us, Ss, Vs = [], [], []
        for l in range(L):
            U, S, Vh = torch.linalg.svd(sds[l][k].double(), full_matrices=False)
            Us.append(U); Ss.append(S); Vs.append(Vh.T)
        logS = torch.stack([s.clamp_min(1e-12).log() for s in Ss])             # [L, r]
        fS = logS[skip_first:]                                                 # drop early boundary layers
        b = (((fl - fl.mean())[:, None] * (fS - fS.mean(0))).sum(0)
             / ((fl - fl.mean()) ** 2).sum())                                  # [r] per-component slope
        spec[k] = (Us, Ss, Vs, b)

    # plain Qwen2, G_stack skeleton, no gates
    cfg_dict = src.config.to_dict()
    for kk in ("model_type", "architectures", "auto_map",
               "transformers_version", "_name_or_path", "layer_types"):
        cfg_dict.pop(kk, None)
    cfg_dict["num_hidden_layers"] = target_L
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(**cfg_dict); cfg.architectures = ["Qwen2ForCausalLM"]
    new = Qwen2ForCausalLM(cfg); new.eval()
    new.model.embed_tokens.load_state_dict(src.model.embed_tokens.state_dict())
    new.model.norm.load_state_dict(src.model.norm.state_dict())
    if not cfg.tie_word_embeddings:
        new.lm_head.load_state_dict(src.lm_head.state_dict())
    new.tie_weights()

    Lint = L - skip_first                                       # interior block (layers skip_first..L-1)
    for p in range(target_L):
        if p < skip_first:                                      # keep early boundary layers verbatim (once)
            new.model.layers[p].load_state_dict(
                {k: v.clone() for k, v in sds[p].items()}, strict=True)
            continue
        j = p - skip_first
        i, tile = skip_first + (j % Lint), j // Lint            # interior source layer, tile index
        out = {}
        for key, v in sds[i].items():
            if v.ndim == 2 and tile > 0 and eps != 0.0:         # tile 0 / eps=0 = plain (skip-first) stacking
                Us, Ss, Vs, b = spec[key]
                expo = torch.clamp(eps * b * Lint * tile, -cap, cap)   # cap per-component drift
                s_new = Ss[i] * torch.exp(expo)                        # extrapolate each singular value
                out[key] = ((Us[i] * s_new) @ Vs[i].T).to(v.dtype)
            else:
                out[key] = v.clone()
        new.model.layers[p].load_state_dict(out, strict=True)

    print(f"G_stack-scale (SVD): L={L} -> {target_L} | keep first {skip_first} layer(s) verbatim, "
          f"tile interior layers {skip_first}..{L-1} + per-singular-value scale extrapolation, eps={eps}")
    print("  mean log-scale slope b per weight: "
          + ", ".join(f"{k.split('.')[-1]}={spec[k][3].mean().item():+.4f}" for k in keys2d))
    return src, new, None


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
def check_preservation(src, new, vocab, seq=64, bsz=2, seed=0, expect_zero=True):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (bsz, seq), generator=g)
    src_loss = src(input_ids=ids, labels=ids).loss.item()
    new_loss = new(input_ids=ids, labels=ids).loss.item()
    print(f"\n[init check] source loss = {src_loss:.6f} | "
          f"grown loss = {new_loss:.6f} | gap = {abs(src_loss-new_loss):.2e}")
    if expect_zero:
        if abs(src_loss - new_loss) > 1e-3:
            print("  WARNING: gap is not ~0 -- gates/wiring are wrong, DO NOT train yet.")
        else:
            print("  OK: gated deep model computes the same function at step 0.")
    else:
        print("  direct insert: a small gap is EXPECTED (no function preservation); the inserted "
              "layers are active from step 0. Lower gap = better interpolated init.")


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
    p.add_argument("--total-insert", type=int, default=12, help="total number of layers to insert")
    p.add_argument("--per-gap", type=int, default=2, help="inserts per gap (>=2); gaps filled from the back")
    p.add_argument("--fold", default=None,
                   help="deploy mode: path to a trained grown model to fold gates into a plain Qwen2")
    p.add_argument("--no-gate", action="store_true",
                   help="direct insert (Sec 4.4): inserted layers active from step 0, no alpha/beta "
                        "gates, no function preservation -- emits a plain Qwen2. Use to test whether "
                        "the zero-init gates are stalling and the new capacity never engages.")
    p.add_argument("--uv-align", action="store_true",
                   help="use the U,V-consistent paired-slerp interpolation (reconstructs endpoints) "
                        "instead of the default independent-geodesic interpolation. OFF by default.")
    p.add_argument("--gstack", action="store_true",
                   help="G_stack baseline: plain depthwise stacking (copy-paste, layer i <- i mod L), "
                        "no interpolation/gates. For a fair comparison against the interpolation method.")
    p.add_argument("--gstack-scale", action="store_true",
                   help="G_stack-scale: G_stack skeleton (frames copied verbatim -- Finding-3-safe) "
                        "+ an EXTRAPOLATED monotonic spectral scale (Finding 2: scale is the predictable "
                        "part, so it can be extended). Multiplies tile k by exp(eps*b*L*k) so the overall "
                        "magnitude climbs monotonically across tiles instead of resetting. Plain Qwen2, "
                        "no gates. eps=0 reproduces G_stack exactly.")
    p.add_argument("--eps", type=float, default=1.0,
                   help="G_stack-scale strength: tile k scaled by exp(eps*b*L*k) along the fitted "
                        "log-scale slope b (0 = exact G_stack; 1 = full trend extrapolation).")
    p.add_argument("--skip-first", type=int, default=1,
                   help="G_stack-scale: number of early boundary layers to EXCLUDE when fitting the "
                        "log-scale trend (layer 0 is embedding-adjacent and off-trend). Default 1.")
    args = p.parse_args()

    if args.fold:
        fold_gates(args.fold, args.out)
        return

    gated = not args.no_gate
    if args.gstack:
        gated = False
        src, new, plan = stack_grow(args.source, args.total_insert)
    elif args.gstack_scale:
        gated = False
        src, new, plan = gstack_scale_grow(args.source, args.total_insert,
                                           eps=args.eps, skip_first=args.skip_first)
    else:
        src, new, plan = grow(args.source, args.total_insert, args.per_gap,
                              gated=gated, uv_align=args.uv_align)
    check_preservation(src, new, vocab=new.config.vocab_size, expect_zero=gated)

    os.makedirs(args.out, exist_ok=True)
    new.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.source).save_pretrained(args.out)
    if gated:
        shutil.copy(os.path.join(HERE, "modeling_gated_qwen2.py"),
                    os.path.join(args.out, "modeling_gated_qwen2.py"))
        print(f"\ngrown (gated) model saved to {args.out}  (load with trust_remote_code=True)")
    else:
        print(f"\ngrown (direct, no-gate) model saved to {args.out}  (plain Qwen2)")


if __name__ == "__main__":
    main()
