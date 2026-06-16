"""Warm-start the grown model's AdamW with the source (small) model's Adam moments.

For every grown parameter that is an EXACT copy of a source parameter -- the kept original
layers (detected by exact weight equality), plus embeddings / final norm / lm_head -- copy
the source's (exp_avg, exp_avg_sq, step) into the grown optimizer's state. Interpolated and
copy-last layers, which have no exact source counterpart, are left fresh (zero moments).

This preserves the training dynamics of the reused weights (cf. Staged Training), so the
grown model does not pay the cold-Adam slowdown on its original layers after growth.

Robustness: the whole thing is best-effort. The caller wraps it in try/except, and it prints
how many parameters / layers were matched so the transfer can be verified before trusting any
result. A wrong mapping would corrupt training silently, so DO check the printed counts.
"""
import os
import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def transfer_moments(grown_model, optimizer, source_ckpt, decay_fn):
    opt_path = os.path.join(source_ckpt, "optimizer.pt")
    if not os.path.exists(opt_path):
        print(f"[momentum transfer] no optimizer.pt in {source_ckpt}; skipping (fresh optimizer).")
        return

    src = AutoModelForCausalLM.from_pretrained(source_ckpt, torch_dtype=torch.float32)
    src.eval()

    # Rebuild the source optimizer with the SAME param grouping (decay first, then no-decay),
    # load its saved state, then read moments keyed by parameter NAME. PyTorch's
    # load_state_dict maps the saved integer indices onto these params by group order, so as
    # long as the grouping matches training, each moment lands on the right parameter.
    decay = set(decay_fn(src))
    g0 = [p for n, p in src.named_parameters() if n in decay and p.requires_grad]
    g1 = [p for n, p in src.named_parameters() if n not in decay and p.requires_grad]
    src_opt = torch.optim.AdamW([{"params": g0}, {"params": g1}], lr=1e-3)
    src_opt.load_state_dict(torch.load(opt_path, map_location="cpu"))
    id2name = {id(p): n for n, p in src.named_parameters()}
    name_moments = {id2name[id(p)]: st for p, st in src_opt.state.items() if id(p) in id2name}

    # Match each grown decoder layer to a source layer by EXACT q_proj weight equality
    # (kept/copy layers are byte-exact copies; interpolated layers match nothing).
    src_layers = src.model.layers

    def match_layer(gl):
        gw = gl.self_attn.q_proj.weight.data.cpu()
        for j, sl in enumerate(src_layers):
            sw = sl.self_attn.q_proj.weight.data.cpu()
            if gw.shape == sw.shape and torch.equal(gw, sw):
                return j
        return None

    grown_to_src = {}
    for shared in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        if shared in name_moments:
            grown_to_src[shared] = shared
    n_layers = 0
    for i, gl in enumerate(grown_model.model.layers):
        j = match_layer(gl)
        if j is None:
            continue
        n_layers += 1
        for sub, _ in gl.named_parameters():
            gn, sn = f"model.layers.{i}.{sub}", f"model.layers.{j}.{sub}"
            if sn in name_moments:
                grown_to_src[gn] = sn

    # Inject the moments into the grown optimizer's state (before the first step).
    grown_named = dict(grown_model.named_parameters())
    n = 0
    for gn, sn in grown_to_src.items():
        gp = grown_named.get(gn)
        st = name_moments.get(sn)
        if gp is None or st is None or st["exp_avg"].shape != gp.shape:
            continue
        step = st.get("step", 0)
        step = step.clone() if torch.is_tensor(step) else torch.tensor(float(step))
        optimizer.state[gp] = {
            "step": step.to(gp.device),
            "exp_avg": st["exp_avg"].to(gp.device).clone(),
            "exp_avg_sq": st["exp_avg_sq"].to(gp.device).clone(),
        }
        n += 1
    print(f"[momentum transfer] injected Adam moments for {n} params across {n_layers} "
          f"matched (kept) layers; interpolated layers left fresh.  source={source_ckpt}")
    del src, src_opt
