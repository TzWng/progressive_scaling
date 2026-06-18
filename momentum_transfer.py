"""Warm-start the grown model's AdamW with the source (small) model's Adam moments.

- KEPT / copy-last layers (detected by exact weight equality) + embeddings / norm / lm_head:
  copy the source moments directly.
- INSERTED (interpolated) layers, when warm_inserted=True: give each inserted layer moments
  interpolated from its two neighbour source layers at the SAME fraction t used for its
  weights -- second moment v (exp_avg_sq, positive) by log/geometric interp (matching the
  spectrum), first moment m (exp_avg, signed) by linear interp. Handles MULTIPLE inserts per
  gap: a run of c inserts between two kept layers gets t = 1/(c+1), 2/(c+1), ..., c/(c+1).

The plan is reconstructed purely from weight matching: kept/copy layers match a source layer
exactly; inserted layers match nothing, and a maximal run of unmatched layers sits between two
kept layers whose source indices are its interpolation neighbours.

Note: for the gated model, inserted layers have gate=0 so their gradients (hence moments) are
moot until the gate opens -- warm_inserted mainly helps the no-gate (direct insert) variant.
"""
import os
import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def transfer_moments(grown_model, optimizer, source_ckpt, decay_fn, warm_inserted=False):
    opt_path = os.path.join(source_ckpt, "optimizer.pt")
    if not os.path.exists(opt_path):
        print(f"[momentum transfer] no optimizer.pt in {source_ckpt}; skipping.")
        return

    src = AutoModelForCausalLM.from_pretrained(source_ckpt, torch_dtype=torch.float32)
    src.eval()
    decay = set(decay_fn(src))
    g0 = [p for n, p in src.named_parameters() if n in decay and p.requires_grad]
    g1 = [p for n, p in src.named_parameters() if n not in decay and p.requires_grad]
    src_opt = torch.optim.AdamW([{"params": g0}, {"params": g1}], lr=1e-3)
    src_opt.load_state_dict(torch.load(opt_path, map_location="cpu"))
    id2name = {id(p): n for n, p in src.named_parameters()}
    M = {id2name[id(p)]: st for p, st in src_opt.state.items() if id(p) in id2name}

    grown_named = dict(grown_model.named_parameters())
    src_layers = src.model.layers

    def plain(sub):                                   # gated o_proj.base.weight -> o_proj.weight
        return sub.replace(".base.weight", ".weight")

    def set_state(gn, m, v, step):
        gp = grown_named.get(gn)
        if gp is None or v.shape != gp.shape:
            return 0
        st = step.clone() if torch.is_tensor(step) else torch.tensor(float(step))
        optimizer.state[gp] = {"step": st.to(gp.device),
                               "exp_avg": m.to(gp.device).clone(),
                               "exp_avg_sq": v.to(gp.device).clone()}
        return 1

    def match(gl):                                    # grown layer -> source idx (exact) or None
        gw = gl.self_attn.q_proj.weight.data.cpu()
        for j, sl in enumerate(src_layers):
            sw = sl.self_attn.q_proj.weight.data.cpu()
            if gw.shape == sw.shape and torch.equal(gw, sw):
                return j
        return None

    n = 0
    for shared in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        st = M.get(shared)
        if st:
            n += set_state(shared, st["exp_avg"], st["exp_avg_sq"], st["step"])

    layers = grown_model.model.layers
    matched = [match(gl) for gl in layers]
    n_kept = n_ins = 0
    i = 0
    while i < len(layers):
        if matched[i] is not None:                    # kept / copy-last: copy source moments
            l = matched[i]
            for sub, _ in layers[i].named_parameters():
                st = M.get(f"model.layers.{l}.{plain(sub)}")
                if st:
                    n += set_state(f"model.layers.{i}.{sub}", st["exp_avg"], st["exp_avg_sq"], st["step"])
            n_kept += 1
            i += 1
        else:                                         # a run of inserted layers between two kept
            j = i
            while j < len(layers) and matched[j] is None:
                j += 1
            la = matched[i - 1] if i > 0 else None     # left/right kept source indices
            lb = matched[j] if j < len(layers) else None  # (run may reach the model's end)
            c = j - i                                  # inserts in this gap
            if warm_inserted and la is not None and lb is not None:
                for k in range(c):
                    gi, t = i + k, (k + 1) / (c + 1)   # same t as the weight interpolation
                    for sub, _ in layers[gi].named_parameters():
                        ps = plain(sub)
                        sl = M.get(f"model.layers.{la}.{ps}")
                        sr = M.get(f"model.layers.{lb}.{ps}")
                        if sl is None or sr is None:
                            continue
                        v = torch.exp((1 - t) * torch.log(sl["exp_avg_sq"].clamp_min(1e-30))
                                      + t * torch.log(sr["exp_avg_sq"].clamp_min(1e-30)))
                        m = (1 - t) * sl["exp_avg"] + t * sr["exp_avg"]
                        n += set_state(f"model.layers.{gi}.{sub}", m, v, sl["step"])
                    n_ins += 1
            i = j

    print(f"[momentum transfer] set {n} params | kept layers={n_kept} | "
          f"warmed inserted layers={n_ins} (warm_inserted={warm_inserted})")
    del src, src_opt
