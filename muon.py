"""Single-device Muon (Keller Jordan) + auxiliary AdamW, in ONE optimizer.

Muon orthogonalizes the momentum update via a 5-step Newton-Schulz iteration, so
every update matrix is (approximately) orthogonal -> it flattens the singular-value
spectrum of the weights (higher effective rank). Only 2D "hidden" matrices go
through Muon; embeddings / lm_head / norms / biases / scalar gates stay on AdamW
(the `use_muon=False` groups).

Param groups passed to the optimizer each carry a `use_muon` flag:
  {"params": [...], "use_muon": True,  "lr": 0.02, "momentum": 0.95, "weight_decay": wd}
  {"params": [...], "use_muon": False, "lr": 3e-4, "betas": (0.9,0.95), "eps": 1e-10, "weight_decay": wd}
A single cosine LambdaLR scales EVERY group's base lr by the same factor, so the
Muon (0.02) and Adam (3e-4) groups decay proportionally -- exactly what we want.
"""
import torch


def zeropower_via_newtonschulz5(G, steps: int = 5):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients are tuned so the singular values of
    the result land in ~[0.7, 1.3] (good enough -- we don't need exact orthogonality).
    Runs in bfloat16 for speed; stable because the iteration is self-correcting.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Normalize so the top singular value is <= 1 before iterating.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _muon_update(grad, momentum_buf, beta=0.95, ns_steps=5, nesterov=True):
    momentum_buf.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buf, beta) if nesterov else momentum_buf
    if update.ndim == 4:  # conv filters -> flatten to 2D
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # Scale so the update's RMS matches AdamW's ~unit-RMS step regardless of shape.
    update *= max(1.0, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def _adam_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    bias1 = 1 - betas[0] ** step
    bias2 = 1 - betas[1] ** step
    return (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for `use_muon=True` groups, decoupled-weight-decay AdamW for the rest."""

    def __init__(self, param_groups):
        for g in param_groups:
            assert "use_muon" in g, "each param group must set use_muon=True/False"
            if g["use_muon"]:
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
                g.setdefault("ns_steps", 5)
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-10)
                g.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    upd = _muon_update(p.grad, state["momentum_buffer"],
                                       beta=group["momentum"], ns_steps=group["ns_steps"])
                    if wd != 0:
                        p.mul_(1 - lr * wd)          # decoupled weight decay
                    p.add_(upd.reshape(p.shape), alpha=-lr)
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    upd = _adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                       state["step"], group["betas"], group["eps"])
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(upd, alpha=-lr)
        return loss
