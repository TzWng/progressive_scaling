"""From-scratch causal-LM pretraining for the initialized Qwen2.5 model.

Device-agnostic: runs on CUDA (H20), Apple MPS, or CPU. bf16 is enabled
automatically on CUDA. Supports checkpointing and resume.

Typical flow:
    python init_model.py
    python prepare_data.py --tokenizer ./model_init
    python train.py

Quick local smoke test (Mac, sub-minute):
    python init_model.py --tiny
    python prepare_data.py --tokenizer ./model_init --max-samples 2000 --seq-len 512
    python train.py --max-steps 20 --batch-size 2

Resume:
    python train.py --resume
"""
import argparse
import json
import math
import os
import time
import types

import torch
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


def make_lr_lambda(total_steps, warmup_steps, prior_steps, rewarmup_steps,
                   min_lr_ratio=0.0, flops_scale=1.0, const_lr=False, decay_frac=0.0):
    """One warmup+cosine schedule indexed by COMPUTE (not raw steps).

    If const_lr=True: warmup -> CONSTANT peak LR (optionally a WSD linear decay
    over the last `decay_frac` of total_steps). This drops all compute/cosine
    dependence -- every phase sits at the same LR everywhere, so flops_scale and
    the prior-steps cosine position are irrelevant (prior_steps then only offsets
    the log + data skip). Much simpler to compare growth vs baseline.

    `total_steps`, `warmup_steps`, `prior_steps` are all measured in
    LARGE-MODEL-EQUIVALENT steps (= cumulative FLOPs / (6 * N_ref * tok/step)),
    so baseline and grown runs ride ONE identical learning-rate-vs-FLOPs curve.

    The only difference from a plain step-cosine is `flops_scale = N_phase /
    N_ref`: a phase advances the cosine by flops_scale per LOCAL step, i.e. in
    proportion to the FLOPs it actually burns.

      * baseline / grown phase-2 (the large model): flops_scale = 1.0, so a
        local step == a large-equivalent step -- behaves exactly like before.
      * small phase-1 (e.g. 6L with N_ref = the 36L target): flops_scale < 1,
        so the cosine (warmup included) decays ~1/flops_scale times slower per
        step -- the small model sits at the FLOPs-correct (higher) LR, and its
        LR at the growth FLOPs equals the large model's LR at that same FLOPs.

    `prior_steps` (compute-equiv) places a grown phase on the global cosine;
    `rewarmup_steps` is an optional uncounted re-warmup up to the resume LR.
    """
    def cos_factor(g):
        if g < warmup_steps:
            return g / max(1, warmup_steps)
        prog = (g - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, max(0.0, prog))
        # cosine from 1.0 down to min_lr_ratio (the floor), not to 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

    def const_factor(g):
        # Warmup -> CONSTANT peak LR. No compute dependence, so baseline and grown
        # runs sit at the same LR at every FLOPs (no flops_scale needed). Optional
        # WSD tail: linear 1.0 -> min_lr_ratio over the LAST decay_frac of total_steps.
        if g < warmup_steps:
            return g / max(1, warmup_steps)
        if decay_frac > 0.0 and total_steps > 0:
            decay_start = total_steps * (1.0 - decay_frac)
            if g > decay_start:
                p = min(1.0, max(0.0, (g - decay_start) / max(1, total_steps - decay_start)))
                return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - p)
        return 1.0

    factor = const_factor if const_lr else cos_factor
    scale = 1.0 if const_lr else flops_scale   # const LR needs no FLOPs alignment

    def lr_lambda(local_step):
        if rewarmup_steps > 0 and local_step < rewarmup_steps:
            # UNCOUNTED warm-up preamble: ramp 0 -> the LR AT the resume point
            # (prior_steps). For const LR that target is the peak (1.0).
            return (local_step / rewarmup_steps) * factor(prior_steps)
        # cosine: advance by flops_scale per local step (compute-indexed).
        # const : scale=1, position only decides warmup / WSD-tail membership.
        eff = local_step - rewarmup_steps
        return factor(prior_steps + scale * eff)

    return lr_lambda


class GrowTrainer(Trainer):
    """Trainer that installs the unified warmup+cosine schedule above."""

    def set_schedule(self, total_steps, warmup_steps, prior_steps, rewarmup_steps,
                     min_lr_ratio=0.0, flops_scale=1.0, const_lr=False, decay_frac=0.0):
        self._sched = dict(total_steps=total_steps, warmup_steps=warmup_steps,
                           prior_steps=prior_steps, rewarmup_steps=rewarmup_steps,
                           min_lr_ratio=min_lr_ratio, flops_scale=flops_scale,
                           const_lr=const_lr, decay_frac=decay_frac)

    def get_decay_parameter_names(self, model):
        # Keep weight decay OFF for the zero-init residual gates (alpha/beta):
        # decaying a gate pulls it back toward 0 and fights it opening, which is
        # exactly the dynamic the grown model relies on. Everything else keeps the
        # Trainer's default decay set (which already excludes biases + norms).
        names = super().get_decay_parameter_names(model)
        return [n for n in names if not n.endswith(".gate")]

    def _get_train_sampler(self, *args, **kwargs):
        # Phase-2 continuation feeds the already globally-permuted dataset IN ORDER
        # (no per-epoch reshuffle), so the grown run lines up sample-for-sample with
        # the baseline from prior_steps onward. Phase-1 itself read epoch-0 in this
        # same permutation order, so this is a faithful continuation.
        if getattr(self, "_sequential_data", False):
            from torch.utils.data import SequentialSampler
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            opt = self.optimizer if optimizer is None else optimizer
            s = self._sched
            total = s["total_steps"] if s["total_steps"] > 0 else s["prior_steps"] + num_training_steps
            self.lr_scheduler = LambdaLR(
                opt,
                make_lr_lambda(total, s["warmup_steps"], s["prior_steps"], s["rewarmup_steps"],
                               s["min_lr_ratio"], s["flops_scale"],
                               s["const_lr"], s["decay_frac"]),
            )
        return self.lr_scheduler

    def create_optimizer(self):
        # Wrap the optimizer so a step with any non-finite gradient is skipped
        # instead of poisoning the weights with NaN/Inf. From-scratch bf16
        # pretraining occasionally hits a degenerate packed block whose backward
        # produces a non-finite grad; skipping that single update keeps training
        # alive and is symmetric across baseline/grown runs (same data+seed ->
        # same block skipped at the same step), so the comparison stays fair.
        opt = super().create_optimizer()

        # Optional: put the alpha/beta gates in their OWN param group with a higher LR so
        # they open faster (gate_lr = base_lr * gate_lr_mult). The cosine scheduler scales
        # every group by the same factor, so this just raises the gates' base LR.
        # gate_lr_mult == 1.0 -> skip entirely -> identical to the original single-LR setup.
        mult = getattr(self, "_gate_lr_mult", 1.0)
        if mult != 1.0:
            gate_ids = {id(p) for n, p in self.model.named_parameters() if n.endswith(".gate")}
            gates = []
            for g in opt.param_groups:
                gates += [p for p in g["params"] if id(p) in gate_ids]
                g["params"] = [p for p in g["params"] if id(p) not in gate_ids]
            if gates:
                opt.add_param_group({"params": gates, "weight_decay": 0.0,
                                     "lr": self.args.learning_rate * mult})
                print(f"[gate-lr] {len(gates)} gate params in a separate group, "
                      f"lr = {mult}x base = {self.args.learning_rate * mult:.2e}")

        self._skipped_nan_steps = 0
        trainer = self
        # Bind as a real method (via MethodType) rather than assigning a plain
        # function -- torch's LambdaLR patches opt.step and expects a bound
        # method (reads .__func__), which a plain function lacks.
        orig_step = type(opt).step

        def safe_step(optimizer, *a, **k):
            finite = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for group in optimizer.param_groups for p in group["params"]
            )
            if not finite:
                trainer._skipped_nan_steps += 1
                print(f"[skip] non-finite grad at global_step "
                      f"{trainer.state.global_step} (total skipped: {trainer._skipped_nan_steps})")
                return None
            return orig_step(optimizer, *a, **k)

        opt.step = types.MethodType(safe_step, opt)

        # Optional: warm-start the moments of the KEPT layers from the source checkpoint's
        # optimizer (Staged-Training style) to avoid the cold-Adam slowdown after growth.
        src = getattr(self, "_init_opt_from", None)
        if src:
            try:
                from momentum_transfer import transfer_moments
                transfer_moments(self.model, opt, src, self.get_decay_parameter_names,
                                 warm_inserted=getattr(self, "_warm_inserted", False))
            except Exception as e:
                print(f"[momentum transfer] FAILED ({e}); continuing with fresh optimizer.")
        return opt


class StopAtStep(TrainerCallback):
    """Stop training at a global step BELOW max_steps, leaving the cosine horizon
    (= max_steps/total_steps) intact. Used to end phase-1 at the growth point."""

    def __init__(self, stop_step):
        self.stop_step = stop_step

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_step > 0 and state.global_step >= self.stop_step:
            control.should_training_stop = True
            # Force a full, resumable checkpoint at the exact stop step even when
            # it isn't a multiple of save_steps (e.g. stop_at=7500, save_steps=1000).
            control.should_save = True
        return control


class MetricLogger(TrainerCallback):
    """Append training metrics to a tab-separated file for later plotting.

    Columns: step  tokens  flops  epoch  loss  lr  grad_norm  seconds
    Load with: pandas.read_csv(path, sep="\\t")

    `step/tokens/flops` include prior-phase offsets, so a grown run's phase-2 log
    continues phase-1 as ONE curve. FLOPs use the standard ~6*N*tokens estimate
    (N = current-phase param count), which is what model-growth papers plot
    loss against for a FLOPs-matched fair comparison.
    """

    HEADER = "step\ttokens\tflops\tepoch\tloss\tlr\tgrad_norm\tseconds\n"

    def __init__(self, path, n_params, tokens_per_step,
                 prior_steps=0, prior_tokens=0, prior_flops=0.0, rewarmup_steps=0):
        self.path = path
        self.n_params = n_params
        self.tokens_per_step = tokens_per_step
        self.prior_steps = prior_steps
        self.prior_tokens = prior_tokens
        self.prior_flops = prior_flops
        self.rewarmup_steps = rewarmup_steps
        self.start = time.time()
        # Write header fresh unless we're appending to an existing, non-empty log.
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w") as f:
                f.write(self.HEADER)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs:  # skip eval/other log events
            return
        # Exclude the uncounted warm-up preamble: effective step 0 == prior_steps, so a
        # grown run's curve continues the baseline at prior_steps with no warm-up gap.
        eff = state.global_step - self.rewarmup_steps
        if eff <= 0:
            return  # warm-up preamble (eff<0) + the prior_steps anchor (eff==0, already
            #         the prior phase's last logged row) -> first grown row is prior_steps+1 on
        phase_tokens = eff * self.tokens_per_step
        tokens = self.prior_tokens + phase_tokens
        flops = self.prior_flops + 6 * self.n_params * phase_tokens
        row = [
            self.prior_steps + eff,
            tokens,
            f"{flops:.6e}",
            round(logs.get("epoch", 0.0), 4),
            logs.get("loss", ""),
            logs.get("learning_rate", ""),
            logs.get("grad_norm", ""),
            round(time.time() - self.start, 1),
        ]
        with open(self.path, "a") as f:
            f.write("\t".join(str(x) for x in row) + "\n")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="./model_init")
    p.add_argument("--data", default="./data_tokenized")
    p.add_argument("--out", default="./checkpoints")
    p.add_argument("--batch-size", type=int, default=16, help="per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.02)
    p.add_argument("--min-lr-ratio", type=float, default=0.0,
                   help="cosine floors at this fraction of peak LR instead of decaying to 0 "
                        "(e.g. 0.1 = G_stack's min-LR = 10%% of max-LR). 0.0 = decay to 0.")
    p.add_argument("--const-lr", action="store_true",
                   help="use a CONSTANT LR after warmup (no cosine). Removes all FLOPs/cosine "
                        "alignment: every phase sits at the same LR everywhere, so --ref-params / "
                        "prior-steps no longer affect the LR (they only offset log + data skip).")
    p.add_argument("--decay-frac", type=float, default=0.0,
                   help="with --const-lr: WSD-style linear decay (1.0 -> min-lr-ratio) over the "
                        "LAST this-fraction of --total-steps. 0 = pure constant. Use >0 only on the "
                        "FINAL run (not phase-1) if you want an end anneal.")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="override epochs; -1 disables")
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--save-total-limit", type=int, default=None,
                   help="max checkpoints to keep (default: keep ALL -- needed when each "
                        "checkpoint is archived/uploaded separately)")
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from", default=None,
                   help="path to a SPECIFIC checkpoint dir to resume from "
                        "(overrides --resume's 'latest' behavior). HF Trainer skips the "
                        "already-consumed batches, so training continues in the same data "
                        "order from this step -- no data reuse.")
    p.add_argument("--seed", type=int, default=42,
                   help="fixed seed for reproducible weight init + data order (keep IDENTICAL "
                        "across baseline and grown runs for a fair comparison)")
    p.add_argument("--fp32", action="store_true",
                   help="disable bf16 mixed precision; train fully in fp32 "
                        "(slower but numerically robust -- use to diagnose/avoid NaN)")
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="disable gradient checkpointing (faster; use when VRAM is ample, e.g. 0.5B on H100)")
    p.add_argument("--log-file", default="log.txt",
                   help="tab-separated file recording step/tokens/flops/loss/lr for plotting")
    # --- fair-comparison schedule controls ---
    p.add_argument("--total-steps", type=int, default=-1,
                   help="GLOBAL cosine horizon in LARGE-MODEL-EQUIVALENT steps, shared by "
                        "baseline + grown run (default: this phase's max-steps). MUST match across runs.")
    p.add_argument("--ref-params", type=float, default=None,
                   help="param count of the LARGE/target model, used to set flops_scale = "
                        "n_params/ref_params so the cosine decays by FLOPs. Leave UNSET for the "
                        "large model (baseline / grown phase-2) -> flops_scale=1. For the SMALL "
                        "phase-1 model, set it to the large model's n_params (e.g. 581876992 for 36L) "
                        "so the 6L rides the FLOPs-correct (slower) cosine.")
    p.add_argument("--growth-ratio", type=float, default=None,
                   help="phase-1 only: stop the small model at this fraction of --total-steps "
                        "(the ablation knob, e.g. 0.25). Cosine stays sized to the full horizon.")
    p.add_argument("--stop-at-step", type=int, default=-1,
                   help="phase-1 only: explicit global step to stop at (alternative to --growth-ratio)")
    p.add_argument("--rewarmup-steps", type=int, default=0,
                   help="grown phase-2: linear re-warmup length right after growth")
    # --- offsets so a grown run's phase-2 log continues phase-1 as ONE curve ---
    p.add_argument("--prior-steps", type=int, default=0,
                   help="optimizer steps already done before this phase (grown-run continuation)")
    p.add_argument("--init-optimizer-from", default=None,
                   help="warm-start AdamW moments from this source checkpoint's optimizer.pt: "
                        "kept/original layers + embeddings get the source moments, inserted layers "
                        "stay fresh. Mitigates the cold-optimizer slowdown after growth.")
    p.add_argument("--warm-inserted", action="store_true",
                   help="with --init-optimizer-from: also warm the INSERTED layers by interpolating "
                        "their two neighbours' moments at the same t (mainly useful for --no-gate).")
    p.add_argument("--gate-lr-mult", type=float, default=1.0,
                   help="LR multiplier for the alpha/beta gate params (separate optimizer group). "
                        "1.0 = identical to the original single-LR setup; >1 opens gates faster.")
    p.add_argument("--skip-samples", type=int, default=0,
                   help="phase-2: number of samples the prior phase consumed (overrides the "
                        "auto value prior_steps*batch_size*grad_accum). Normally leave 0 and just "
                        "pass --prior-steps; the grown run then continues the data stream in order, "
                        "warm-up re-reading the prior phase's last rewarmup_steps steps.")
    p.add_argument("--prior-tokens", type=int, default=0,
                   help="tokens already consumed before this phase")
    p.add_argument("--prior-flops", type=float, default=0.0,
                   help="FLOPs already spent before this phase (e.g. small-model phase)")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device()
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported() and not args.fp32
    print(f"Device: {device} | bf16: {use_bf16}")

    # Prefer flash-attn if installed; otherwise fall back to PyTorch SDPA
    # (which has its own fused flash kernel on CUDA -- no compilation needed).
    attn_impl = "sdpa"
    if device == "cuda":
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            print("flash-attn not installed; using PyTorch SDPA backend instead.")

    print(f"Loading initialized model: {args.model} (attn={attn_impl})")
    # IMPORTANT: load the model in fp32 even when training in bf16. The Trainer's
    # bf16=True does autocast for the forward/backward (fast bf16 compute) while
    # keeping fp32 MASTER WEIGHTS for the optimizer. Loading the weights directly
    # in bf16 means AdamW updates bf16 params with no fp32 copy, which is
    # numerically fragile and can diverge to NaN on an unlucky batch.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation=attn_impl,
        trust_remote_code=True,  # load the grown model's custom gated architecture
    )
    use_ckpt = device == "cuda" and not args.no_grad_ckpt
    model.config.use_cache = False if use_ckpt else model.config.use_cache
    if use_ckpt:
        model.gradient_checkpointing_enable()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading tokenized dataset: {args.data}")
    # --data may be a COMMA-SEPARATED list of packed dirs; concatenate them at LOAD
    # time (memory-mapped, no new copy on disk). Order is fixed by the string, so
    # len(ds) + the seeded permutation stay identical across runs -> skip-samples
    # continuation remains valid. Pass the SAME order everywhere.
    _paths = [p for p in args.data.split(",") if p]
    _parts = [load_from_disk(p) for p in _paths]
    if len(_parts) == 1:
        ds = _parts[0]
    else:
        from datasets import concatenate_datasets
        ds = concatenate_datasets(_parts)
        print(f"Concatenated {len(_parts)} datasets at load (no disk copy): "
              f"{[len(x) for x in _parts]} -> {len(ds)} blocks")
    # phase-2 continuation (prior_steps>0): line the grown run up with the baseline
    # sample-for-sample. Phase-1 read epoch-0 in the seeded permutation order
    # randperm(n, seed); we rebuild that permutation and feed the slice IN ORDER
    # (SequentialSampler, see GrowTrainer). The counted stream starts exactly at the
    # prior phase's stopping sample (prior_steps), while the UNCOUNTED warm-up
    # re-reads the prior phase's last `rewarmup_steps` steps -> zero misalignment.
    sequential_data = False
    if args.prior_steps > 0 or args.skip_samples > 0:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        samples_per_step = args.batch_size * args.grad_accum * ws
        prior_samples = args.skip_samples if args.skip_samples > 0 else args.prior_steps * samples_per_step
        warmup_samples = args.rewarmup_steps * samples_per_step
        start = max(0, prior_samples - warmup_samples)
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(ds), generator=g).tolist()
        ds = ds.select(perm[start:])
        sequential_data = True
        eff_total = args.total_steps if args.total_steps > 0 else (args.prior_steps + args.max_steps)
        need = samples_per_step * (args.rewarmup_steps + max(0, eff_total - args.prior_steps))
        print(f"Continuing data IN ORDER from sample {start} "
              f"(prior {prior_samples} - warmup {warmup_samples}); warm-up re-reads the prior "
              f"phase's last {args.rewarmup_steps} steps, then the COUNTED stream starts at the "
              f"prior stopping sample {prior_samples}. {len(ds)} samples available (need {need}).")
    ds.set_format(type="torch", columns=["input_ids", "labels"])

    # --- accounting for tokens / FLOPs logging ---
    seq_len = len(ds[0]["input_ids"])
    n_params = sum(p.numel() for p in model.parameters())
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_step = args.batch_size * args.grad_accum * seq_len * world_size
    print(f"Model layers: {model.config.num_hidden_layers} | params: {n_params/1e6:.1f}M | "
          f"seq_len: {seq_len} | tokens/step: {tokens_per_step}")

    # flops_scale = N_phase / N_ref makes the cosine decay by FLOPs. The large model
    # (baseline / grown phase-2) has ref unset -> scale 1.0. The small phase-1 model
    # passes --ref-params = the large model's n_params -> scale < 1, so its cosine
    # advances in proportion to the (smaller) FLOPs it burns per step.
    ref_params = args.ref_params if args.ref_params else n_params
    flops_scale = n_params / ref_params
    print(f"FLOPs-indexed schedule: n_params={n_params} ref_params={int(ref_params)} "
          f"flops_scale={flops_scale:.4f}  (1.0 = large model; <1 = small phase-1)")

    # Resolve the GLOBAL cosine horizon and warmup (shared across baseline + grow), in
    # LARGE-EQUIVALENT steps. For a from-scratch small phase-1 (prior_steps=0) the cosine
    # spans total_steps large-equiv steps, but the phase only runs max_steps REAL steps;
    # at real step s it sits at large-equiv g = flops_scale*s.
    total_steps = args.total_steps if args.total_steps > 0 else (args.prior_steps + args.max_steps)
    warmup_steps = int(args.warmup_ratio * total_steps) if total_steps > 0 else 0
    stop_at = args.stop_at_step
    if args.growth_ratio is not None:
        stop_at = round(args.growth_ratio * total_steps)
    sched_kind = (f"CONST-LR (decay_frac={args.decay_frac})" if args.const_lr
                  else f"cosine (min_lr_ratio={args.min_lr_ratio})")
    print(f"Schedule: {sched_kind} | total_steps(global)={total_steps} warmup={warmup_steps} "
          f"prior_steps={args.prior_steps} rewarmup={args.rewarmup_steps} stop_at={stop_at}")

    # For a grown phase, the optimizer-step budget is the UNCOUNTED warm-up preamble
    # plus the remaining cosine (prior_steps -> total_steps). Compute it so the run ends
    # exactly at the baseline horizon and the warm-up adds no counted steps.
    phase_max_steps = args.max_steps
    if args.prior_steps > 0 and total_steps > 0:
        phase_max_steps = args.rewarmup_steps + (total_steps - args.prior_steps)
        print(f"Grown phase: max_steps={phase_max_steps} "
              f"(rewarmup {args.rewarmup_steps} + remaining {total_steps - args.prior_steps}); "
              f"warm-up is not counted in the logged step.")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=0,  # warmup handled by the custom unified scheduler
        num_train_epochs=args.epochs,
        max_steps=phase_max_steps,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=use_bf16,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        dataloader_num_workers=2,
        seed=args.seed,
        data_seed=args.seed,
    )

    # Persist the full run definition so the baseline and grown runs are
    # reproducible and provably comparable (same data, seed, hyperparameters).
    os.makedirs(args.out, exist_ok=True)
    run_cfg = dict(
        model=args.model, data=args.data, seed=args.seed,
        num_hidden_layers=model.config.num_hidden_layers, n_params=n_params,
        seq_len=seq_len, batch_size=args.batch_size, grad_accum=args.grad_accum,
        tokens_per_step=tokens_per_step, world_size=world_size,
        lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        lr_scheduler="cosine", max_steps=args.max_steps, total_steps=total_steps,
        warmup_steps=warmup_steps, epochs=args.epochs,
        growth_ratio=args.growth_ratio, stop_at_step=stop_at,
        rewarmup_steps=args.rewarmup_steps,
        prior_steps=args.prior_steps, prior_tokens=args.prior_tokens,
        prior_flops=args.prior_flops,
        ref_params=int(ref_params), flops_scale=flops_scale,
        min_lr_ratio=args.min_lr_ratio, const_lr=args.const_lr, decay_frac=args.decay_frac,
    )
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    callbacks = [MetricLogger(args.log_file, n_params, tokens_per_step,
                              args.prior_steps, args.prior_tokens, args.prior_flops,
                              rewarmup_steps=args.rewarmup_steps)]
    if stop_at and stop_at > 0:
        callbacks.append(StopAtStep(stop_at))

    trainer = GrowTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=default_data_collator,
        processing_class=tok,
        callbacks=callbacks,
    )
    trainer.set_schedule(total_steps, warmup_steps, args.prior_steps, args.rewarmup_steps,
                         min_lr_ratio=args.min_lr_ratio, flops_scale=flops_scale,
                         const_lr=args.const_lr, decay_frac=args.decay_frac)
    trainer._sequential_data = sequential_data  # feed the permuted slice in order (phase-2)
    trainer._init_opt_from = args.init_optimizer_from  # optional Adam-moment warm-start
    trainer._warm_inserted = args.warm_inserted        # also warm inserted layers' moments
    trainer._gate_lr_mult = args.gate_lr_mult          # separate (higher) LR for alpha/beta gates

    print(f"Logging metrics to: {args.log_file}")
    print("Starting pretraining...")
    resume = args.resume_from if args.resume_from else args.resume
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Done. Final model saved to: {args.out}")


if __name__ == "__main__":
    main()
