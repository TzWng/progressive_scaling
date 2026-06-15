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


def make_lr_lambda(total_steps, warmup_steps, prior_steps, rewarmup_steps):
    """One warmup+cosine schedule spanning `total_steps` (GLOBAL horizon).

    Returns a function of the LOCAL step within the current phase. The global
    step is prior_steps + local_step, so every phase rides the SAME cosine:

      * baseline / phase-1 (prior_steps=0): standard warmup -> cosine to 0,
        optionally stopped early (see --stop-at-step / --growth-ratio).
      * grown phase-2 (prior_steps>0): a short linear re-warmup over
        `rewarmup_steps` ramps the LR back up to the cosine value at the resume
        point, then rejoins and follows the SAME global cosine down to 0.

    This keeps the baseline and the grown run on an identical LR trajectory
    (a fair comparison), differing only by the small re-warmup bump at growth.
    """
    def cos_factor(g):
        if g < warmup_steps:
            return g / max(1, warmup_steps)
        prog = (g - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, max(0.0, prog))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    def lr_lambda(local_step):
        g = prior_steps + local_step
        if rewarmup_steps > 0 and local_step < rewarmup_steps:
            return (local_step / rewarmup_steps) * cos_factor(prior_steps + rewarmup_steps)
        return cos_factor(g)

    return lr_lambda


class GrowTrainer(Trainer):
    """Trainer that installs the unified warmup+cosine schedule above."""

    def set_schedule(self, total_steps, warmup_steps, prior_steps, rewarmup_steps):
        self._sched = dict(total_steps=total_steps, warmup_steps=warmup_steps,
                           prior_steps=prior_steps, rewarmup_steps=rewarmup_steps)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            opt = self.optimizer if optimizer is None else optimizer
            s = self._sched
            total = s["total_steps"] if s["total_steps"] > 0 else s["prior_steps"] + num_training_steps
            self.lr_scheduler = LambdaLR(
                opt,
                make_lr_lambda(total, s["warmup_steps"], s["prior_steps"], s["rewarmup_steps"]),
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
                 prior_steps=0, prior_tokens=0, prior_flops=0.0):
        self.path = path
        self.n_params = n_params
        self.tokens_per_step = tokens_per_step
        self.prior_steps = prior_steps
        self.prior_tokens = prior_tokens
        self.prior_flops = prior_flops
        self.start = time.time()
        # Write header fresh unless we're appending to an existing, non-empty log.
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w") as f:
                f.write(self.HEADER)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs:  # skip eval/other log events
            return
        phase_tokens = state.global_step * self.tokens_per_step
        tokens = self.prior_tokens + phase_tokens
        flops = self.prior_flops + 6 * self.n_params * phase_tokens
        row = [
            self.prior_steps + state.global_step,
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
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="override epochs; -1 disables")
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--save-total-limit", type=int, default=3,
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
                   help="GLOBAL cosine horizon shared by baseline + grown run "
                        "(default: this phase's max-steps). MUST match across runs.")
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
    )
    use_ckpt = device == "cuda" and not args.no_grad_ckpt
    model.config.use_cache = False if use_ckpt else model.config.use_cache
    if use_ckpt:
        model.gradient_checkpointing_enable()

    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading tokenized dataset: {args.data}")
    ds = load_from_disk(args.data)
    ds.set_format(type="torch", columns=["input_ids", "labels"])

    # --- accounting for tokens / FLOPs logging ---
    seq_len = len(ds[0]["input_ids"])
    n_params = sum(p.numel() for p in model.parameters())
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_step = args.batch_size * args.grad_accum * seq_len * world_size
    print(f"Model layers: {model.config.num_hidden_layers} | params: {n_params/1e6:.1f}M | "
          f"seq_len: {seq_len} | tokens/step: {tokens_per_step}")

    # Resolve the GLOBAL cosine horizon and warmup (shared across baseline + grow).
    total_steps = args.total_steps if args.total_steps > 0 else (args.prior_steps + args.max_steps)
    warmup_steps = int(args.warmup_ratio * total_steps) if total_steps > 0 else 0
    stop_at = args.stop_at_step
    if args.growth_ratio is not None:
        stop_at = round(args.growth_ratio * total_steps)
    print(f"Schedule: total_steps(global)={total_steps} warmup={warmup_steps} "
          f"prior_steps={args.prior_steps} rewarmup={args.rewarmup_steps} stop_at={stop_at}")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=0,  # warmup handled by the custom unified scheduler
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
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
    )
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    callbacks = [MetricLogger(args.log_file, n_params, tokens_per_step,
                              args.prior_steps, args.prior_tokens, args.prior_flops)]
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
    trainer.set_schedule(total_steps, warmup_steps, args.prior_steps, args.rewarmup_steps)

    print(f"Logging metrics to: {args.log_file}")
    print("Starting pretraining...")
    resume = args.resume_from if args.resume_from else args.resume
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Done. Final model saved to: {args.out}")


if __name__ == "__main__":
    main()
