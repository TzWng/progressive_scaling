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
import os
import time

import torch
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
        # Write header fresh unless we're appending to an existing log.
        if not os.path.exists(path):
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
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42,
                   help="fixed seed for reproducible weight init + data order (keep IDENTICAL "
                        "across baseline and grown runs for a fair comparison)")
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="disable gradient checkpointing (faster; use when VRAM is ample, e.g. 0.5B on H100)")
    p.add_argument("--log-file", default="log.txt",
                   help="tab-separated file recording step/tokens/flops/loss/lr for plotting")
    # --- offsets for continuing a grown run as ONE continuous curve (phase 2) ---
    p.add_argument("--prior-steps", type=int, default=0,
                   help="optimizer steps already done before this phase (grown-run continuation)")
    p.add_argument("--prior-tokens", type=int, default=0,
                   help="tokens already consumed before this phase")
    p.add_argument("--prior-flops", type=float, default=0.0,
                   help="FLOPs already spent before this phase (e.g. small-model phase)")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device()
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
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

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=use_bf16,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
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
        lr_scheduler="cosine", max_steps=args.max_steps, epochs=args.epochs,
        prior_steps=args.prior_steps, prior_tokens=args.prior_tokens,
        prior_flops=args.prior_flops,
    )
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=default_data_collator,
        processing_class=tok,
        callbacks=[MetricLogger(args.log_file, n_params, tokens_per_step,
                                args.prior_steps, args.prior_tokens, args.prior_flops)],
    )

    print(f"Logging metrics to: {args.log_file}")
    print("Starting pretraining...")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Done. Final model saved to: {args.out}")


if __name__ == "__main__":
    main()
