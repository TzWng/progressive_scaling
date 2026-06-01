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
)


class LossLogger(TrainerCallback):
    """Append training metrics to a tab-separated file for later plotting.

    Columns: step  epoch  loss  lr  grad_norm  seconds
    Load with: pandas.read_csv(path, sep="\\t")
    """

    def __init__(self, path):
        self.path = path
        self.start = time.time()
        # Write header fresh unless we're resuming onto an existing log.
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("step\tepoch\tloss\tlr\tgrad_norm\tseconds\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs:  # skip eval/other log events
            return
        row = [
            state.global_step,
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
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="disable gradient checkpointing (faster; use when VRAM is ample, e.g. 0.5B on H100)")
    p.add_argument("--log-file", default="log.txt",
                   help="tab-separated file recording step/loss/lr for plotting")
    args = p.parse_args()

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
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=default_data_collator,
        processing_class=tok,
        callbacks=[LossLogger(args.log_file)],
    )

    print(f"Logging loss to: {args.log_file}")
    print("Starting pretraining...")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Done. Final model saved to: {args.out}")


if __name__ == "__main__":
    main()
