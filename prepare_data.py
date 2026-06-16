"""Download + tokenize + pack a pretraining corpus into fixed-length blocks.

Default dataset is wikitext-103 (English, small, no auth, fast) -- perfect for
validating the from-scratch pretrain pipeline. Other presets are provided for
when you scale up.

Presets (--dataset):
    fineweb-edu     HuggingFaceFW/fineweb-edu sample-10BT (English, high quality)  [RECOMMENDED]
    dclm            mlfoundations/dclm-baseline-1.0 (English, strong filtering)
    fineweb         HuggingFaceFW/fineweb sample-10BT (English, big, lower filter)
    chinese-fineweb opencsg/chinese-fineweb-edu (Chinese quality)
    skypile         Skywork/SkyPile-150B (Chinese, very large)
    wikitext        Salesforce/wikitext  (English, ~100M tokens, no-auth fast test)
    tinystories     roneneldan/TinyStories (tiny, fastest smoke test)

For progressive/depth-growth (G_stack-style) training, use the SAME corpus
across all growth stages -- growth is a schedule, not a data change. fineweb-edu
is the recommended starting corpus.

Usage:
    python prepare_data.py --tokenizer ./model_init --out ./data_tokenized
    python prepare_data.py --dataset chinese-fineweb --max-samples 50000
    python prepare_data.py --seq-len 1024 --max-samples 20000   # quick validation
"""
import argparse
import os

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

PRESETS = {
    "fineweb-edu": dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", text="text", stream=True),
    "dclm": dict(path="mlfoundations/dclm-baseline-1.0", name=None, split="train", text="text", stream=True),
    "fineweb": dict(path="HuggingFaceFW/fineweb", name="sample-10BT", split="train", text="text", stream=True),
    "chinese-fineweb": dict(path="opencsg/chinese-fineweb-edu", name=None, split="train", text="text", stream=True),
    "skypile": dict(path="Skywork/SkyPile-150B", name=None, split="train", text="text", stream=True),
    "slimpajama": dict(path="cerebras/SlimPajama-627B", name=None, split="train", text="text", stream=True),
    "slimpajama-6b": dict(path="DKYoon/SlimPajama-6B", name=None, split="train", text="text"),
    "wikitext": dict(path="Salesforce/wikitext", name="wikitext-103-raw-v1", split="train", text="text"),
    "tinystories": dict(path="roneneldan/TinyStories", name=None, split="train", text="text"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fineweb-edu", choices=list(PRESETS))
    p.add_argument("--local-files", default=None,
                   help="glob to local parquet/json files (e.g. './fineweb_raw/**/*.parquet'). "
                        "Bypasses all HF networking -- use on flaky/offline servers.")
    p.add_argument("--text-col", default="text",
                   help="text column name when using --local-files")
    p.add_argument("--tokenizer", default="./model_init")
    p.add_argument("--out", default="./data_tokenized")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=50000,
                   help="cap raw docs pulled (keeps validation runs fast); -1 for all")
    p.add_argument("--num-proc", type=int, default=4,
                   help="parallel processes for the TOKENIZE stage")
    p.add_argument("--pack-num-proc", type=int, default=1,
                   help="parallel processes for the PACK stage. Keep at 1 on large "
                        "corpora -- multi-proc packing tends to OOM/crash a worker.")
    args = p.parse_args()

    seq_len = args.seq_len
    # Intermediate cache of the tokenized (pre-pack) dataset. Saving it before the
    # pack stage means a crash during packing never wastes the (slow) tokenization:
    # re-running just reloads this and re-packs.
    tok_dir = args.out.rstrip("/") + ".tokenized"

    if os.path.exists(tok_dir):
        print(f"Found cached tokenized dataset -- skipping download+tokenize: {tok_dir}")
        tokenized = load_from_disk(tok_dir)
    else:
        if args.local_files:
            # Load straight from local files -- no HF Hub calls at all.
            ext = "parquet" if ".parquet" in args.local_files else "json"
            print(f"Loading local {ext} files: {args.local_files}")
            ds = load_dataset(ext, data_files=args.local_files, split="train")
            spec = {"text": args.text_col}
            streaming = False
        else:
            spec = PRESETS[args.dataset]
            streaming = spec.get("stream", False)
            print(f"Loading dataset: {spec['path']} ({spec.get('name')}) streaming={streaming}")
            ds = load_dataset(spec["path"], spec["name"], split=spec["split"], streaming=streaming)

        # Cap the number of raw documents so validation runs stay quick.
        if args.max_samples and args.max_samples > 0:
            if streaming:
                ds = ds.take(args.max_samples)
                from datasets import Dataset
                ds = Dataset.from_list(list(ds))
            else:
                n = min(args.max_samples, len(ds))
                ds = ds.select(range(n))
        print(f"Raw documents: {len(ds)}")

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        text_col = spec["text"]
        eos = tok.eos_token_id

        def tokenize(batch):
            out = tok(batch[text_col], add_special_tokens=False)
            # Append EOS after each document so the model learns boundaries.
            ids = [seq + [eos] for seq in out["input_ids"]]
            return {"input_ids": ids}

        print("Tokenizing...")
        tokenized = ds.map(
            tokenize,
            batched=True,
            remove_columns=ds.column_names,
            num_proc=args.num_proc,
        )
        # Persist BEFORE packing so a packing crash doesn't waste tokenization.
        print(f"Saving tokenized (pre-pack) cache to: {tok_dir}")
        tokenized.save_to_disk(tok_dir)

    def pack(batch):
        # Concatenate all token ids, then split into seq_len chunks (drop remainder).
        concat = []
        for ids in batch["input_ids"]:
            concat.extend(ids)
        total = (len(concat) // seq_len) * seq_len
        chunks = [concat[i : i + seq_len] for i in range(0, total, seq_len)]
        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    print(f"Packing into blocks of {seq_len} tokens (pack_num_proc={args.pack_num_proc})...")
    packed = tokenized.map(
        pack,
        batched=True,
        batch_size=1000,
        remove_columns=tokenized.column_names,
        num_proc=args.pack_num_proc,
        writer_batch_size=1000,
    )

    print(f"Packed blocks: {len(packed)}  (~{len(packed) * seq_len / 1e6:.1f}M tokens)")
    packed.save_to_disk(args.out)
    print(f"Saved tokenized+packed dataset to: {args.out}")
    print(f"(intermediate tokenized cache kept at: {tok_dir} -- safe to delete to free space)")


if __name__ == "__main__":
    main()
