"""Download + tokenize + pack SlimPajama-6B into fixed-length blocks.

A dedicated, self-contained variant of prepare_data.py for the SlimPajama-6B
subset (DKYoon/SlimPajama-6B, ~6B tokens). This subset is a regular
(non-streaming) dataset, so capping with --max-samples uses .select() and will
NOT load everything into RAM -- safe on Colab.

The output format is identical to prepare_data.py (columns: input_ids, labels),
so train.py consumes it unchanged.

Usage (Colab):
    !python3 prepare_data_sp.py --tokenizer Qwen/Qwen2.5-0.5B \
        --out $BASE/data_sp6b --seq-len 1024 --max-samples 1000000 --num-proc 2
"""
import argparse
import os

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

# SlimPajama-6B: a ~6B-token sampled subset of SlimPajama-627B. Non-streaming,
# text column is "text". For progressive/depth-growth runs, prepare ONCE and
# reuse the same packed dataset across baseline + every growth stage.
DATASET_PATH = "DKYoon/SlimPajama-6B"
TEXT_COL = "text"
SPLIT = "train"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="./model_init")
    p.add_argument("--out", default="./data_sp6b")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=1000000,
                   help="cap raw docs pulled; -1 for the full ~6B-token subset")
    p.add_argument("--start-doc", type=int, default=0,
                   help="skip this many docs from the start (select docs [start, start+max-samples)). "
                        "Use to prepare a DISJOINT extension chunk, e.g. --start-doc 3000000 to take "
                        "docs the first 3M-doc dataset never saw, then concatenate the two packed sets.")
    p.add_argument("--num-proc", type=int, default=4,
                   help="parallel processes for the TOKENIZE stage")
    p.add_argument("--pack-num-proc", type=int, default=1,
                   help="parallel processes for the PACK stage. Keep at 1 -- "
                        "multi-proc packing tends to OOM/crash a worker.")
    args = p.parse_args()

    seq_len = args.seq_len
    # Intermediate cache of the tokenized (pre-pack) dataset. Saving it before the
    # pack stage means a crash during packing never wastes the (slow) tokenization.
    tok_dir = args.out.rstrip("/") + ".tokenized"

    if os.path.exists(tok_dir):
        print(f"Found cached tokenized dataset -- skipping download+tokenize: {tok_dir}")
        tokenized = load_from_disk(tok_dir)
    else:
        print(f"Loading dataset: {DATASET_PATH} (split={SPLIT})")
        ds = load_dataset(DATASET_PATH, split=SPLIT)

        # Select docs [start_doc, start_doc + max_samples). Non-streaming -> select(), no OOM.
        start = max(0, args.start_doc)
        if args.max_samples and args.max_samples > 0:
            end = min(start + args.max_samples, len(ds))
        else:
            end = len(ds)                      # max-samples=-1 -> take all from start
        ds = ds.select(range(start, end))
        print(f"Raw documents: {len(ds)} (docs [{start}, {end}))")

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        eos = tok.eos_token_id

        def tokenize(batch):
            out = tok(batch[TEXT_COL], add_special_tokens=False)
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
