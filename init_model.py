"""Initialize a Qwen2.5-style model FROM SCRATCH (random weights).

We reuse the official Qwen2.5 tokenizer (training a tokenizer from scratch is a
separate concern and not needed to validate the pretrain pipeline), but the
model weights are randomly initialized -- this is a true from-scratch pretrain.

Usage:
    python init_model.py                      # Qwen2.5-0.5B architecture
    python init_model.py --tiny               # ~6M params, for fast smoke tests
    python init_model.py --out ./my_model
"""
import argparse

from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

# Official Qwen2.5-0.5B architecture (https://huggingface.co/Qwen/Qwen2.5-0.5B)
QWEN2_5_0_5B = dict(
    hidden_size=896,
    intermediate_size=4864,
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
    max_position_embeddings=32768,
    rope_theta=1000000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)

# Minimal config for a sub-minute CPU/MPS smoke test of the full pipeline.
TINY = dict(
    hidden_size=256,
    intermediate_size=768,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)

TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./model_init", help="output dir")
    p.add_argument("--tiny", action="store_true", help="use a tiny config for smoke tests")
    p.add_argument("--tokenizer", default=TOKENIZER_NAME)
    args = p.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    arch = TINY if args.tiny else QWEN2_5_0_5B
    cfg = Qwen2Config(
        vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        **arch,
    )

    print("Building model with random weights (from scratch)...")
    model = Qwen2ForCausalLM(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_params/1e6:.1f}M")
    print(f"Trainable params: {n_train/1e6:.1f}M")

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved initialized model + tokenizer to: {args.out}")


if __name__ == "__main__":
    main()
