"""Pack the ordered FineWeb-Edu sample used for pretraining into 1024-token blocks."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER = "Dream-org/Dream-v0-Base-7B"
TOKENIZER_REVISION = "ebb8da7fc21b8dccf8d53de7579d20410604208e"
DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=3_500_000_000)
    parser.add_argument("--val-tokens", type=int, default=8_000_000)
    parser.add_argument("--documents", type=int, default=3_900_000)
    parser.add_argument("--num-proc", type=int, default=16)
    args = parser.parse_args()
    seq = 1024
    n_blocks = (args.train_tokens + args.val_tokens) // seq
    if n_blocks <= args.val_tokens // seq:
        parser.error("training split must contain at least one block")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "all_tokens.npy").exists() or (args.out / "meta.json").exists():
        parser.error("output already exists; choose a new directory")

    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION, trust_remote_code=True)
    ds = load_dataset(DATASET, "sample-10BT", revision=DATASET_REVISION, split="train")
    if len(ds) < args.documents:
        raise ValueError(f"dataset has {len(ds)} documents, requested {args.documents}")
    ds = ds.select(range(args.documents))

    def tokenize(batch):
        return {"ids": tok(batch["text"], add_special_tokens=False)["input_ids"]}

    tokenized = ds.map(tokenize, batched=True, batch_size=256, num_proc=args.num_proc,
                       remove_columns=ds.column_names)
    tokens = np.lib.format.open_memmap(args.out / "all_tokens.npy", mode="w+", dtype=np.uint32,
                                       shape=(n_blocks, seq))
    buffer, block, started = [], 0, time.time()
    for item in tokenized:
        buffer.extend(item["ids"])
        buffer.append(tok.eos_token_id)
        while len(buffer) >= seq and block < n_blocks:
            tokens[block] = np.asarray(buffer[:seq], dtype=np.uint32)
            del buffer[:seq]
            block += 1
            if block % 400_000 == 0:
                print(f"{block}/{n_blocks} blocks, {time.time() - started:.0f}s", flush=True)
        if block == n_blocks:
            break
    tokens.flush()
    if block != n_blocks:
        (args.out / "all_tokens.npy").unlink()
        raise RuntimeError(f"selected documents supplied only {block}/{n_blocks} blocks")
    metadata = {"seq_len": seq, "n_blocks": block, "n_val_blocks": args.val_tokens // seq,
                "eos": int(tok.eos_token_id), "tokenizer": f"{TOKENIZER}@{TOKENIZER_REVISION}",
                "dataset": f"{DATASET}/sample-10BT@{DATASET_REVISION}",
                "documents": args.documents, "packing": "document tokens followed by EOS, no padding"}
    (args.out / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {block} blocks in {args.out}", flush=True)


if __name__ == "__main__":
    main()
