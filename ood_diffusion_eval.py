"""Paired OOD masked-CE evaluation of the final diffusion residual and mHC models.

Run from the pretrain_mdlm directory, for example:
    CUDA_VISIBLE_DEVICES=0 python ood_diffusion_eval.py --output ood_diffusion_results.json

The protocol matches eval_bench.py: 256 packed 1024-token blocks per corpus,
Dream tokenizer, mask ratios 0.1/0.5/0.9, seeds 901/905/909 and BF16 forward.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ.setdefault("PRETRAIN_DATA", "data/pretrain_fwe_3600")
os.environ["MHC_PAPER"] = "1"
os.environ["MHC_PROPER_INIT"] = "1"

import model_and_train as mt  # noqa: E402  (model configuration reads the environment)


ROOT = Path(__file__).resolve().parent
RATIOS = (0.1, 0.5, 0.9)
MODELS = {
    "residual": ("diff36_base", "baseline", None),
    "mhc_n1": ("diff36_lit_s1", "mhc", 1),
    "mhc_n2": ("diff36_lit_s2", "mhc", 2),
    "mhc_n3": ("diff36_lit_s3", "mhc", 3),
    "mhc_n4": ("diff36_lit", "mhc", 4),
}


def pack(texts, tok, n_blocks):
    buffer, blocks = [], []
    for content in texts:
        if not content or not content.strip():
            continue
        buffer.extend(tok(content, add_special_tokens=False)["input_ids"])
        buffer.append(tok.eos_token_id)
        while len(buffer) >= mt.SEQ:
            blocks.append(buffer[: mt.SEQ])
            del buffer[: mt.SEQ]
            if len(blocks) == n_blocks:
                return torch.tensor(blocks, dtype=torch.long)
    raise ValueError(f"Corpus yielded only {len(blocks)} blocks; requested {n_blocks}")


def load_corpora(tok, n_blocks):
    return {
        "openwebtext": pack(
            (x["text"] for x in load_dataset("Skylion007/openwebtext", split="train", streaming=True).take(2000)),
            tok, n_blocks,
        ),
        "wikitext103": pack(
            load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"],
            tok, n_blocks,
        ),
        "lambada": pack(
            load_dataset("EleutherAI/lambada_openai", "en", split="test")["text"],
            tok, n_blocks,
        ),
        "c4": pack(
            (x["text"] for x in load_dataset("allenai/c4", "en", split="validation", streaming=True).take(3000)),
            tok, n_blocks,
        ),
    }


def make_masks(blocks, batch_size):
    result = {}
    for ratio in RATIOS:
        generator = torch.Generator().manual_seed(900 + int(ratio * 10))
        # Match the historical evaluator's batchwise CPU RNG draws exactly.
        pieces = [torch.rand(blocks[i : i + batch_size].shape, generator=generator) < ratio
                  for i in range(0, len(blocks), batch_size)]
        result[str(ratio)] = torch.cat(pieces)
    return result


def evaluate(model, blocks, masks, batch_size):
    result = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ratio in RATIOS:
            sums, counts = [], []
            all_mask = masks[str(ratio)]
            for i in range(0, len(blocks), batch_size):
                ids = blocks[i : i + batch_size].cuda(non_blocking=True)
                mask = all_mask[i : i + batch_size].cuda(non_blocking=True)
                corrupted = ids.masked_fill(mask, mt.MASK_ID)
                logits = model(corrupted)
                for b in range(len(ids)):
                    targets = ids[b][mask[b]]
                    loss = F.cross_entropy(logits[b][mask[b]].float(), targets, reduction="sum")
                    sums.append(float(loss))
                    counts.append(int(targets.numel()))
                del logits
            result[str(ratio)] = {"loss_sum": sums, "tokens": counts,
                                  "ce": sum(sums) / sum(counts)}
            print(f"  r={ratio:.1f}: CE {result[str(ratio)]['ce']:.6f}", flush=True)
    result["mean_ce"] = float(np.mean([result[str(r)]["ce"] for r in RATIOS]))
    return result


def paired_summary(results, n_resamples):
    rng = np.random.default_rng(0)
    summary = {}
    for corpus in results["corpora"]:
        base = results["scores"]["residual"][corpus]
        for name in ("mhc_n1", "mhc_n2", "mhc_n3", "mhc_n4"):
            if name not in results["scores"]:
                continue
            comparison = results["scores"][name][corpus]
            point = comparison["mean_ce"] - base["mean_ce"]
            n = len(base["0.1"]["tokens"])
            draws = np.empty(n_resamples, dtype=np.float64)
            for j in range(n_resamples):
                idx = rng.integers(0, n, n)
                deltas = []
                for ratio in RATIOS:
                    key = str(ratio)
                    count = np.asarray(base[key]["tokens"], dtype=np.float64)[idx].sum()
                    lhs = np.asarray(comparison[key]["loss_sum"], dtype=np.float64)[idx].sum()
                    rhs = np.asarray(base[key]["loss_sum"], dtype=np.float64)[idx].sum()
                    deltas.append((lhs - rhs) / count)
                draws[j] = np.mean(deltas)
            summary[f"{name}/{corpus}"] = {
                "delta_ce": point,
                "ci95_delta_ce": np.percentile(draws, [2.5, 97.5]).tolist(),
                "exp_ce_reduction_pct": 100 * (1 - math.exp(point)),
                "bootstrap_p_no_gain": float(np.mean(draws >= 0)),
            }
    return summary


def save(path, results):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "ood_diffusion_results.json")
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs_pt")
    parser.add_argument("--offline-tokenizer", action="store_true", help="Require a previously cached tokenizer")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", revision="ebb8da7fc21b8dccf8d53de7579d20410604208e",
                                        trust_remote_code=True, local_files_only=args.offline_tokenizer)
    corpora = load_corpora(tok, args.blocks)
    masks = {name: make_masks(blocks, args.batch_size) for name, blocks in corpora.items()}
    results = {
        "protocol": {"blocks_per_corpus": args.blocks, "sequence_length": mt.SEQ,
                     "batch_size": args.batch_size, "ratios": RATIOS,
                     "mask_seeds": {str(r): 900 + int(r * 10) for r in RATIOS},
                     "dtype": "bfloat16", "step": 53399,
                     "tokenizer": "Dream-org/Dream-v0-Base-7B@ebb8da7fc21b8dccf8d53de7579d20410604208e",
                     "metric": "mean of three ratio-specific masked-token CE values",
                     "bootstrap_resamples": args.bootstrap},
        "corpora": {name: list(blocks.shape) for name, blocks in corpora.items()},
        "scores": {},
    }
    print(f"Packed blocks: {results['corpora']}", flush=True)

    for name, (run, arm, streams) in MODELS.items():
        mt.STREAMS = streams or 4
        checkpoint = args.runs_dir / run / "ckpt_53399.pt"
        print(f"Loading {name}: {checkpoint}", flush=True)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = mt.Model(arm, causal=False)
        model.load_state_dict(payload["model"], strict=True)
        del payload
        model = model.cuda().eval()
        results["scores"][name] = {}
        for corpus, blocks in corpora.items():
            print(f"Evaluating {name} / {corpus}", flush=True)
            results["scores"][name][corpus] = evaluate(model, blocks, masks[corpus], args.batch_size)
            save(args.output, results)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    results["paired_comparisons"] = paired_summary(results, args.bootstrap)
    save(args.output, results)
    for name, score in results["paired_comparisons"].items():
        print(f"{name}: ΔCE={score['delta_ce']:+.6f} "
              f"CI={score['ci95_delta_ce']} "
              f"exp(CE) reduction={score['exp_ce_reduction_pct']:.2f}%", flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
