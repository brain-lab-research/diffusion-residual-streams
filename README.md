# From Evidence to Predictions: Residual Computation in Diffusion Language Models

**Ekaterina Alimaskina · Suren Kazarian · Gleb Molodtsov · Aleksandr Beznosikov**

Masked diffusion models use visible tokens as evidence and masked tokens as prediction targets in the same forward pass. We study whether multiple persistent residual streams help the model handle these roles. The repository contains the model, training code, and experiments for the 337M and 1B models in the paper.

## Main results

| Residual interface | 337M validation CE ↓ | Mean external PPL improvement ↑ |
| --- | ---: | ---: |
| Standard residual | 3.605514 | 0.00% |
| Learned 1 stream | 3.580209 | 1.06% |
| **2 streams** | 3.546667 | **5.13%** |
| 3 streams | **3.541492** | 4.78% |
| 4 streams | 3.542203 | 4.13% |

Two streams capture **86.6%** of the largest validation-CE gain beyond one learned stream. Across four external corpora, the two-stream model has the largest mean perplexity improvement. Interventions show that masked and visible positions use distinct read and write routes, while write strength remains token-dependent within each role. A role-factorized model trained with this structure reaches **3.5634 validation CE**.

## Repository contents

| Directory or file | Contents |
| --- | --- |
| `textdiffusion/`, `frr.py` | Multi-stream models and role-factorized interface |
| `model_and_train.py`, `model_and_train_ddp.py` | Training code |
| `scripts/` | Training and analysis entry points |
| `mechanism_audit_n2/` | Routing, state, evidence-transfer, and autoregressive analyses |
| `volume_analysis/` | Physical write-volume, destination, and joint interventions |
| `analysis_1b/` | Frozen 1B checkpoint experiments |
| `ood_diffusion_eval.py` | Evaluation on external corpora |

## Setup

Use Python 3.10+ and install the pinned dependencies. Training and checkpoint analyses require NVIDIA GPUs with BF16 support.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.smoke
```

The smoke check runs on CPU and needs no data or checkpoint.

Prepare the ordered FineWeb-Edu `sample-10BT` data with the Dream tokenizer:

```bash
python prepare_data.py --out data/pretrain_fwe_3600
export PRETRAIN_DATA="$PWD/data/pretrain_fwe_3600"
```

The preparation script uses 3.9 million documents, adds EOS after each document, and packs 1,024-token blocks without padding. It downloads the dataset and tokenizer and writes about 14 GB of token data. The dataset and tokenizer revisions are pinned in the preparation script.

## Train and analyze

### 337M models

All arms use the same data order, seed 1337, and 53,400 optimizer updates. Checkpoints go to `runs_pt/` by default.

```bash
for arm in baseline n1 n2 n3 n4 frr; do
  bash scripts/train.sh "$arm"
done

export GPU_ID=0
bash scripts/analyze_337m.sh
```

The analysis script prepares disjoint calibration and evaluation splits, runs the routing and state experiments, and produces the physical write-intervention results. Set `GPU_ID` to an idle physical GPU index. To use checkpoints outside `runs_pt/`, set `RUNS_DIR` before running the analysis.

Evaluate the frozen 337M checkpoints on OpenWebText, WikiText-103, LAMBADA, and C4:

```bash
python ood_diffusion_eval.py --runs-dir runs_pt --output ood_diffusion_results.json
```

### 1B models

The 1B training launcher uses two GPUs and stops at checkpoint index `124000`, the checkpoint analyzed in the paper.

```bash
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/train_1b.sh baseline
bash scripts/train_1b.sh n2

export CUDA_VISIBLE_DEVICES=0
bash scripts/analyze_1b.sh
```

The analysis uses the two training logs and the two-stream checkpoint. If they are stored elsewhere, set `RUNS_DIR` for the logs and `ONE_B_CHECKPOINT=/path/to/ckpt_124000.pt` for the checkpoint.

### Autoregressive controls

```bash
export GPU_ID=0
export CUDA_VISIBLE_DEVICES="$GPU_ID"
for arm in ar_baseline ar_n1 ar_n2 ar_n3 ar_n4; do
  bash scripts/train.sh "$arm"
done
for model in full_n2 full_n3 full_n4 ar_n2 ar_n3 ar_n4; do
  python -m mechanism_audit_n2.stream_survey --model "$model" --gpu "$GPU_ID"
done
python -m mechanism_audit_n2.ar_volume --stage full --gpu "$GPU_ID"
python -m mechanism_audit_n2.ar_volume_streams --stage full --gpu "$GPU_ID"
```

## Write-intervention protocol

For current two-stream write gains `g_0, g_1`, let `V = g_0 + g_1` and `q_i = g_i/V`. A volume replacement proposes `g'_i = V' q_i`. The proposal is applied only when `V` and `V'` are positive and both proposed gains are finite and in `[0, 2]`. Otherwise, both original gains remain; gains are never clipped.

Within-role shuffling exchanges volumes between random disjoint pairs formed separately for each sequence, role, and interface. A pair is changed only when both directions are feasible, so the volume multiset is preserved. Destination means are calibrated by averaging `q_i` directly. Joint interventions replace read, mixing, destination, and volume together; rejection of the write-gain pair does not undo the read and mixing replacements.

Calibration and evaluation sequences are disjoint. Intervention effects use masked-target CE and 5,000 paired bootstrap resamples of sequences. The repository ships code, not datasets, checkpoints, or generated results.
