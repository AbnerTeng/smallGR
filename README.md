# smallGR

Compare T5-small and T5-base against T5-Large on Natural Questions generative retrieval, using the same document IDs and training pairs. Fine-tuning uses Axolotl; experiment tracking uses Weights & Biases.
Source code lives in `src/`. The launcher invokes `python -m src`, and the Axolotl configs load `src.axolotl_plugin.RQPlugin`.


## Experiment

Each document has three semantic RQ tokens and a fourth token that disambiguates documents sharing the same semantic prefix:

```text
query: when is the last episode of season 8 of the walking dead
-> <r0_38> <r1_12> <r2_110> <r3_0> </s>
```

The semantic codes reuse levels 0–2 of the existing NQ RQ9 artifacts in `~/git/t5_gr`. The suffix is assigned deterministically within each prefix group. Each of the four namespaces contains 256 tokens; the resulting document IDs are unique.

Training includes real queries, document text, and pseudo queries:

| Split/source | Examples |
| --- | ---: |
| Real training queries | 291,801 |
| Documents | 109,650 |
| Pseudo queries | 548,194 |
| Total training pairs | 949,645 |
| Validation queries | 15,572 |
| Test queries | 7,830 |

| Model | Config | Embedding width |
| --- | --- | ---: |
| T5-small | [configs/t5-small.yaml](configs/t5-small.yaml) | 512 |
| T5-base | [configs/t5-base.yaml](configs/t5-base.yaml) | 768 |
| T5-Large | [configs/t5-large.yaml](configs/t5-large.yaml) | 1,024 |

The defaults match the reference training schedule: 10 epochs, learning rate `3e-4`, cosine decay, 10% warmup, micro-batch 32, gradient accumulation 4, effective batch 128, and maximum input length 512. Parameters remain float32 with bf16 computation. Validation and checkpoint saving occur every 2,000 optimizer steps; the best checkpoint is selected by validation Hits@10.

Evaluation uses 10-beam decoding constrained to the full document-ID trie and reports Hits@1, Hits@5, and Hits@10. The current evaluator supports one GPU.

### RQ embedding initialization

The default `rq_init: projected` initializes the first three token namespaces from the reference codebooks. T5-small and T5-base use a fixed Gaussian projection, seeded with 42, from 1,024 dimensions to their embedding width. T5-Large uses the original codebook vectors. Vectors are normalized and scaled to the mean pretrained embedding norm. The suffix namespace retains the default newly added token initialization.

Projection introduces an initialization difference across model widths. To compare all three sizes with the default newly added token initialization, run each with `--init random` and a separate output directory.

## Performance comparison

The reference T5-Large run uses the same NQ training data, pseudo queries, and RQ3 + collision-suffix IDs. It completed 10 epochs (74,200 optimizer steps); validation Hits@10 selected checkpoint 68,000, at approximately epoch 9.16. The saved reference metrics and original source paths are recorded in [results/t5-large-nq-rq3-suffix.json](results/t5-large-nq-rq3-suffix.json).

### Validation: 10 beams, 15,572 queries

All values below are percentages. This matches the decoding budget used by the current training configs.

| Model | Full-training status | Hits@1 | Hits@5 | Hits@10 |
| --- | --- | ---: | ---: | ---: |
| T5-small | Pending; one-step smoke check passed | — | — | — |
| T5-base | Pending; one-step smoke check passed | — | — | — |
| T5-Large | Reference checkpoint 68,000 | 63.57 | 80.24 | 83.59 |

### Held-out test: 7,830 queries

The saved T5-Large test evaluations use larger beam budgets. All metrics are percentages; document recall and ID hit rates coincide because the four-token IDs are unique.

| Model | Beams | Hits@1 | Hits@10 | Hits@100 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T5-Large | 100 | 63.64 | 83.24 | 91.21 | 70.59 | 73.68 |
| T5-Large | 300 | 63.69 | 83.21 | 91.17 | 70.62 | 73.69 |

A 10-beam T5-Large test result was not found among the saved baseline artifacts. The current `evaluate` command uses 10 beams, so a direct test comparison requires evaluating every model with the same beam budget and split. T5-small and T5-base still need full training and held-out evaluation before their performance can be compared with these reference results.

The reference run used the Transformers trainer in `t5_gr`; this repository uses Axolotl with the matched data and training schedule. Smaller models also use projected codebook embeddings, as described in [RQ embedding initialization](#rq-embedding-initialization). These implementation and initialization differences should remain explicit when reporting the comparison.

## Environment

Run commands from the repository root. Python 3.12 is required. [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), and [.python-version](.python-version) specify the environment; Torch and xformers use CUDA 12.8 wheels.

On this machine, the launcher activates `~/git/rar/.venv`. To build an environment on another server:

```bash
uv sync --locked
source .venv/bin/activate
export SMALLGR_VENV="$PWD/.venv"
wandb login
```

`SMALLGR_VENV` overrides the launcher's environment path. The launcher explicitly sets `CUDA_VISIBLE_DEVICES=1`; adjust that line in [scripts/run.sh](scripts/run.sh) if the other server uses a different GPU index.

The dependency overrides align Axolotl 0.12.2 with the Torch/Transformers stack used for the local checks. Smoke tests ran in the shared `~/git/rar/.venv`, and lockfile consistency was checked. A fresh `uv sync` installation has not been tested locally.

## Prepare or transfer data

Preparation imports existing artifacts from the reference project. It requires these directories under `--source-root`:

```text
data/gr/nq_rq9_prefix3_collision_suffix_dsi_qg_single_route_s42/
data/rq_runs/nq_rq9_prefix3_collision_suffix_s42/
```

```bash
bash scripts/run.sh prepare --source-root ~/git/t5_gr
```

This creates `data/nq/` containing `train.jsonl`, `validation.jsonl`, `test.jsonl`, `idx_to_rqid.json`, `codebooks.npy`, and `manifest.json`. Preparation checks document-ID uniqueness, target/document alignment, and the training row count, and records split checksums. An existing output directory is refused; use `--output-dir` to select a fresh directory.

**Data is git-ignored. Copy the complete `data/nq/` directory to the other server alongside the repository.** Git push/pull transfers the code, configs, and lockfile. Once the prepared data is copied, training and evaluation do not require the reference repository.

## Train

```bash
bash scripts/run.sh train --config configs/t5-small.yaml
bash scripts/run.sh train --config configs/t5-base.yaml
```

An optional T5-Large run uses the same pipeline:

```bash
bash scripts/run.sh train --config configs/t5-large.yaml
```

Runs log online to the W&B project `smallgr-nq`. Override tracking with `--wandb-mode offline` or `--wandb-mode disabled`. Use `--data-dir` for prepared data stored elsewhere.

The default output directories are `checkpoints/nq-t5-{small,base,large}-rq3-suffix-pseudo-s42/`. Each contains model/tokenizer files, Trainer checkpoints and state, and `results.json` with training logs, validation metrics, parameter count, and initialization settings.

For an initialization ablation:

```bash
bash scripts/run.sh train --config configs/t5-small.yaml \
  --init random --output-dir checkpoints/nq-t5-small-random-s42
```

Resume from an actual saved checkpoint:

```bash
bash scripts/run.sh train --config configs/t5-small.yaml \
  --resume checkpoints/nq-t5-small-rq3-suffix-pseudo-s42/checkpoint-2000
```

Keep the model config, data, initialization setting, and output directory consistent when resuming. Nonempty output directories require `--resume`; new experiments should use `--output-dir`.

## Evaluate

Evaluate the saved model on the held-out test split:

```bash
bash scripts/run.sh evaluate \
  --checkpoint checkpoints/nq-t5-small-rq3-suffix-pseudo-s42 \
  --split test

bash scripts/run.sh evaluate \
  --checkpoint checkpoints/nq-t5-base-rq3-suffix-pseudo-s42 \
  --split test
```

Use `--split validation` for validation, `--batch-size` to adjust evaluation memory use, and `--data-dir` for a different prepared-data directory. Metrics are printed and saved as `test_metrics.json` or `validation_metrics.json` inside the checkpoint directory.

## Smoke tests

On a fresh output directory:

```bash
bash scripts/run.sh smoke --source-root ~/git/t5_gr
bash scripts/test.sh
```

The smoke command prepares `data/smoke/nq/` with four examples from each training source, four validation queries, and four test queries, while retaining the full corpus ID mapping. It runs exactly one optimizer step for T5-small and T5-base, saves and reloads each checkpoint, and evaluates retrieval. W&B tracking defaults to offline. A copied `data/smoke/nq/` also allows the smoke command to run without the reference repository.

Smoke outputs use fixed directories `outputs/smoke/t5-small/` and `outputs/smoke/t5-base/`. To repeat an individual check without overwriting them:

```bash
bash scripts/run.sh train --config configs/t5-small.yaml \
  --data-dir data/smoke/nq --smoke --output-dir outputs/smoke-repeat/t5-small
```

Completed local checks on GPU 1:

| Model | Optimizer steps | Training loss | Checkpoint reload |
| --- | ---: | ---: | --- |
| T5-small | 1 | 27.9312 | Passed |
| T5-base | 1 | 35.7687 | Passed |

All four retrieval-constraint tests passed, including collision suffix handling and rejection of invalid prefixes. Both smoke checkpoints had zero Hits@1/5/10 on the four validation queries. These checks establish that the training and evaluation pipeline runs; full training is needed to compare retrieval performance.
