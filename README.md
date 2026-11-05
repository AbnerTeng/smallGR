# smallGR

Compare T5-small, T5-base, T5-Large, and T5-3B on Natural Questions generative retrieval, using the same document IDs and training pairs. Fine-tuning uses Axolotl; experiment tracking uses Weights & Biases.
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
| T5-3B | [configs/t5-3b.yaml](configs/t5-3b.yaml) | 1,024 |

The defaults match the reference training schedule: 10 epochs, learning rate `3e-4`, cosine decay, 10% warmup, micro-batch 32, gradient accumulation 4, effective batch 128, and maximum input length 512. Parameters remain float32 with bf16 computation. Validation occurs every 2,000 optimizer steps; the best checkpoint is selected by validation Hits@10. Small, base, and large save every 2,000 steps. T5-3B saves when validation Hits@10 improves and retains one checkpoint, including optimizer state, to reduce storage use; it also enables gradient checkpointing and uses evaluation batches of 8.

Evaluation uses 10-beam decoding constrained to the full document-ID trie and reports Hits@1, Hits@5, and Hits@10. The current evaluator supports one GPU.

### RQ embedding initialization

The default `rq_init: projected` initializes the first three token namespaces from the reference codebooks. T5-small and T5-base use a fixed Gaussian projection, seeded with 42, from 1,024 dimensions to their embedding width. T5-Large and T5-3B use the original codebook vectors. Vectors are normalized and scaled to the mean pretrained embedding norm. The suffix namespace retains the default newly added token initialization.

Projection introduces an initialization difference across model widths. To compare all four sizes with the default newly added token initialization, run each with `--init random` and a separate output directory.

## Performance comparison

The reference T5-Large run uses the same NQ training data, pseudo queries, and RQ3 + collision-suffix IDs. It completed 10 epochs (74,200 optimizer steps); validation Hits@10 selected checkpoint 68,000, at approximately epoch 9.16. The saved reference metrics and original source paths are recorded in [results/t5-large-nq-rq3-suffix.json](results/t5-large-nq-rq3-suffix.json).

T5-small completed 10 epochs (74,200 optimizer steps) on `Abner0803/NQ320k-semid`, with 61,016,576 parameters and projected codebook initialization (seed 42). Validation Hits@10 selected checkpoint 70,000, at approximately epoch 9.43. Training reported 10,621.79 seconds (about 2 hours 57 minutes), with peak allocated GPU memory of 5.53 GiB. The saved metrics, dataset revision, and source paths are recorded in [results/t5-small-nq-rq3-suffix.json](results/t5-small-nq-rq3-suffix.json).

T5-base completed the same 10 epochs (74,200 optimizer steps), with 223,668,480 parameters and projected codebook initialization (seed 42). Validation Hits@10 selected checkpoint 70,000, at approximately epoch 9.43. Training reported 23,333.51 seconds (about 6 hours 29 minutes), with peak allocated GPU memory of 16.98 GiB. The saved metrics, dataset revision, and source paths are recorded in [results/t5-base-nq-rq3-suffix.json](results/t5-base-nq-rq3-suffix.json).

T5-Large also completed the Axolotl pipeline in this repository: 10 epochs, 74,200 optimizer steps, and 738,688,000 parameters. Validation selected checkpoint 70,000 (epoch 9.43), with Hits@10 of 83.61%. Training reported 43,056.03 seconds (about 11 hours 58 minutes), with peak allocated GPU memory of 47.19 GiB. See [results/t5-large-axolotl-nq-rq3-suffix.json](results/t5-large-axolotl-nq-rq3-suffix.json).

T5-3B completed 10 epochs (74,200 optimizer steps) with 2,852,618,240 parameters, using the same data and codebook initialization (seed 42). Job `435108` finished on November 4, 2026 at 23:38 (Asia/Taipei). Training reported 112,830.23 seconds (about 31 hours 21 minutes), with peak allocated GPU memory of 53.20 GiB. The final model achieved validation Hits@1/5/10 of 67.35% / 82.00% / 85.01%; Hits@10 is 1.40 percentage points above the Axolotl T5-Large result. The saved results and trainer state report no best checkpoint, so these metrics describe the final model at step 74,200. W&B tracking was offline. See [results/t5-3b-nq-rq3-suffix.json](results/t5-3b-nq-rq3-suffix.json).

### Validation: 10 beams, 15,572 queries

All values below are percentages. This matches the decoding budget used by the current training configs.

| Model | Full-training status | Hits@1 | Hits@5 | Hits@10 |
| --- | --- | ---: | ---: | ---: |
| T5-small | Completed 10 epochs; checkpoint 70,000 | 33.23 | 54.03 | 60.30 |
| T5-base | Completed 10 epochs; checkpoint 70,000 | 54.69 | 74.65 | 78.85 |
| T5-Large (Axolotl) | Completed 10 epochs; checkpoint 70,000 | 63.41 | 80.33 | 83.61 |
| T5-Large (reference) | Completed 10 epochs; checkpoint 68,000 | 63.57 | 80.24 | 83.59 |
| T5-3B | Completed 10 epochs; final model at step 74,200 | 67.35 | 82.00 | 85.01 |

### Held-out test: 7,830 queries

The saved reference T5-Large test evaluations use larger beam budgets; these are not test results from the new Axolotl run. All metrics are percentages; document recall and ID hit rates coincide because the four-token IDs are unique.

| Model | Beams | Hits@1 | Hits@10 | Hits@100 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T5-Large | 100 | 63.64 | 83.24 | 91.21 | 70.59 | 73.68 |
| T5-Large | 300 | 63.69 | 83.21 | 91.17 | 70.62 | 73.69 |

A 10-beam T5-Large test result was not found among the saved baseline artifacts. The current `evaluate` command uses 10 beams, so a direct test comparison requires evaluating every model with the same beam budget and split. T5-small, T5-base, the Axolotl T5-Large, and T5-3B have completed full training; their held-out test evaluations are pending.

The reference run used the Transformers trainer in `t5_gr`; this repository uses Axolotl with the matched data and training schedule. Smaller models also use projected codebook embeddings, as described in [RQ embedding initialization](#rq-embedding-initialization). These implementation and initialization differences should remain explicit when reporting the comparison.

## Environment

Run commands from the repository root. Python 3.12 is required. [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), and [.python-version](.python-version) specify the environment; Torch 2.6.0 and xformers use CUDA 12.4 wheels.

The launcher defaults to `~/git/rar/.venv`; Nano5 jobs use `/work/abner0803/venvs/smallgr-py312` through `SMALLGR_VENV`. To build an environment on another server:

```bash
uv sync --locked
source .venv/bin/activate
export SMALLGR_VENV="$PWD/.venv"
wandb login
```

`SMALLGR_VENV` overrides the launcher's environment path. Outside Slurm, the launcher sets `CUDA_VISIBLE_DEVICES=1`; inside Slurm it preserves the allocated GPU selection. Adjust that line in [scripts/run.sh](scripts/run.sh) if the other server uses a different GPU index.

The dependency overrides align Axolotl 0.12.2 with the Torch/Transformers stack used on Nano5. The Nano5 environment completed small, base, and large training; the lockfile consistency and four retrieval-constraint tests also passed. For Nano5 setup, copy the repository under `/work/abner0803` and run `bash scripts/setup_nano5.sh small base large 3b` on a node with download access before submitting offline jobs.

## Prepare or transfer data

### Hugging Face

The complete dataset is available at [Abner0803/NQ320k-semid](https://huggingface.co/datasets/Abner0803/NQ320k-semid).

| Split | Repository file | Rows |
| --- | --- | ---: |
| Train | `data/train.jsonl` | 949,645 |
| Validation | `data/validation.jsonl` | 15,572 |
| Test | `data/test.jsonl` | 7,830 |

These are the original JSONL files, verified against the local SHA-256 checksums. The repository also includes the ID mapping, codebooks, and source manifest under `artifacts/`. The requested repository name contains NQ320k; the training split includes all 949,645 augmented pairs.

To populate `data/nq/` on another server without the reference repository:

```bash
bash scripts/run.sh prepare --dataset Abner0803/NQ320k-semid --output-dir data/nq
```

For direct analysis, `load_dataset("Abner0803/NQ320k-semid")` returns all three splits. Its shared schema exposes the training-only missing `gt_rqid` field as null; `output` is the prediction target in every split.

The uploader is reproducible with `bash scripts/upload_datasets.sh`; `--check` only inspects authentication and repository status. Verified publication details are saved in [results/huggingface_datasets.json](results/huggingface_datasets.json).

### Import from the reference project

Preparation imports existing artifacts from the reference project. It requires these directories under `--source-root`:

```text
data/gr/nq_rq9_prefix3_collision_suffix_dsi_qg_single_route_s42/
data/rq_runs/nq_rq9_prefix3_collision_suffix_s42/
```

```bash
bash scripts/run.sh prepare --source-root ~/git/t5_gr
```

This creates `data/nq/` containing `train.jsonl`, `validation.jsonl`, `test.jsonl`, `idx_to_rqid.json`, `codebooks.npy`, and `manifest.json`. Preparation checks document-ID uniqueness, target/document alignment, and the training row count, and records split checksums. An existing output directory is refused; use `--output-dir` to select a fresh directory.

**Data is git-ignored.** On another server, prepare it from Hugging Face using the command above or copy the complete `data/nq/` directory. Git push/pull transfers the code, configs, and lockfile. Prepared data allows training and evaluation without the reference repository.

## Train

```bash
bash scripts/run.sh train --config configs/t5-small.yaml
bash scripts/run.sh train --config configs/t5-base.yaml
```

T5-Large and T5-3B use the same pipeline:

```bash
bash scripts/run.sh train --config configs/t5-large.yaml
bash scripts/run.sh train --config configs/t5-3b.yaml
```

Runs log online to the W&B project `smallgr-nq`. Override tracking with `--wandb-mode offline` or `--wandb-mode disabled`. Use `--data-dir` for prepared data stored elsewhere.

The default output directories are `checkpoints/nq-t5-{small,base,large,3b}-rq3-suffix-pseudo-s42/`. Each contains model/tokenizer files, Trainer checkpoints and state, and `results.json` with training logs, validation metrics, parameter count, and initialization settings.

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
