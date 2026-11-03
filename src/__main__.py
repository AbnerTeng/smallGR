import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def train(args):
    import yaml
    import torch
    import wandb
    from axolotl.cli.config import load_cfg
    from axolotl.integrations.base import PluginManager
    from axolotl.train import train as axolotl_train
    from axolotl.utils.dict import DictDefault
    from transformers import AutoTokenizer, set_seed

    from .ids import TOKENS, validate_tokens

    raw = yaml.safe_load(Path(args.config).read_text())
    size = raw["base_model"].rsplit("/", 1)[-1]
    if args.data_dir:
        raw["rq_data_dir"] = args.data_dir
    if args.init:
        raw["rq_init"] = args.init
    raw["rq_data_dir"] = str(Path(raw["rq_data_dir"]).resolve())
    raw["datasets"] = [{"path": str(Path(raw["rq_data_dir"]) / "train.jsonl"), "type": "completion"}]
    raw["test_datasets"] = [{"path": str(Path(raw["rq_data_dir"]) / "validation.jsonl"), "type": "completion"}]
    raw["dataset_num_proc"] = 1
    if args.smoke:
        raw.update(
            output_dir=f"outputs/smoke/{size}",
            wandb_name=f"smoke-{size}-rq3-suffix-pseudo-s42",
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            eval_batch_size=2,
            max_steps=1,
            num_epochs=1,
            warmup_ratio=0.0,
            warmup_steps=0,
            logging_steps=1,
            eval_steps=1,
            save_steps=1,
            dataloader_num_workers=0,
            load_best_model_at_end=False,
        )
    if args.output_dir:
        raw["output_dir"] = args.output_dir
    if args.resume:
        raw["resume_from_checkpoint"] = str(Path(args.resume).resolve())
    output = Path(raw["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output already exists; use --resume or --output-dir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = args.wandb_mode or ("offline" if args.smoke else "online")
    os.environ["WANDB_DIR"] = str(output.resolve())
    set_seed(raw["seed"])
    cfg = load_cfg(DictDefault(raw))
    cfg.torch_dtype = torch.float32  # Match the reference: fp32 parameters, bf16 autocast.
    datasets = PluginManager.get_instance().load_datasets(cfg)
    model, tokenizer, trainer = axolotl_train(cfg, datasets)
    metrics = trainer.evaluate()
    if args.smoke and trainer.state.global_step != 1:
        raise RuntimeError("Smoke run must complete exactly one optimizer step")
    if not any("loss" in row and math.isfinite(row["loss"]) for row in trainer.state.log_history):
        raise RuntimeError("No finite training loss was logged")
    reloaded = AutoTokenizer.from_pretrained(output)
    validate_tokens(reloaded)
    if tokenizer.convert_tokens_to_ids(TOKENS) != reloaded.convert_tokens_to_ids(TOKENS):
        raise RuntimeError("RQ token IDs changed on save/reload")
    result = {
        "base_model": raw["base_model"],
        "optimizer_steps": trainer.state.global_step,
        "parameters": sum(p.numel() for p in model.parameters()),
        "rq_init": raw["rq_init"],
        "projection_seed": raw["rq_projection_seed"],
        "training_config": raw,
        "data_manifest": json.loads((Path(raw["rq_data_dir"]) / "manifest.json").read_text()),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "smoke": args.smoke,
        "metrics": metrics,
        "train_log": trainer.state.log_history,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    trainer.save_state()
    if wandb.run:
        wandb.run.summary.update({**metrics, "optimizer_steps": trainer.state.global_step})
        wandb.finish()
    print(json.dumps(result, indent=2))
    del trainer, model
    torch.cuda.empty_cache()


def evaluate(args):
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
    from transformers import DataCollatorForSeq2Seq
    from .axolotl_plugin import tokenize_dataset, retrieval_metrics
    from .ids import validate_tokens

    root = Path(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    validate_tokens(tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cuda")
    dataset = tokenize_dataset(root / f"{args.split}.jsonl", tokenizer, 512, root / "cache")
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=TrainingArguments(
            output_dir=str(Path(args.checkpoint) / "eval"),
            per_device_eval_batch_size=args.batch_size,
            bf16=True,
            report_to="none",
            dataloader_drop_last=False,
        ),
        data_collator=collator,
    )
    routes = json.loads((root / "idx_to_rqid.json").read_text())
    metrics = retrieval_metrics(trainer, dataset, routes, 10)
    destination = Path(args.checkpoint) / f"{args.split}_metrics.json"
    destination.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-root", default="~/git/t5_gr")
    prep.add_argument("--output-dir", default="data/nq")
    prep.add_argument("--smoke", action="store_true")
    prep.add_argument("--dataset", help="Hugging Face repository containing data/ and artifacts/")
    from .data import HF_REVISION
    prep.add_argument("--revision", default=HF_REVISION)
    training = sub.add_parser("train")
    training.add_argument("--config", default="configs/t5-small.yaml")
    training.add_argument("--data-dir")
    training.add_argument("--output-dir")
    training.add_argument("--smoke", action="store_true")
    training.add_argument("--resume")
    training.add_argument("--init", choices=["projected", "random"])
    training.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--data-dir", default="data/nq")
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    evaluation.add_argument("--batch-size", type=int, default=64)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--source-root", default="~/git/t5_gr")
    smoke.add_argument("--data-dir", default="data/smoke/nq")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        prepare(args.source_root, args.output_dir, args.smoke, args.dataset, args.revision)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "smoke":
        from .data import prepare
        if not Path(args.data_dir).exists():
            prepare(args.source_root, args.data_dir, smoke=True)
        for size in ("small", "base"):
            subprocess.run(
                [sys.executable, "-m", "src", "train", "--config", f"configs/t5-{size}.yaml",
                 "--data-dir", args.data_dir, "--smoke"],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "src", "evaluate",
                 "--checkpoint", f"outputs/smoke/t5-{size}", "--data-dir", args.data_dir,
                 "--split", "validation", "--batch-size", "2"],
                check=True,
            )


if __name__ == "__main__":
    main()
