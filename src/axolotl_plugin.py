import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from pydantic import BaseModel
from transformers import DataCollatorForSeq2Seq, LogitsProcessorList, Trainer

from axolotl.common.datasets import TrainDatasetMeta
from axolotl.core.trainers import AxolotlTrainer
from axolotl.integrations.base import BasePlugin
from axolotl.loaders import load_tokenizer

from .ids import TOKENS, route_ids, validate_tokens
from .trie import RouteConstraint


class RQArgs(BaseModel):
    rq_data_dir: str = "data/nq"
    rq_init: str = "projected"
    rq_projection_seed: int = 42
    rq_eval_beams: int = 10


def tokenize_dataset(path, tokenizer, max_length, cache_dir):
    dataset = load_dataset("json", data_files=str(path), split="train", cache_dir=str(cache_dir))

    def encode(batch):
        encoded = tokenizer(batch["input"], truncation=True, max_length=max_length)
        encoded["labels"] = [
            route_ids(tokenizer, route) + [tokenizer.eos_token_id]
            for route in batch["output"]
        ]
        return encoded

    return dataset.map(encode, batched=True, remove_columns=dataset.column_names)


@torch.no_grad()
def retrieval_metrics(trainer, dataset, routes, beams):
    if trainer.args.world_size != 1:
        raise ValueError("Constrained retrieval evaluation currently requires one GPU")
    model = trainer.model
    tokenizer = trainer.processing_class
    constraint = RouteConstraint(
        [route_ids(tokenizer, route) for route in routes],
        tokenizer.eos_token_id,
        model.get_output_embeddings().weight.shape[0],
        model.device,
    )
    hits = torch.zeros(3, device=model.device)
    count = 0
    was_training = model.training
    model.eval()
    try:
        for batch in trainer.get_eval_dataloader(dataset):
            batch = trainer._prepare_inputs(batch)
            with trainer.compute_loss_context_manager():
                generated = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_beams=beams,
                    num_return_sequences=beams,
                    max_new_tokens=5,
                    do_sample=False,
                    length_penalty=1.0,
                    logits_processor=LogitsProcessorList([constraint]),
                )
            batch_size = len(batch["input_ids"])
            predictions = generated.reshape(batch_size, beams, -1)[:, :, 1:5]
            matches = (predictions == batch["labels"][:, None, :4]).all(-1)
            for i, k in enumerate((1, 5, 10)):
                hits[i] += matches[:, :k].any(-1).sum()
            count += batch_size
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Empty evaluation dataset")
    return {f"hits@{k}": float(hits[i].item() / count) for i, k in enumerate((1, 5, 10))}


class RQTrainer(AxolotlTrainer):
    def __init__(self, *args, processing_class=None, **kwargs):
        super().__init__(*args, processing_class=processing_class, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # T5 computes encoder-decoder CE internally; never shift causal-LM labels here.
        return Trainer.compute_loss(self, model, inputs, return_outputs=return_outputs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        metrics = retrieval_metrics(
            self,
            eval_dataset if eval_dataset is not None else self.eval_dataset,
            self.rq_routes,
            self.rq_eval_beams,
        )
        metrics = {f"{metric_key_prefix}_{key}": value for key, value in metrics.items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


class RQPlugin(BasePlugin):
    def get_input_args(self):
        return "src.axolotl_plugin.RQArgs"

    def register(self, cfg):
        cfg["special_tokens"] = {"additional_special_tokens": TOKENS}
        if cfg.get("rq_init", "projected") not in ("projected", "random"):
            raise ValueError("rq_init must be projected or random")

    def load_datasets(self, cfg, preprocess=False):
        tokenizer = load_tokenizer(cfg)
        validate_tokens(tokenizer)
        root = Path(cfg.rq_data_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        if not manifest["source"]["training_sources"]["pseudo_query"]:
            raise ValueError("Pseudo queries are required")
        train = tokenize_dataset(root / "train.jsonl", tokenizer, cfg.sequence_len, root / "cache")
        valid = tokenize_dataset(root / "validation.jsonl", tokenizer, cfg.sequence_len, root / "cache")
        steps = cfg.max_steps if cfg.max_steps and cfg.max_steps > 0 else math.ceil(
            math.ceil(len(train) / cfg.batch_size) * cfg.num_epochs
        )
        return TrainDatasetMeta(train_dataset=train, eval_dataset=valid, total_num_steps=steps)

    def post_model_build(self, cfg, model):
        self.base_vocab_size = model.get_input_embeddings().num_embeddings

    def post_model_load(self, cfg, model):
        tokenizer = load_tokenizer(cfg)
        validate_tokens(tokenizer)
        model.config.use_cache = True
        if cfg.resume_from_checkpoint or cfg.rq_init == "random":
            return
        codebooks = np.load(Path(cfg.rq_data_dir) / "codebooks.npy")
        if codebooks.shape != (3, 256, 1024):
            raise ValueError(f"Expected three semantic codebooks, got {codebooks.shape}")
        width = model.config.d_model
        if width != codebooks.shape[-1]:
            rng = np.random.default_rng(cfg.rq_projection_seed)
            projection = rng.standard_normal((1024, 1024)).astype(np.float32)[:, :width]
            codebooks = codebooks @ projection
        vectors = torch.from_numpy(codebooks).float()
        vectors /= vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        embedding = model.get_input_embeddings().weight
        base_vocab_size = self.base_vocab_size
        with torch.no_grad():
            scale = embedding[:base_vocab_size].float().norm(dim=-1).mean()
            for level in range(3):
                ids = tokenizer.convert_tokens_to_ids(TOKENS[level * 256:(level + 1) * 256])
                embedding[ids] = (vectors[level].to(embedding.device) * scale).to(embedding.dtype)
        # The fourth namespace keeps its model initialization: it encodes only collisions.

    def get_trainer_cls(self, cfg):
        return RQTrainer

    def get_collator_cls_and_kwargs(self, cfg, is_eval=False):
        return DataCollatorForSeq2Seq, {"label_pad_token_id": -100, "pad_to_multiple_of": None}

    def post_trainer_create(self, cfg, trainer):
        trainer.rq_routes = json.loads((Path(cfg.rq_data_dir) / "idx_to_rqid.json").read_text())
        trainer.rq_eval_beams = cfg.rq_eval_beams
