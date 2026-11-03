import argparse
import hashlib
import json
from pathlib import Path

import yaml
from datasets import Features, Value, load_dataset
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

REPO = "Abner0803/NQ320k-semid"
FEATURES = Features({
    "input": Value("string"),
    "output": Value("string"),
    "doc_idx": Value("int64"),
    "gt_rqid": Value("string"),
})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(api):
    account = api.whoami()
    result = {"authenticated_user": account["name"], "repo_id": REPO}
    try:
        info = api.dataset_info(REPO)
        result.update(exists=True, private=info.private, files=[item.rfilename for item in info.siblings])
    except RepositoryNotFoundError:
        result["exists"] = False
    print(json.dumps(result, indent=2), flush=True)
    return result


def card_text(local, root):
    metadata = {
        "language": ["en"],
        "task_categories": ["text-retrieval"],
        "tags": ["natural-questions", "generative-retrieval", "semantic-ids", "rq3"],
        "configs": [{
            "config_name": "default",
            "data_files": [{"split": split, "path": f"data/{split}.jsonl"} for split in local],
        }],
        "dataset_info": {
            "features": [{"name": name, "dtype": value.dtype} for name, value in FEATURES.items()],
            "splits": [
                {"name": split, "num_bytes": dataset.data.nbytes, "num_examples": len(dataset)}
                for split, dataset in local.items()
            ],
            "download_size": sum((root / f"{split}.jsonl").stat().st_size for split in local),
            "dataset_size": sum(dataset.data.nbytes for dataset in local.values()),
        },
    }
    body = """
# NQ320k-semid

Natural Questions generative-retrieval pairs with three semantic RQ tokens and a deterministic fourth token that disambiguates documents sharing the same semantic prefix.

## Files and splits

| Split | Original JSONL file | Rows |
| --- | --- | ---: |
| Train | data/train.jsonl | 949,645 |
| Validation | data/validation.jsonl | 15,572 |
| Test | data/test.jsonl | 7,830 |

The original JSONL files are uploaded byte-for-byte. The requested name contains NQ320k; training contains 949,645 augmented pairs: 291,801 real-query pairs, 109,650 document-text pairs, and 548,194 pseudo-query pairs. Validation is retained as a separate split for model selection. Testing is held out.

## Columns

- `input`: original query or document text, including its task prefix.
- `output`: four-token document ID, such as `<r0_38> <r1_12> <r2_110> <r3_0>`.
- `doc_idx`: zero-based corpus index.
- `gt_rqid`: ground-truth ID in the original validation/test rows, equal to `output`. This field is absent from the raw training file; the declared shared schema exposes it as null when loading the training split. Use `output` as the target for every split.

Each of the four namespaces contains 256 tokens. The target stores four RQ tokens without EOS; the training tokenizer appends EOS. Full IDs are unique across 109,650 documents.

## Construction and artifacts

The first three tokens reuse levels 0–2 of the existing NQ RQ9 codes. The suffix is assigned deterministically within each exact prefix group. The reference artifact is `nq_rq9_prefix3_collision_suffix_dsi_qg_single_route_s42`; identifiers come from `nq_rq9_prefix3_collision_suffix_s42`.

- `artifacts/idx_to_rqid.json`: shared corpus-index to document-ID mapping.
- `artifacts/codebooks.npy`: semantic codebooks, shape `(3, 256, 1024)`.
- `artifacts/manifest.json`: original counts, provenance, and JSONL SHA-256 checksums.

Only the three semantic namespaces use centroid initialization. The fourth namespace is a collision suffix.

## Load

```python
from datasets import load_dataset

dataset = load_dataset("Abner0803/NQ320k-semid")
train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]
```

To use the original files directly:

```python
from huggingface_hub import hf_hub_download

train_file = hf_hub_download("Abner0803/NQ320k-semid", "data/train.jsonl", repo_type="dataset")
test_file = hf_hub_download("Abner0803/NQ320k-semid", "data/test.jsonl", repo_type="dataset")
```
"""
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n" + body.strip() + "\n"


def upload(root):
    root = Path(root)
    api = HfApi()
    account = inspect(api)
    if account["authenticated_user"] != "Abner0803":
        raise ValueError("Expected the Abner0803 account")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["smoke"]:
        raise ValueError("Upload requires the full dataset")
    splits = ("train", "validation", "test")
    for split in splits:
        filename = f"{split}.jsonl"
        if sha256(root / filename) != manifest["files"][filename]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {filename}")
    local = load_dataset(
        "json", data_files={split: str(root / f"{split}.jsonl") for split in splits},
        features=FEATURES, cache_dir="data/hf_upload/raw_local",
    )
    for split in splits:
        if len(local[split]) != manifest["files"][f"{split}.jsonl"]["rows"]:
            raise ValueError(f"Source row count mismatch: {split}")
    print("Validated original JSONL files: " + json.dumps({
        split: {"rows": len(local[split]), "bytes": (root / f"{split}.jsonl").stat().st_size}
        for split in splits
    }), flush=True)

    api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
    attrs_path = hf_hub_download(REPO, ".gitattributes", repo_type="dataset")
    attrs = Path(attrs_path).read_text()
    rule = "*.jsonl filter=lfs diff=lfs merge=lfs -text"
    if rule not in attrs:
        api.upload_file(
            repo_id=REPO, repo_type="dataset", path_in_repo=".gitattributes",
            path_or_fileobj=(attrs.rstrip() + "\n" + rule + "\n").encode(),
            commit_message="Track original JSONL dataset files with LFS",
        )
    operations = [
        CommitOperationAdd(path_in_repo=f"data/{split}.jsonl", path_or_fileobj=root / f"{split}.jsonl")
        for split in splits
    ]
    for name in ("manifest.json", "idx_to_rqid.json", "codebooks.npy"):
        operations.append(CommitOperationAdd(path_in_repo=f"artifacts/{name}", path_or_fileobj=root / name))
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card_text(local, root).encode()))
    commit = api.create_commit(
        repo_id=REPO, repo_type="dataset", operations=operations,
        commit_message="Upload original NQ semantic-ID train, validation, and test JSONL files",
    )
    print(f"Dataset committed: {commit.oid}; verifying remote files and splits", flush=True)
    info = api.dataset_info(REPO, revision=commit.oid, files_metadata=True)
    remote = load_dataset(REPO, revision=commit.oid, cache_dir="data/hf_upload/raw_verification")
    if set(remote) != set(splits):
        raise ValueError("Remote splits differ from uploaded splits")
    for split in splits:
        if len(remote[split]) != len(local[split]):
            raise ValueError(f"Remote row count mismatch: {split}")
        for index in (0, len(local[split]) // 2, len(local[split]) - 1):
            if remote[split][index] != local[split][index]:
                raise ValueError(f"Remote row mismatch: {split}/{index}")
    files = {item.rfilename: item for item in info.siblings}
    for split in splits:
        file = files[f"data/{split}.jsonl"]
        if file.lfs is None or file.lfs.sha256 != manifest["files"][f"{split}.jsonl"]["sha256"]:
            raise ValueError(f"Remote original-file checksum mismatch: {split}")
    for name in ("manifest.json", "idx_to_rqid.json", "codebooks.npy"):
        path = hf_hub_download(
            REPO, f"artifacts/{name}", repo_type="dataset", revision=commit.oid,
            cache_dir="data/hf_upload/raw_verification",
        )
        if sha256(path) != sha256(root / name):
            raise ValueError(f"Remote artifact checksum mismatch: {name}")
    result = {
        "repo_id": REPO,
        "url": f"https://huggingface.co/datasets/{REPO}",
        "revision": commit.oid,
        "private": info.private,
        "splits": {split: len(remote[split]) for split in splits},
        "source_sha256": {
            f"data/{split}.jsonl": manifest["files"][f"{split}.jsonl"]["sha256"]
            for split in splits
        },
        "remote_verification": "passed",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/huggingface_datasets.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Upload and remote verification completed:\n" + json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--data-dir", default="data/nq")
    args = parser.parse_args()
    if args.check:
        inspect(HfApi())
    else:
        upload(args.data_dir)


if __name__ == "__main__":
    main()
