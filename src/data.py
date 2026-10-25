import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

DATASET = "nq_rq9_prefix3_collision_suffix_dsi_qg_single_route_s42"
RQ = "nq_rq9_prefix3_collision_suffix_s42"


def prepare(source_root, output_dir, smoke=False):
    source_root, output_dir = Path(source_root).expanduser(), Path(output_dir)
    source = source_root / "data/gr" / DATASET
    rq = source_root / "data/rq_runs" / RQ
    manifest = json.loads((source / "manifest.json").read_text())
    routes = json.loads((rq / "idx_to_rqid.json").read_text())
    if len(routes) != manifest["n_documents"] or len(routes) != len(set(routes)):
        raise ValueError("Document IDs must be unique and cover the corpus")
    if manifest["training_sources"]["pseudo_query"] <= 0:
        raise ValueError("Expected pseudo-query training examples")
    codebooks = np.load(rq / "rq_codebooks_semantic3.npy")
    if codebooks.shape != (3, 256, 1024):
        raise ValueError(f"Unexpected codebook shape: {codebooks.shape}")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copy2(rq / "idx_to_rqid.json", output_dir / "idx_to_rqid.json")
    shutil.copy2(rq / "rq_codebooks_semantic3.npy", output_dir / "codebooks.npy")
    copied = {}
    for src_name, dst_name in [
        ("train.jsonl", "train.jsonl"),
        ("validation_expanded.jsonl", "validation.jsonl"),
        ("test_expanded.jsonl", "test.jsonl"),
    ]:
        destination = output_dir / dst_name
        digest = hashlib.sha256()
        count = 0
        if src_name == "train.jsonl":
            starts = [0, manifest["training_sources"]["query"]]
            starts.append(starts[-1] + manifest["training_sources"]["document"])
            selected = {start + j for start in starts for j in range(4)}
        else:
            selected = set(range(4))
        with (source / src_name).open() as src, destination.open("w") as dst:
            for i, line in enumerate(src):
                if smoke and i not in selected:
                    if i > max(selected):
                        break
                    continue
                row = json.loads(line)
                if row["output"] != routes[int(row["doc_idx"])]:
                    raise ValueError(f"Target/doc_idx mismatch in {src_name}, row {i}")
                if "gt_rqid" in row and row["gt_rqid"] != row["output"]:
                    raise ValueError(f"Ground truth mismatch in {src_name}, row {i}")
                dst.write(line)
                digest.update(line.encode())
                count += 1
        copied[dst_name] = {"rows": count, "sha256": digest.hexdigest()}
    expected = manifest["n_train_examples"]
    if not smoke and copied["train.jsonl"]["rows"] != expected:
        raise ValueError("Training row count differs from source manifest")
    result = {
        "source": manifest,
        "smoke": smoke,
        "smoke_examples_per_training_source": 4 if smoke else None,
        "files": copied,
    }
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return output_dir
