"""Freeze and execute the reserved evaluation once; never train or tune."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.core import INSTRUCTION, PARSER_VERSION
from minideepseek.data import audit, digest, key, load_rows
from minideepseek.protocol import freeze
from minideepseek.runtime import hardware, save_json

ADAPTER = "outputs/phase2/grpo-base16"
DATA = "data/expanded-v2"
CONFIG = "configs/phase2-eval.json"


def normalized(question):
    return key(unicodedata.normalize("NFKC", question).replace("\u2212", "-"))


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    splits = {s: load_rows(Path(DATA) / f"{s}.jsonl") for s in ("train", "val", "test")}
    # Include every previously used training/validation version, not only v2.
    historical = [
        p for p in Path("data").rglob("*.jsonl") if p.stem in ("train", "val")
    ]
    test_keys = {normalized(r["question"]) for r in splits["test"]}
    overlap = {}
    for path in historical:
        rows = load_rows(path)
        overlap[str(path)] = len(test_keys & {normalized(r["question"]) for r in rows})
    if any(overlap.values()):
        raise ValueError(f"reserved question overlap: {overlap}")
    canonical = load_rows("data/test.jsonl")
    if [(r["question"], r["answer"]) for r in canonical] != [
        (r["question"], r["answer"]) for r in splits["test"]
    ]:
        raise ValueError("reserved copy differs from original")
    unique = list(dict.fromkeys(key(r["question"]) for r in splits["test"]))
    files = [
        *Path("src/minideepseek").glob("*.py"),
        Path("scripts/evaluate.py"),
        Path("scripts/phase3_final.py"),
        Path(CONFIG),
        Path("requirements-lock.txt"),
    ]
    model_files = {
        str(p): digest(p) for p in Path("outputs/model").glob("*") if p.is_file()
    }
    adapter_files = {str(p): digest(p) for p in Path(ADAPTER).glob("*") if p.is_file()}
    validation = {
        str(p): digest(p) for p in Path("outputs/phase2").glob("*val-v2.json")
    }
    manifest = {
        **freeze(CONFIG, DATA, [ADAPTER]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "finalists": {"base": "Qwen/Qwen3-0.6B", "grpo": ADAPTER},
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "model_files": model_files,
        "adapter_files": adapter_files,
        "code_sha256": {str(p): digest(p) for p in files},
        "code_revision": "No Git history at freeze; SHA256 source snapshot is authoritative",
        "validation_files": validation,
        "all_datasets": {str(p): digest(p) for p in Path("data").rglob("*.jsonl")},
        "software": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "hardware": hardware(),
        "audit": audit(DATA),
        "historical_normalized_overlap": overlap,
        "protocol": {
            "raw_test_rows": len(splits["test"]),
            "questions": len(unique),
            "selection": "all first-occurrence unique questions in existing order; duplicate rows excluded before outcomes",
            "samples_per_question": 8,
            "seed": "42 + unique question index; paired across models",
            "config": json.loads(Path(CONFIG).read_text()),
            "generation_batch_size": 8,
            "top_k": 0,
            "cpu_threads": 1,
            "system_prompt": INSTRUCTION,
            "parser": PARSER_VERSION,
            "thinking": False,
            "accuracy_at_1": "correctness of first stochastic candidate",
            "pass_at_8": "at least one of eight candidates correct; empirical pass, not extrapolated estimator",
            "majority_at_8": "unique modal parsed value correct; ties or no parsed values fail; invalids abstain",
            "ci": "Wilson 95% accuracy; paired question bootstrap 5000 seed42; exact two-sided McNemar",
            "latency": "GPU-synchronized batch generation plus decoding; amortized /8; excludes loading",
            "memory": "torch peak allocated after loading/reset; not total device or reserved VRAM",
            "limits": "same 256 completion tokens; truncations separate; no outcome-dependent reruns",
            "run_max_seconds": 7200,
            "test_access_before_freeze": "schema, deterministic reference integrity and split exclusion audits only; no saved test generation/scoring found",
            "limitations": "shared synthetic templates, repeated numeric structures, one sample set, finite cap, limited arithmetic task coverage",
        },
    }
    save_json(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(digest(root / "manifest.json") + "\n")
    (root / "source").mkdir()
    for p in files:
        target = root / "source" / p.as_posix()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    print(
        f"Frozen {len(unique)} unique questions from {len(splits['test'])} rows",
        flush=True,
    )


def verify(root):
    if digest(root / "manifest.json") != (root / "manifest.sha256").read_text().strip():
        raise ValueError("manifest hash mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    for section in (
        "model_files",
        "adapter_files",
        "code_sha256",
        "validation_files",
        "all_datasets",
    ):
        for path, expected in manifest[section].items():
            if digest(path) != expected:
                raise ValueError(f"frozen input changed: {path}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        prepare(root)
    manifest = verify(root)
    if not args.execute:
        return
    # Exclusive durable marker prevents accidental repetition after seeing outcomes.
    with (root / "evaluation.started").open("x") as marker:
        marker.write(datetime.now(timezone.utc).isoformat())
    for name, adapter in (("base", None), ("grpo", ADAPTER)):
        verify(root)
        command = [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            CONFIG,
            "--data",
            DATA,
            "--split",
            "test",
            "--max-examples",
            str(manifest["protocol"]["questions"]),
            "--frozen-protocol",
            str(root / "manifest.json"),
            "--max-seconds",
            "7200",
            "--output",
            str(root / f"{name}-test.json"),
        ]
        if adapter:
            command.extend(["--adapter", adapter])
        save_json(root / f"{name}-command.json", command)
        with (root / f"{name}-evaluation.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((root / f"{name}-test.json").read_text())
        if result["status"] != "complete":
            raise RuntimeError(
                "runtime limit reached; do not silently rerun final evaluation"
            )
        print(f"Completed {name}: {result['metrics']}", flush=True)
    verify(root)
    save_json(
        root / "evaluation.completed.json",
        {"completed_utc": datetime.now(timezone.utc).isoformat()},
    )


if __name__ == "__main__":
    main()
