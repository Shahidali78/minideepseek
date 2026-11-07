import json

import pytest

from minideepseek.data import audit, grpo_records, load_rows, sft_records, training_rows
from minideepseek.protocol import freeze, verify_frozen
from minideepseek.runtime import Settings


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_contamination(tmp_path):
    for split, q in [("train", "a"), ("val", "b"), ("test", "c")]:
        write(tmp_path / f"{split}.jsonl", [{"question": q, "answer": "1"}])
    assert audit(tmp_path)["train"]["rows"] == 1
    write(tmp_path / "val.jsonl", [{"question": " A ", "answer": "1"}])
    with pytest.raises(ValueError, match="contamination"):
        audit(tmp_path)


def test_dedup_and_formats(tmp_path):
    for split, q in [("train", "a"), ("val", "b"), ("test", "c")]:
        write(tmp_path / f"{split}.jsonl", [{"question": q, "answer": "1"}] * 2)
    rows = training_rows(tmp_path)
    assert len(rows) == 1
    assert "solution" not in grpo_records(rows)[0]
    assert grpo_records(rows)[0]["answer"] == "1"
    assert sft_records(rows)[0]["completion"][0]["content"] == "\\boxed{1}"
    write(
        tmp_path / "bad.jsonl",
        [{"question": "a", "answer": "1", "solution": "\\boxed{2}"}],
    )
    with pytest.raises(ValueError):
        load_rows(tmp_path / "bad.jsonl")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_generations": 4},
        {"temperature": 0},
        {"gradient_accumulation_steps": 3},
        {"beta": -1},
        {"max_length": 1},
        {"lora_r": 0},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs).validate()


def test_freeze(tmp_path):
    config = tmp_path / "cfg.json"
    config.write_text("{}")
    for split in ["train", "val", "test"]:
        write(tmp_path / f"{split}.jsonl", [{"question": split, "answer": "1"}])
    frozen = tmp_path / "freeze.json"
    frozen.write_text(json.dumps(freeze(config, tmp_path, [])))
    verify_frozen(frozen, config, tmp_path, None)
    config.write_text('{"seed":1}')
    with pytest.raises(ValueError):
        verify_frozen(frozen, config, tmp_path, None)


def test_generator():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "create_data", "scripts/create_data.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rows = m.build("train", 1, 30, 100, 42)
    assert rows == m.build("train", 1, 30, 100, 42)
    assert len({r["question"] for r in rows}) == 100
    assert len({r["category"] for r in rows}) == 9


def test_comparison():
    from minideepseek.comparison import validate_comparison

    a = {
        "config": {},
        "dataset_sha256": "x",
        "resolved_model_revision": "r",
        "split": "val",
        "records": [{"id": "a"}],
    }
    validate_comparison([a, dict(a)])
    with pytest.raises(ValueError):
        validate_comparison([a, {**a, "split": "test"}])


def test_generated_references():
    import importlib.util

    from minideepseek.core import numeric
    from minideepseek.data import verified_synthetic_answer

    spec = importlib.util.spec_from_file_location(
        "create_data", "scripts/create_data.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for r in m.build("train", 1, 50, 100, 42):
        assert verified_synthetic_answer(r["question"]) == numeric(r["answer"])


def test_wrong_reference(tmp_path):
    write(tmp_path / "wrong.jsonl", [{"question": "Calculate 3 * 4.", "answer": "11"}])
    with pytest.raises(ValueError, match="incorrect synthetic"):
        load_rows(tmp_path / "wrong.jsonl")


def test_paired_interval():
    from minideepseek.comparison import paired_interval

    a = [{"accuracy_at_1": False}] * 4
    b = [{"accuracy_at_1": True}] * 4
    assert paired_interval(a, b, "accuracy_at_1")["difference"] == 1
    assert paired_interval(a, b, "accuracy_at_1")["paired_bootstrap_ci95"] == [1, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gradient_accumulation_steps": 0},
        {"temperature": float("nan")},
        {"beta": float("inf")},
        {"seed": -1},
        {"lora_r": 1.5},
    ],
)
def test_invalid_numeric_config(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs).validate()
