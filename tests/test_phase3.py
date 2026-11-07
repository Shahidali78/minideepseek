import json
import subprocess
import sys
from pathlib import Path

import pytest

from minideepseek.comparison import paired_interval
from minideepseek.final_analysis import error_label, error_summary
from minideepseek.protocol import freeze, verify_frozen


def test_frozen_protocol_rejects_changed_inputs(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.jsonl").write_text(split)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(freeze(config, tmp_path, [])))
    verify_frozen(manifest, config, tmp_path, None)
    (tmp_path / "test.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="protocol changed"):
        verify_frozen(manifest, config, tmp_path, None)


@pytest.mark.parametrize(
    "improved,regressed,p", [(0, 0, 1), (4, 0, 0.125), (1, 1, 1), (6, 0, 0.03125)]
)
def test_exact_discordant_test(improved, regressed, p):
    base = [{"x": 0}] * improved + [{"x": 1}] * regressed + [{"x": 1}]
    other = [{"x": 1}] * improved + [{"x": 0}] * regressed + [{"x": 1}]
    result = paired_interval(base, other, "x", repetitions=200)
    assert result["mcnemar_exact_p_two_sided"] == p
    assert result["corrected_questions"] == improved
    assert result["regressed_questions"] == regressed


@pytest.mark.parametrize(
    "text,outcome,truncated,label",
    [
        (r"\boxed{4}", "correct", False, "correct"),
        (r"\boxed{5}", "incorrect", False, "arithmetic_error"),
        ("still calculating", "parse_failure", True, "truncated_completion"),
        ("4", "parse_failure", False, "answer_formatting_failure"),
        (r"\boxed{unknown}", "parse_failure", False, "unsupported_answer_pattern"),
        ("unknown", "parse_failure", False, "failed_answer_extraction"),
    ],
)
def test_observable_errors(text, outcome, truncated, label):
    assert error_label(text, outcome, truncated, "Calculate 2 + 2.") == label


def test_error_summary_preserves_denominators():
    row = {
        "question": "Calculate 2 + 2.",
        "responses": [r"\boxed{4}"] * 8,
        "outcomes": ["correct"] * 8,
        "truncated": [False] * 8,
    }
    result = error_summary([row])
    assert sum(result["all_samples"].values()) == 8
    assert sum(result["first_samples"].values()) == 1


def test_final_runner_help_has_no_gpu_side_effects():
    result = subprocess.run(
        [sys.executable, "scripts/phase3_final.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--execute" in result.stdout


def test_ci_never_trains():
    workflows = list(Path(".github/workflows").glob("*.y*ml"))
    assert workflows
    for workflow in workflows:
        text = workflow.read_text()
        assert "train_grpo.py" not in text
        assert "train_sft.py" not in text
        assert "phase3_final.py" not in text


def test_final_start_marker_prevents_second_generation(tmp_path):
    import hashlib

    sections = (
        "model_files",
        "adapter_files",
        "code_sha256",
        "validation_files",
        "all_datasets",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({s: {} for s in sections}))
    (tmp_path / "manifest.sha256").write_text(
        hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    (tmp_path / "evaluation.started").write_text("already started")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase3_final.py",
            "--root",
            str(tmp_path),
            "--execute",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert not list(tmp_path.glob("*-test.json"))


def test_publication_scan_detects_private_fields_and_credentials(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "publication_audit", "scripts/publication_audit.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"responses": ["private"]}))
    _, issues = audit.inspect([path])
    assert any("raw evaluation" in x["reason"] for x in issues)
    path.write_text(json.dumps({"token": "ghp_" + "a" * 30}))
    _, issues = audit.inspect([path])
    assert any("credential" in x["reason"] for x in issues)
    assert all("a" * 30 not in json.dumps(x) for x in issues)


def test_report_exports_only_aggregates_and_refuses_overwrite(tmp_path):
    """Synthetic fixture runs in an isolated directory, never the real evidence tree."""
    import hashlib

    from minideepseek.core import question_metrics

    pytest.importorskip("matplotlib")
    root = tmp_path / "fixture-run"
    root.mkdir()
    (root / "source").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        "<!-- FINAL_RESULTS_START -->pending<!-- FINAL_RESULTS_END -->"
    )
    manifest = {
        "protocol": {"questions": 1, "raw_test_rows": 1},
        "adapters": {"fixture": "abc"},
        "code_sha256": {},
        "model_revision": "fixture-revision",
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    (root / "manifest.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    (root / "evaluation.completed.json").write_text("{}")
    for name, answer in (("base", "4"), ("grpo", "5")):
        responses = ["\\boxed{" + answer + "}"] * 8
        scores = question_metrics(responses, "4")
        row = {
            "id": "fixture",
            "question": "Calculate 2 + 2.",
            "answer": "4",
            "responses": responses,
            "answers": [answer] * 8,
            "truncated": [False] * 8,
            "outcomes": ["correct" if answer == "4" else "incorrect"] * 8,
            "tokens": [3] * 8,
            "seconds": 1,
            **scores,
        }
        metrics = {
            **scores,
            "accuracy_at_1_ci95": [0, 1],
            "answer_extraction_success": 1,
            "truncation_rate": 0,
            "average_generated_tokens": 3,
            "seconds_per_question": 1,
            "seconds_per_response": 0.125,
            "peak_vram_bytes": 1024,
        }
        run = {
            "status": "complete",
            "records": [row],
            "metrics": metrics,
            "adapter": None if name == "base" else "fixture",
            "evaluation_protocol": {},
            "config": {},
            "dataset_sha256": "fixture",
            "resolved_model_revision": "fixture",
            "split": "test",
        }
        (root / f"{name}-test.json").write_text(json.dumps(run))
    script = Path("scripts/phase3_report.py").resolve()
    command = [sys.executable, str(script), "--root", str(root)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    public = tmp_path / "docs/evidence/fixture-run"
    summary = json.loads((public / "summary.json").read_text())
    assert summary["n_completions"] == 16
    assert "responses" not in (public / "summary.json").read_text()
    assert len(list((tmp_path / "docs/final-assets/fixture-run").glob("*.png"))) == 4
    before = (public / "summary.json").read_bytes()
    repeated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert (public / "summary.json").read_bytes() == before
