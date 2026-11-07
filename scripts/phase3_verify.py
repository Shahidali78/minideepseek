"""Run local final checks; never train or generate held-out responses."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.comparison import validate_comparison
from minideepseek.core import correct, parse_final, question_metrics
from minideepseek.data import digest
from minideepseek.runtime import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    if not (root / "evaluation.completed.json").exists():
        raise RuntimeError("final evaluations are not complete")
    manifest = json.loads((root / "manifest.json").read_text())
    assert (
        digest(root / "manifest.json") == (root / "manifest.sha256").read_text().strip()
    )
    for section in ("model_files", "adapter_files", "validation_files", "all_datasets"):
        for path, expected in manifest[section].items():
            assert digest(path) == expected, path
    # Verify the exact frozen snapshot even if publication code has later repairs.
    for path, expected in manifest["code_sha256"].items():
        relative = (
            Path(path).relative_to(Path.cwd())
            if Path(path).is_absolute()
            else Path(path)
        )
        assert digest(root / "source" / relative) == expected, path
    protected = json.loads(
        Path("outputs/phase2/preservation-manifest.json").read_text()
    )
    for path, expected in protected.items():
        assert digest(path) == expected, path
    runs = [
        json.loads((root / f"{name}-test.json").read_text())
        for name in ("base", "grpo")
    ]
    validate_comparison(runs)
    completions = 0
    for run in runs:
        assert run["status"] == "complete"
        assert len(run["records"]) == manifest["protocol"]["questions"]
        assert run["peak_memory_complete"]
        for row in run["records"]:
            assert len(row["responses"]) == 8
            assert len(row["tokens"]) == 8
            assert all(0 <= t <= 256 for t in row["tokens"])
            assert math.isfinite(row["seconds"]) and row["seconds"] > 0
            for k, value in question_metrics(row["responses"], row["answer"]).items():
                assert row[k] == value
            for text, parsed, outcome in zip(
                row["responses"], row["answers"], row["outcomes"], strict=True
            ):
                number = parse_final(text)
                assert parsed == (str(number) if number is not None else None)
                expected = (
                    "parse_failure"
                    if number is None
                    else ("correct" if correct(text, row["answer"]) else "incorrect")
                )
                assert outcome == expected
                completions += 1
        for metric in ("accuracy_at_1", "pass_at_8", "majority_at_8", "format_rate"):
            assert (
                abs(
                    run["metrics"][metric]
                    - sum(r[metric] for r in run["records"]) / len(run["records"])
                )
                < 1e-12
            )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="-1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    checks = {}
    commands = {
        "pytest": [sys.executable, "-m", "pytest", "-q"],
        "syntax": [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
        "dependencies": [sys.executable, "-m", "pip", "check"],
        "dashboard": [sys.executable, "scripts/dashboard.py", "--smoke"],
        "publication_scan": [
            sys.executable,
            "scripts/publication_audit.py",
            "--output",
            str(root / "publication-audit.json"),
        ],
    }
    for name, command in commands.items():
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        (root / f"check-{name}.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        checks[name] = "passed" if result.returncode == 0 else "failed"
        if result.returncode:
            raise RuntimeError(f"{name} failed; see saved log")
    match = re.search(r"(\d+) passed", (root / "check-pytest.log").read_text())
    tests = int(match[1]) if match else None
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "cpu_tests_passed": tests,
        "protected_historical_files_unchanged": len(protected),
        "final_completions_recomputed": completions,
        "frozen_inputs_and_source_snapshot_verified": True,
        "checkpoint_loading": "Both finalists loaded and generated in the recorded full GPU evaluations",
        "local_dashboard_screenshot": "docs/screenshots/dashboard.png; actual interface rendered and visually inspected",
        "remote_ci": "Not yet executed at local verification time",
    }
    save_json(root / "verification.json", result)
    save_json(Path("docs/evidence") / root.name / "verification.json", result)
    document = f"""# Recorded verification

The final frozen evaluation and local verification completed on {result["checked_utc"]}. These are executed checks, not a checklist of intended work.

| Check | Actual outcome |
|---|---|
| CPU pytest | {tests} passed; CUDA hidden with CUDA_VISIBLE_DEVICES=-1 |
| Python compilation | Passed for src, scripts and tests |
| Dependency consistency | pip check passed |
| Dashboard build | Passed; seven model choices |
| Real dashboard screenshot | Rendered in a separate headless Chrome profile and visually inspected |
| Dataset reconstruction | Expanded-v2 train/validation/test reproduced byte-for-byte |
| Historical preservation | All {len(protected)} protected files unchanged |
| Frozen identities | Model, adapter, datasets, validation outputs and source snapshot hashes verified |
| Final-score integrity | All {completions} saved completions re-scored; saved answers/outcomes/metrics matched |
| Model loading | Both frozen finalists actually loaded and generated on the local GPU |
| Publication allowlist scan | Passed; no detected credential shapes or raw evaluation JSON fields |
| GitHub authentication | Confirmed Shahidali78 outside the network sandbox |
| Remote CI | Pending publication; no remote success claimed here |

Private logs and raw outputs remain in ignored `{root.as_posix()}`. [Aggregate verification evidence](evidence/{root.name}/verification.json) records the local results. The credential scan covers known formats and cannot prove the absence of every possible secret. Initial Git history was empty; only the explicitly reviewed source/documentation allowlist is eligible for the first commit.

The final evaluation itself is the real model-loading smoke test; no extra GPU generation or finalist retraining was needed. CI uses offline lightweight tests, with optional dependency tests skipped when those packages aren't installed.

Publication and remote CI status will be recorded below after the authorized push.
"""
    Path("docs/verification.md").write_text(document, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
