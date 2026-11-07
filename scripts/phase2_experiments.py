"""Bounded Phase 2 experiments; never evaluates the reserved test."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.runtime import save_json

ROOT = Path("outputs/phase2")
DATA = "data/expanded-v2"
EVAL = "configs/phase2-eval.json"
commands = (
    json.loads((ROOT / "commands.json").read_text())
    if (ROOT / "commands.json").exists()
    else []
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def command(script, args, log):
    cmd = [sys.executable, "scripts/" + script, *args]
    commands.append(cmd)
    save_json(ROOT / "commands.json", commands)
    print("RUN", script, " ".join(args), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    with (ROOT / log).open("a", encoding="utf-8") as f:
        f.write("\nCOMMAND: " + json.dumps(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)


def evaluate(name, adapter=None):
    out = ROOT / f"{name}-val-v2.json"
    if not out.exists():
        args = [
            "--config",
            EVAL,
            "--data",
            DATA,
            "--max-examples",
            "100",
            "--generation-batch-size",
            "8",
            "--output",
            str(out),
        ]
        if adapter:
            args += ["--adapter", str(adapter)]
        if Path(str(out) + ".partial.json").exists():
            args += ["--resume"]
        command("evaluate.py", args, f"{name}-val-v2.log")
    r = read(out)
    if r.get("status") != "complete" or r["metrics"]["n_questions"] != 100:
        raise RuntimeError("incomplete evaluation: " + name)
    if r["config"] != read(EVAL):
        raise ValueError("evaluation config mismatch")
    return r


def train(stage, name, steps, config, adapter=None, base=False):
    out = ROOT / name
    if not (out / "experiment.json").exists():
        args = [
            "--config",
            config,
            "--data",
            DATA,
            "--steps",
            str(steps),
            "--output",
            str(out),
            "--max-seconds",
            "1200",
        ]
        if adapter:
            args += ["--sft-adapter", str(adapter)]
        if base:
            args += ["--from-base"]
        command("train_" + stage + ".py", args, name + ".log")
    r = read(out / "experiment.json")
    if r["stop_reason"] is not None:
        print("Training stopped:", name, r["stop_reason"], flush=True)
    return out, r


def main():
    global ROOT, commands
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/phase2")
    parser.add_argument(
        "--wait-for-base",
        action="store_true",
        help="Use only when a separate base evaluation is already running",
    )
    args = parser.parse_args()
    ROOT = Path(args.output_root)
    ROOT.mkdir(parents=True, exist_ok=True)
    commands = read(ROOT / "commands.json") if (ROOT / "commands.json").exists() else []
    if args.wait_for_base:
        deadline = time.monotonic() + 1900
        while True:
            try:
                result = read(ROOT / "base-val-v2.json")
                if "metrics" in result:
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("base evaluation not complete")
            time.sleep(1)
    evaluate("base")
    candidates = []
    for name, adapter, budget in [
        ("original-sft", "outputs/sft-run1", 20),
        ("original-grpo", "outputs/grpo-run1", None),
    ]:
        result = evaluate(name, adapter)
        if budget:
            candidates.append((budget, adapter, result))
    for budget in [50, 100]:
        adapter, run = train("sft", f"sft{budget}", budget, "configs/pilot.json")
        result = evaluate(f"sft{budget}", adapter)
        candidates.append((budget, str(adapter), result))
    best = max(
        candidates,
        key=lambda c: (
            c[2]["metrics"]["accuracy_at_1"],
            c[2]["metrics"]["majority_at_8"],
            c[2]["metrics"]["pass_at_8"],
            -c[0],
        ),
    )
    selection = {
        "selected_sft": best[1],
        "optimizer_steps": best[0],
        "criterion": "Accuracy@1, then Majority@8, Pass@8, then smaller budget",
        "scores": [
            {"steps": s, "adapter": p, "metrics": r["metrics"]}
            for s, p, r in candidates
        ],
    }
    save_json(ROOT / "selection.json", selection)
    rl = {}
    for label, adapter, from_base in [("sft", best[1], False), ("base", None, True)]:
        config = "configs/phase2-eval.json"
        pilot, run = train("grpo", f"grpo-{label}-pilot", 4, config, adapter, from_base)
        if run["reward_summary"]["mixed_correctness_fraction"] < 0.25:
            exploration = read(config)
            exploration["temperature"] = 0.9
            config = f"configs/phase2-grpo-{label}-explore.json"
            if not Path(config).exists():
                save_json(config, exploration)
            pilot, run = train(
                "grpo", f"grpo-{label}-pilot-t09", 4, config, adapter, from_base
            )
        if run["reward_summary"]["mixed_correctness_fraction"] >= 0.25:
            chosen, run = train(
                "grpo", f"grpo-{label}16", 16, config, adapter, from_base
            )
        else:
            chosen = pilot
        rl[label] = {
            "adapter": str(chosen),
            "experiment": str(chosen / "experiment.json"),
        }
        evaluate(f"grpo-{label}", chosen)
    save_json(ROOT / "rl-selection.json", rl)
    print("ALL PHASE 2 TRAINING AND VALIDATION RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
