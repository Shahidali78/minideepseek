"""Run three controlled validation pilots, stopping immediately on failure."""

import argparse
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/pilot.json")
p.add_argument("--data", default="data")
p.add_argument("--examples", type=int, default=8)
p.add_argument("--sft-steps", type=int, default=2)
p.add_argument("--grpo-steps", type=int, default=2)
p.add_argument("--output", default="outputs/pilot")
a = p.parse_args()
out = Path(a.output)
if out.exists():
    raise FileExistsError("choose a new run directory")
out.mkdir(parents=True)
common = ["--config", a.config, "--data", a.data]


def run(script, args):
    subprocess.run([sys.executable, "scripts/" + script, *common, *args], check=True)


run(
    "evaluate.py",
    ["--max-examples", str(a.examples), "--output", str(out / "base-val.json")],
)
run("train_sft.py", ["--steps", str(a.sft_steps), "--output", str(out / "sft")])
run(
    "evaluate.py",
    [
        "--adapter",
        str(out / "sft"),
        "--max-examples",
        str(a.examples),
        "--output",
        str(out / "sft-val.json"),
    ],
)
run(
    "train_grpo.py",
    [
        "--sft-adapter",
        str(out / "sft"),
        "--steps",
        str(a.grpo_steps),
        "--output",
        str(out / "grpo"),
    ],
)
run(
    "evaluate.py",
    [
        "--adapter",
        str(out / "grpo"),
        "--max-examples",
        str(a.examples),
        "--output",
        str(out / "grpo-val.json"),
    ],
)
subprocess.run(
    [
        sys.executable,
        "scripts/plot_results.py",
        "--runs",
        str(out / "sft/experiment.json"),
        str(out / "grpo/experiment.json"),
        "--evaluations",
        str(out / "base-val.json"),
        str(out / "sft-val.json"),
        str(out / "grpo-val.json"),
        "--output",
        str(out / "charts"),
    ],
    check=True,
)

subprocess.run(
    [
        sys.executable,
        "scripts/compare_results.py",
        str(out / "base-val.json"),
        str(out / "sft-val.json"),
        str(out / "grpo-val.json"),
        "--output",
        str(out / "comparison.json"),
    ],
    check=True,
)
