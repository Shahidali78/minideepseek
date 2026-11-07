"""Plot only measurements present in experiment files."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--evaluations", nargs="*", default=[])
    p.add_argument("--output", default="outputs/charts")
    a = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(Path(p).read_text(encoding="utf-8-sig")) for p in a.runs]
    for field in [
        "loss",
        "reward",
        "reward_std",
        "kl",
        "peak_vram_bytes",
        "completion_length",
    ]:
        fig, ax = plt.subplots()
        found = False
        for run in runs:
            points = [(r["step"], r[field]) for r in run.get("logs", []) if field in r]
            if field == "completion_length":
                points = [
                    (r["step"], r["completions/mean_length"])
                    for r in run.get("logs", [])
                    if "completions/mean_length" in r
                ]
            if points:
                ax.plot(*zip(*points), label=run["stage"])
                found = True
        if found:
            ax.set(xlabel="Optimizer step", ylabel=field)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"{field}.png")
        plt.close(fig)
    for field in ["mean_reward", "reward_std", "reward_variance", "zero_variance"]:
        fig, ax = plt.subplots()
        found = False
        for run in runs:
            groups = run.get("groups", [])
            if groups:
                ax.plot([g[field] for g in groups], label=run["stage"])
                found = True
        if found:
            ax.set(xlabel="Reward group", ylabel=field)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"groups_{field}.png")
        plt.close(fig)
    fig, ax = plt.subplots()
    ax.bar([r["stage"] for r in runs], [r["training_seconds"] for r in runs])
    ax.set_ylabel("Training seconds")
    fig.tight_layout()
    fig.savefig(out / "duration.png")
    plt.close(fig)
    evals = [json.loads(Path(p).read_text(encoding="utf-8-sig")) for p in a.evaluations]
    if evals:
        from minideepseek.comparison import validate_comparison

        validate_comparison(evals)
        fig, ax = plt.subplots()
        for field in ["accuracy_at_1", "pass_at_8", "majority_at_8"]:
            labels = (
                ["Base", "SFT", "GRPO"]
                if len(evals) == 3
                else [
                    Path(r["adapter"]).name if r["adapter"] else "Base" for r in evals
                ]
            )
            values = [r["metrics"][field] for r in evals]
            if all(field + "_ci95" in r["metrics"] for r in evals):
                bounds = [r["metrics"][field + "_ci95"] for r in evals]
                ax.errorbar(
                    labels,
                    values,
                    yerr=[
                        [v - b[0] for v, b in zip(values, bounds)],
                        [b[1] - v for v, b in zip(values, bounds)],
                    ],
                    marker="o",
                    capsize=3,
                    label=field,
                )
            else:
                ax.plot(labels, values, marker="o", label=field)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Validation pilot (n={len(evals[0]['records'])}; 95% Wilson intervals)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "accuracy.png")
        plt.close(fig)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
