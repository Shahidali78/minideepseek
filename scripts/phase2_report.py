"""Measured Phase 2 plots, paired comparisons, and research report."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.comparison import paired_interval, validate_comparison
from minideepseek.data import audit, load_rows
from minideepseek.runtime import save_json

ROOT = Path("outputs/phase2")
LABELS = {
    "base": "Base",
    "original-sft": "Original SFT 20",
    "original-grpo": "Original SFT + GRPO 8",
    "sft50": "SFT 50",
    "sft100": "SFT 100",
    "grpo-sft": "Selected SFT + GRPO",
    "grpo-base": "Base + GRPO",
}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/phase2")
    args = parser.parse_args()
    ROOT = Path(args.root)
    tag = ROOT.name
    runs = {k: read(ROOT / f"{k}-val-v2.json") for k in LABELS}
    validate_comparison(list(runs.values()))
    if any(
        r["status"] != "complete" or r["metrics"]["n_questions"] != 100
        for r in runs.values()
    ):
        raise ValueError("all 100-question evaluations required")
    selection = read(ROOT / "selection.json")
    rl = read(ROOT / "rl-selection.json")
    sftkey = {20: "original-sft", 50: "sft50", 100: "sft100"}[
        selection["optimizer_steps"]
    ]
    training = {}
    for path in sorted(ROOT.glob("*/experiment.json")):
        training[path.parent.name] = read(path)
    training["original-sft20"] = read("outputs/sft-run1/experiment.json")
    preferred = [
        "original-sft20",
        "sft50",
        "sft100",
        "grpo-sft-pilot",
        "grpo-sft16",
        "grpo-base-pilot",
        "grpo-base16",
    ]
    training = {k: training[k] for k in preferred if k in training} | {
        k: v for k, v in training.items() if k not in preferred
    }

    paired = []
    for a, b in [
        ("base", sftkey),
        ("original-sft", "original-grpo"),
        ("original-sft", "sft50"),
        ("original-sft", "sft100"),
        (sftkey, "grpo-sft"),
        ("base", "grpo-base"),
    ]:
        paired.append(
            {
                "reference": a,
                "candidate": b,
                **{
                    f: paired_interval(runs[a]["records"], runs[b]["records"], f)
                    for f in ["accuracy_at_1", "pass_at_8", "majority_at_8"]
                },
            }
        )
    save_json(ROOT / "paired-comparisons.json", paired)
    rows = [
        {
            "model": LABELS[k],
            "peak_memory_complete": r.get("peak_memory_complete", True),
            **r["metrics"],
        }
        for k, r in runs.items()
    ]
    save_json(
        ROOT / "comparison.json",
        {"rows": rows, "paired": paired, "selection": selection},
    )
    import csv

    with (ROOT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(plots / (name + ".png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    for field, name in [
        ("accuracy_at_1", "model_accuracy"),
        ("pass_at_8", "pass_at_8"),
        ("majority_at_8", "majority_at_8"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        vals = [r["metrics"][field] for r in runs.values()]
        ci = [r["metrics"][field + "_ci95"] for r in runs.values()]
        ax.bar(range(len(vals)), vals, color="#356a92", alpha=0.8)
        ax.errorbar(
            range(len(vals)),
            vals,
            yerr=[
                [v - c[0] for v, c in zip(vals, ci)],
                [c[1] - v for v, c in zip(vals, ci)],
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        ax.set_xticks(range(len(vals)), LABELS.values(), rotation=25, ha="right")
        ax.set(
            ylim=(0, 1),
            ylabel=field,
            title="100 validation questions; 95% Wilson intervals",
        )
        finish(fig, name)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ["original-sft20", "sft50", "sft100"]:
        log = training[key]["logs"]
        points = [(x["step"], x["loss"]) for x in log if "loss" in x]
        ax.plot(*zip(*points), label=key)
    ax.set(
        xlabel="Optimizer step",
        ylabel="Completion-only training loss",
        title="Independent SFT budgets",
    )
    ax.legend()
    finish(fig, "sft_loss")
    fig, ax = plt.subplots(figsize=(6, 4))
    budgets = [20, 50, 100]
    vals = [
        runs[k]["metrics"]["accuracy_at_1"] for k in ["original-sft", "sft50", "sft100"]
    ]
    cis = [
        runs[k]["metrics"]["accuracy_at_1_ci95"]
        for k in ["original-sft", "sft50", "sft100"]
    ]
    ax.errorbar(
        budgets,
        vals,
        yerr=[
            [v - c[0] for v, c in zip(vals, cis)],
            [c[1] - v for v, c in zip(vals, cis)],
        ],
        marker="o",
        capsize=4,
    )
    ax.set(
        ylim=(0, 1),
        xlabel="Independent SFT optimizer-step budget",
        ylabel="Validation Accuracy@1",
        title="100 questions; selection uses this validation set",
    )
    finish(fig, "validation_budgets")
    for field, name in [
        ("mean_reward", "grpo_reward"),
        ("reward_std", "grpo_reward_std"),
        ("zero_variance", "zero_variance_groups"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for stage, entry in rl.items():
            groups = read(entry["experiment"])["groups"]
            values = [g[field] for g in groups]
            if field == "zero_variance":
                values = [
                    100 * sum(values[: i + 1]) / (i + 1) for i in range(len(values))
                ]
            ax.plot(range(1, len(values) + 1), values, marker=".", label=stage)
        ax.set(
            xlabel="Generation group",
            ylabel="Cumulative zero-variance %" if field == "zero_variance" else field,
            title="Training questions vary across groups",
        )
        ax.legend()
        finish(fig, name)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].bar(
        list(LABELS.values()),
        [r["metrics"]["peak_vram_bytes"] / 2**30 for r in runs.values()],
    )
    for bar, r in zip(axes[0].patches, runs.values()):
        if not r.get("peak_memory_complete", True):
            bar.set_hatch("///")
            bar.set_label("Resumed segment only")
    axes[0].legend()
    axes[0].set_ylabel("Inference peak (GiB)")
    axes[1].bar(
        list(training), [t["peak_vram_bytes"] / 2**30 for t in training.values()]
    )
    axes[1].set_ylabel("Training peak (GiB)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    finish(fig, "gpu_memory")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(list(training), [t["training_seconds"] for t in training.values()])
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Training + save seconds")
    finish(fig, "training_duration")
    # Fine-grained task breakdown is descriptive: around eleven questions per category.
    categories = {
        r["id"]: r["category"] for r in load_rows("data/expanded-v2/val.jsonl")
    }
    breakdown = {}
    for key, r in runs.items():
        breakdown[key] = {
            str(c): {
                "n": sum(categories[x["id"]] == c for x in r["records"]),
                "correct": sum(
                    x["accuracy_at_1"] for x in r["records"] if categories[x["id"]] == c
                ),
            }
            for c in range(9)
        }
    save_json(ROOT / "category-results.json", breakdown)
    table = [
        "| Model | Accuracy@1 [95% CI] | Pass@8 | Majority@8 | Invalid | Truncated | Amortized s/sample | Peak allocated GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, r in runs.items():
        m = r["metrics"]
        lo, hi = m["accuracy_at_1_ci95"]
        peak_marker = "*" if not r.get("peak_memory_complete", True) else ""
        table.append(
            f"| {LABELS[key]} | {m['accuracy_at_1']:.1%} [{lo:.1%}, {hi:.1%}] | {m['pass_at_8']:.1%} | {m['majority_at_8']:.1%} | {m['invalid_answer_rate']:.1%} | {m['truncation_rate']:.1%} | {m['seconds_per_response']:.3f} | {m['peak_vram_bytes'] / 2**30:.3f}{peak_marker} |"
        )
    table += [
        "",
        "Memory note: an asterisk marks SFT100 inference memory measured only in its resumed final segment; the earlier peak was unavailable.",
    ]
    lines = [
        "# Phase 2 experiment report",
        "",
        "## Research objective",
        "",
        "Test whether a larger SFT budget and more informative eight-response GRPO groups improve final-answer accuracy of the requested Qwen3-0.6B checkpoint on existing validation data. No held-out final test was evaluated. A correct answer does not establish a correct reasoning trace.",
        "",
        "## Historical context and diagnosis",
        "",
        "The original nine-question Base/SFT/GRPO Accuracy@1 scores were 44.4%, 33.3% and 33.3%; Pass@8 was 66.7%, 55.6% and 55.6%. The base hit its 96-token cap on 30/72 outputs. SFT learned much shorter, more consistently formatted responses. The old GRPO run had 75% zero-variance groups. Re-scoring old raw outputs with the new parser recovered no extra correct responses. See [diagnosis](phase2-diagnosis.md) and [reward analysis](grpo-reward-analysis.md).",
        "",
        "These historical scores must not be compared directly with Phase 2 as a training gain: question count, completion budget, parser capability, sampling batch and CPU-thread setting changed. Phase 2 model comparisons below share one evaluation protocol.",
        "",
        "## Environment and data",
        "",
        "The original project environment was retained. RTX 4050 Laptop GPU, 6,141 MiB VRAM; AMD Ryzen 7 7435HS; 31.8 GiB RAM; native Windows; Python 3.12.8; PyTorch 2.8.0+cu128, BF16. Full dependency versions remain in [requirements-lock.txt](../requirements-lock.txt). Native PyTorch generation is used without vLLM, FlashAttention extensions or cloud resources.",
        "",
        "Data: 1,200 unique synthetic training examples and 150 unique validation examples across nine categories. The fixed first 100 validation questions were used, with no exact or commuted-arithmetic overlap with training. Existing files were not rewritten. A hash manifest protects 96 historical datasets, reports, charts and adapter files. Shared templates and unknown pretraining exposure remain limitations. Equivalent numeric expressions are accepted, so exact-division prompts can sometimes be answered by repeating an input fraction. This is semantic final-value correctness, not proof that the model performed the computation.",
        "",
        '"Base" means the unmodified requested model, which already includes post-training according to its [model card](https://huggingface.co/Qwen/Qwen3-0.6B), rather than a pretraining-only release. Non-thinking mode remains fixed; this is not a long-thinking R1 replication.',
        "",
        "## Reward and evaluation",
        "",
        "Rewards are 1 for a verified final answer and 0 otherwise, with no length or formatting bonus. The bounded rational-expression parser supports signed numbers, fractions, decimals, percentages and arithmetic equivalence. It uses a strict AST whitelist, never model code execution. Multiple final-answer fields are rejected. References are a separate reward column and are not included in generation prompts.",
        "",
        "Eight samples per question, seed 42 plus question index, temperature 0.6, top-p 0.95, top-k disabled, completion cap 256, generation batch eight and one CPU thread. Prompt chat templates and model revision are identical across model comparisons. EOS-aware truncation is separate from parsing failure and parsed incorrect answers. Saved token counts exclude padding after EOS. Majority@8 uses the unique most frequent normalized valid answer; ties fail and invalid answers abstain. It is a plurality vote, not a requirement for five correct answers out of eight. The quoted per-sample time is batch time divided by eight, a throughput measure; batch/question latency is also saved and is more relevant to interactive waiting.",
        "",
        "A three-question smoke test finished with no truncated outputs and peak PyTorch allocation below 1.4 GiB. All full comparisons below contain 100 questions and 800 sampled completions per model. Wilson intervals describe per-question metrics; paired bootstrap differences use matched questions, 5,000 resamples and a fixed seed. These intervals do not account for every dependence induced by shared templates.",
        "",
        "## Validation measurements",
        "",
        *table,
        "",
        f"Selected SFT: `{selection['selected_sft']}` ({selection['optimizer_steps']} steps). Selection rule was declared before full results: Accuracy@1, then Majority@8, Pass@8, then the smaller training budget. Selection and reporting use the same validation set, so this is a tuning result, not an unbiased final-test estimate.",
        "",
        "## Training budgets and evidence",
        "",
        "The original 20-step adapter was preserved. Fresh 50- and 100-step SFT runs used the same verified answer-only labels, seed, rank-8 query/value LoRA, AdamW, microbatch 1, accumulation 8 and initial LR 1e-4. Completion-only masking was checked with an offline trainer fixture. Loss decreases alone do not establish reasoning improvement. Different step budgets imply different linear learning-rate schedules.",
        "",
        "GRPO used four-step pilots from the selected SFT and from the unmodified model. A separate temperature-0.9 pilot was allowed if fewer than 25% of groups had mixed correctness; otherwise temperature remained 0.6. Sixteen-step final runs were allowed only when mixed groups supplied learning signal. The training completion cap was 256; LR was 5e-6; beta was zero. Initialization comparisons can also differ in the selected exploration setting, and are not a pure causal ablation if that fallback was used.",
        "",
        "| Run | Optimizer steps | Parameters changed | Seconds incl. save | Peak allocated GiB | Mean event-sampled GPU utilization |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name, t in training.items():
        samples = []
        for log in t["logs"]:
            try:
                samples.append(float(log["gpu_utilization_and_memory"].split(",")[0]))
            except (KeyError, ValueError):
                pass
        util = f"{sum(samples) / len(samples):.1f}%" if samples else "Unavailable"
        lines.append(
            f"| {name} | {t['optimizer_steps']} | {t['parameters_changed']} | {t['training_seconds']:.1f} | {t['peak_vram_bytes'] / 2**30:.3f} | {util} |"
        )
    lines += [
        "",
        "GPU utilization is sampled at logging events, not continuously. PyTorch peak allocation excludes driver/display memory. Timing includes checkpoint saving and logging, excludes loading and preprocessing, and is not a controlled speed benchmark against the historical run.",
        "",
        "## Reward variance and paired effects",
        "",
    ]
    for label, entry in rl.items():
        t = read(entry["experiment"])
        g = t["reward_summary"]
        lines.append(
            f"{label}: {g['n_groups']} groups; zero variance {g['zero_variance_fraction']:.1%}; any correct {g['any_correct_fraction']:.1%}; mixed correctness {g['mixed_correctness_fraction']:.1%}; mean correct responses {g['average_correct_count']:.3f}/8. Adapter parameters changed: {t['parameters_changed']}."
        )
        lines.append("")
    for pair in paired:
        d = pair["accuracy_at_1"]
        lo, hi = d["paired_bootstrap_ci95"]
        lines.append(
            f"{LABELS[pair['candidate']]} minus {LABELS[pair['reference']]}: {100 * d['difference']:+.1f} percentage points, paired bootstrap 95% interval [{100 * lo:+.1f}, {100 * hi:+.1f}]; corrected {d['corrected_questions']}, regressed {d['regressed_questions']}; exact paired McNemar p={d['mcnemar_exact_p_two_sided']:.4g}."
        )
    lines += [
        "",
        "Near-zero GRPO scalar loss can occur with centered advantages and nonzero gradients. Weight-change and gradient checks distinguish real optimization from zero-variance stalls. Training reward varies with the sampled question, so a rising reward curve alone is not evidence of validation improvement. KL was disabled and no measured KL curve is claimed.",
        "",
        "## Verification and reproduction",
        "",
        "Offline tests cover rational/percentage equivalence, malformed and speculative outputs, AST exploitation, group statistics, EOS truncation, question selection, dashboard scoring, completion-only masking, adapter round-trip and checkpoint resume. Ordinary CI only runs CPU tests; it does not train the full model. Targeted GPU evidence consists of the actual training and adapter inference runs. Test logs, dependency checks and integrity verification are stored under `outputs/phase2/`.",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python scripts/evaluate.py --config configs/phase2-eval.json --data data/expanded-v2 --max-examples 100 --generation-batch-size 8 --output outputs/phase2/base-val-v2.json",
        ".\\.venv\\Scripts\\python scripts/phase2_experiments.py",
        ".\\.venv\\Scripts\\python scripts/phase2_report.py",
        ".\\.venv\\Scripts\\python scripts/dashboard.py",
        ".\\.venv\\Scripts\\python -m pytest -q",
        "```",
        "",
        "The commands reproduce the fixed local protocol. Complete artifacts are reused and never overwritten; use `scripts/phase2_experiments.py --output-root outputs/phase2-reproduction` and `scripts/phase2_report.py --root outputs/phase2-reproduction` for a fresh versioned run. The runner records every executed training/evaluation command in `outputs/phase2/commands.json`. Interrupted evaluations use `--resume` and verify protocol/dataset/adapter hashes. Trainer checkpoints can resume into a new output directory using `--resume-from`, with the original planned hyperparameters. Runtime limits are 30 minutes per evaluation and 20 minutes per training run.",
        "",
        "## Scientific limits and next direction",
        "",
        "This is one seed, one small already-post-trained model and a narrow synthetic dataset. Final-answer-only SFT may suppress useful reasoning, while a strict single-answer format can penalize otherwise correct calculations. A 256-token cap can still censor responses; measured truncation rates above quantify this. Validation selection introduces optimism, multiple comparisons add uncertainty, and there is no final-test result. Absence of a convincing gain is a valid outcome, not evidence that GRPO can never help.",
        "",
        "A useful paper direction is a controlled study of format collapse, output censoring and reward-gradient starvation in small-model RL. Future experiments should independently vary answer-only versus verified explanatory targets, completion budget and mixed-reward group frequency, using multiple seeds and a separately reserved benchmark. Do not infer valid reasoning from a numerical reward alone.",
        "",
        "## Measured plots",
        "",
    ]
    for name in [
        "sft_loss",
        "grpo_reward",
        "grpo_reward_std",
        "zero_variance_groups",
        "validation_budgets",
        "model_accuracy",
        "pass_at_8",
        "gpu_memory",
        "training_duration",
    ]:
        lines += [f"![{name}]({tag}-assets/{name}.png)", ""]
    details = [
        "## Extraction, length and latency",
        "",
        "| Model | Extracted answers | Parsed incorrect samples | Mean generated tokens | Mean eight-sample batch seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, r in runs.items():
        m = r["metrics"]
        details.append(
            f"| {LABELS[key]} | {m['answer_extraction_success']:.2%} | {m['parsed_incorrect_rate']:.2%} | {m['average_generated_tokens']:.2f} | {m['seconds_per_question']:.3f} |"
        )
    details += [
        "",
        "Batch latency and amortized sample time are distinct; neither is a controlled speed comparison with the original serial nine-question run.",
        "",
    ]
    at = lines.index("## Training budgets and evidence")
    lines[at:at] = details
    best_accuracy = max(r["metrics"]["accuracy_at_1"] for r in runs.values())
    best_labels = [
        LABELS[k]
        for k, r in runs.items()
        if r["metrics"]["accuracy_at_1"] == best_accuracy
    ]
    lines += [
        "",
        f"Highest observed Accuracy@1 was {best_accuracy:.1%}, attained by {', '.join(best_labels)}. This is a validation observation with the confidence intervals and selection limitations stated above, not a final-test claim.",
    ]
    transition_path = ROOT / "sft-transition-analysis.json"
    if transition_path.exists():
        transition = read(transition_path)
        lines += [
            "",
            f"On the paired first samples, SFT100 corrected {transition['parsed_wrong_to_correct']} numerically wrong SFT20 answers and {transition['unparseable_to_correct']} unparseable answers, while losing {transition['correct_to_wrong']} previously correct answers. The net Accuracy@1 change is not explained merely by extracting more boxes; output lengths stayed similar. Internal reasoning quality remains unverified.",
        ]
    lines += [
        "",
        "SFT100 evaluation resumed after 89 saved questions following an interruption. Its earlier peak-memory counter was unavailable: the recorded SFT100 inference peak covers only the final 11-question segment, not the full run. Other completed inference runs report their full recorded peak. Per-question seeds, responses and timings were preserved; no suspension-sized timing outlier was present in the saved records.",
    ]
    lines += [
        "",
        "The Base + GRPO Accuracy@1 gain is four corrected questions and no regressions. Its bootstrap interval is positive, but only four outcomes differ and the two-sided exact paired McNemar test gives p=0.125. This does not provide conventional 5% significance evidence of a general improvement. The empirical bootstrap cannot introduce an unobserved regression, so small-discordance intervals need particular caution. Pass@8 rose from 96% to 98% while Majority@8 fell from 91% to 90%; the metrics do not show uniform dominance. Multiple seeds and independent held-out evaluation are needed. The exact test uses the binomial distribution conditional on discordant pairs, as described in the [statsmodels McNemar documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).",
    ]
    verification_path = ROOT / "final-verification.json"
    if verification_path.exists():
        verification = read(verification_path)
        lines += [
            "",
            f"Recorded final verification: {verification['cpu_tests_passed']} CPU tests passed; {verification['measured_completions']:,} measured completions re-scored across {verification['completed_models']} models; {verification['verified_plots']} plots verified; all {verification['historical_files_unchanged']} protected historical files unchanged. Dependency consistency, compilation, static checks and the Gradio build smoke check passed. The reserved final test was not evaluated.",
        ]
    Path(f"docs/{tag}-experiment-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    assets = Path(f"docs/{tag}-assets")
    assets.mkdir(exist_ok=True)
    import shutil

    for image in plots.glob("*.png"):
        shutil.copyfile(image, assets / image.name)
    save_json(ROOT / "data-audit.json", audit("data/expanded-v2"))
    if tag != "phase2":
        print(
            f"Report saved to docs/{tag}-experiment-report.md without changing the main README."
        )
        return
    readme = Path("README.md")
    original = readme.read_text(encoding="utf-8-sig")
    start = "<!-- PHASE2_RESULTS_START -->"
    end = "<!-- PHASE2_RESULTS_END -->"
    section = [
        start,
        "## Phase 2 validation study",
        "",
        "Seven configurations were evaluated on the same 100 validation questions with eight samples each and a 256-token completion cap. These are validation/tuning measurements; the reserved final test remains unused.",
        "",
        "| Model | Accuracy@1 | Pass@8 | Majority@8 |",
        "|---|---:|---:|---:|",
    ]
    for key, r in runs.items():
        m = r["metrics"]
        section.append(
            f"| {LABELS[key]} | {m['accuracy_at_1']:.1%} | {m['pass_at_8']:.1%} | {m['majority_at_8']:.1%} |"
        )
    section += [
        "",
        f"The selected SFT budget was {selection['optimizer_steps']} steps. The highest observed Accuracy@1 was {best_accuracy:.1%} ({', '.join(best_labels)}). Selection uses validation data, and small changes require the paired intervals before interpretation.",
        "",
        "Base + GRPO corrected four first answers and regressed none; exact paired p=0.125. The observed +4-point gain is promising but not conclusive evidence of general improvement.",
        "",
        "[Phase 2 research report](docs/phase2-experiment-report.md) contains confidence intervals, reward-variance analysis, actual configurations, runtime, memory, failure analysis and reproduction commands. [Diagnosis](docs/phase2-diagnosis.md), [reward analysis](docs/grpo-reward-analysis.md), and [portfolio summary](docs/phase2-portfolio.md) are saved separately from the historical report.",
        "",
        "![Phase 2 measured accuracy](docs/phase2-assets/model_accuracy.png)",
        "",
        end,
        "",
    ]
    if verification_path.exists():
        section.insert(
            -2,
            f"Recorded Phase 2 verification: {verification['cpu_tests_passed']} CPU tests passed; {verification['verified_plots']} measured plots verified; all {verification['historical_files_unchanged']} protected historical files unchanged.",
        )
    block = "\n".join(section)
    if start in original:
        left, right = original.split(start, 1)
        old, tail = right.split(end, 1)
        original = left + block + tail
    else:
        original = original.replace(
            "## Measured local pilot", block + "\n## Historical nine-question pilot", 1
        )
    original = original.replace(
        "Eight samples are generated serially during evaluation to fit VRAM.",
        "The historical pilot sampled serially. Phase 2 uses a fixed eight-sample generation batch, recorded in the evaluation protocol; amortized throughput and batch latency are saved separately.",
    )
    readme.write_text(original, encoding="utf-8")
    portfolio = [
        "# Phase 2 portfolio summary",
        "",
        "Studied final-answer SFT, output censoring and sparse group-relative rewards in an already-post-trained small model. The study kept original artifacts intact, separated validation tuning from the reserved test, and compared candidate correctness support rather than treating lower loss as better reasoning.",
        "",
        f"The highest observed validation Accuracy@1 was {best_accuracy:.1%}; see the paired intervals and selection caveats in the [research report](phase2-experiment-report.md). This does not establish general mathematical reasoning improvement.",
        "",
        "Suggested repository description",
        "",
        "Reproducible Qwen3-0.6B experiments on a 6 GB GPU: controlled math validation, LoRA SFT budgets, eight-candidate GRPO, safe numeric rewards and reward-variance diagnostics.",
        "",
        "Three resume-ready achievements",
        "",
        "- Ran matched 100-question, eight-sample evaluations across seven Qwen3-0.6B configurations and controlled SFT budget comparisons on a native Windows RTX 4050 6 GB.",
        "- Implemented safe rational-expression and percentage rewards, EOS-aware truncation diagnostics, group-relative advantage statistics and paired uncertainty estimates without using the reserved final test.",
        "- Built versioned experiment automation, resumable evaluation and checkpoints, artifact-integrity verification, offline tests and a Gradio dashboard for candidate-level model comparison.",
        "",
        "Research-paper direction",
        "",
        "A controlled study of format collapse, output censoring and reward-gradient starvation in small-model RL. Compare verified explanatory targets with answer-only SFT, multiple completion budgets and equal-compute inference voting; repeat across seeds and an independently reserved benchmark.",
    ]
    Path("docs/phase2-portfolio.md").write_text(
        "\n".join(portfolio) + "\n", encoding="utf-8"
    )
    print("Phase 2 report and measured plots saved.")


if __name__ == "__main__":
    main()
