"""Publish aggregate evidence only after both frozen held-out runs complete."""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.comparison import comparison
from minideepseek.core import question_metrics
from minideepseek.data import digest
from minideepseek.final_analysis import error_summary
from minideepseek.runtime import save_json


def percent(x):
    return f"{100 * x:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(root / "manifest.json") != (root / "manifest.sha256").read_text().strip():
        raise ValueError("manifest was modified")
    if not (root / "evaluation.completed.json").exists():
        raise RuntimeError("both frozen evaluations must complete first")
    runs = [
        json.loads((root / f"{name}-test.json").read_text())
        for name in ("base", "grpo")
    ]
    n = manifest["protocol"]["questions"]
    for run in runs:
        if run["status"] != "complete" or len(run["records"]) != n:
            raise ValueError("incomplete final evaluation")
        for row in run["records"]:
            scores = question_metrics(row["responses"], row["answer"])
            if any(row[k] != v for k, v in scores.items()):
                raise ValueError("saved scores differ from recomputation")
    paired = comparison(runs)
    errors = {
        name: error_summary(run["records"])
        for name, run in zip(("Base", "Base + GRPO"), runs)
    }
    evidence = {
        "run_id": root.name,
        "manifest_sha256": digest(root / "manifest.json"),
        "n_unique_questions": n,
        "n_raw_rows": manifest["protocol"]["raw_test_rows"],
        "n_completions": n * 8 * 2,
        "metrics": {
            name: run["metrics"] for name, run in zip(("Base", "Base + GRPO"), runs)
        },
        "paired_comparison": paired,
        "errors": errors,
        "input_hashes_verified_after_evaluation": True,
        "raw_output_sha256": {
            f"{name}-test.json": digest(root / f"{name}-test.json")
            for name in ("base", "grpo")
        },
    }
    public = Path("docs/evidence") / root.name
    public.mkdir(parents=True, exist_ok=False)
    save_json(public / "summary.json", evidence)
    frozen = dict(manifest)
    frozen["code_sha256"] = {
        (
            Path(path).relative_to(Path.cwd()).as_posix()
            if Path(path).is_absolute()
            else Path(path).as_posix()
        ): value
        for path, value in manifest["code_sha256"].items()
    }
    frozen["adapters"] = {
        "outputs/phase2/grpo-base16": next(iter(manifest["adapters"].values()))
    }
    frozen["private_manifest_sha256"] = digest(root / "manifest.json")
    frozen["public_redaction"] = (
        "Absolute adapter path replaced with repository-relative path; raw completions and question-level outcomes excluded. Original manifest remains private and hashed."
    )
    save_json(public / "manifest.json", frozen)
    (public / "manifest.sha256").write_text(digest(public / "manifest.json") + "\n")
    shutil.copytree(root / "source", public / "frozen-source")
    save_json(root / "analysis.json", evidence)
    review = {}
    for name, run in zip(("Base", "Base + GRPO"), runs):
        failures = [row for row in run["records"] if not row["accuracy_at_1"]][:4]
        review[name] = [
            {
                "id": row["id"],
                "question": row["question"],
                "reference": row["answer"],
                "first_response": row["responses"][0],
                "extracted": row["answers"][0],
                "truncated": row["truncated"][0],
                "outcome": row["outcomes"][0],
                "observation": "First candidate fails final-answer scoring; no hidden reasoning cause inferred.",
            }
            for row in failures
        ]
    save_json(root / "failure-review.json", review)
    validation = []
    for path in sorted(Path("outputs/phase2").glob("*val-v2.json")):
        run = json.loads(path.read_text())
        validation.append(
            {
                "experiment": path.stem,
                "adapter": run["adapter"],
                "metrics": run["metrics"],
            }
        )
    save_json(public / "validation-summary.json", validation)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "figure.dpi": 140})
    assets = Path("docs/final-assets") / root.name
    assets.mkdir(parents=True, exist_ok=False)
    names = ["Base", "Base + GRPO"]
    colors = ["#376f9d", "#bc652c"]
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for j, (name, run) in enumerate(zip(names, runs)):
        values = [
            100 * run["metrics"][k]
            for k in ("accuracy_at_1", "pass_at_8", "majority_at_8")
        ]
        bars = ax.bar(
            [i + (j - 0.5) * 0.36 for i in range(3)],
            values,
            width=0.36,
            label=name,
            color=colors[j],
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set(
        xticks=range(3),
        xticklabels=["Accuracy@1", "Pass@8", "Majority@8"],
        ylim=(0, 114),
        ylabel="Questions (%)",
        title=f"Frozen held-out evaluation · {n} unique questions",
    )
    ax.set_title(f"Frozen held-out evaluation · {n} unique questions", pad=50)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    fig.savefig(assets / "final-comparison.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (name, run) in enumerate(zip(names, runs)):
        value = run["metrics"]["accuracy_at_1"]
        low, high = run["metrics"]["accuracy_at_1_ci95"]
        ax.errorbar(
            100 * value,
            i,
            xerr=[[100 * (value - low)], [100 * (high - value)]],
            fmt="o",
            capsize=5,
            color=colors[i],
            label=name,
        )
    ax.set(
        yticks=[0, 1],
        yticklabels=names,
        xlabel="Accuracy@1 (%) · Wilson 95% interval",
        title="Held-out accuracy uncertainty",
    )
    fig.savefig(assets / "accuracy-uncertainty.png", bbox_inches="tight")
    plt.close(fig)
    labels = [
        "arithmetic_error",
        "answer_formatting_failure",
        "failed_answer_extraction",
        "truncated_completion",
        "unsupported_answer_pattern",
    ]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    for j, name in enumerate(names):
        ax.barh(
            [i + (j - 0.5) * 0.36 for i in range(len(labels))],
            [errors[name]["all_samples"][k] for k in labels],
            height=0.36,
            color=colors[j],
            label=name,
        )
    ax.set(
        yticks=range(len(labels)),
        yticklabels=[x.replace("_", " ") for x in labels],
        xlabel="Incorrect completions (exclusive operational labels)",
        title="Observable held-out errors",
    )
    ax.legend()
    fig.savefig(assets / "error-analysis.png", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, metric, title in zip(
        axes,
        ("seconds_per_response", "peak_vram_bytes"),
        ("Amortized seconds / sample", "Peak allocated GPU GiB"),
    ):
        values = [
            r["metrics"][metric] / (2**30 if metric == "peak_vram_bytes" else 1)
            for r in runs
        ]
        bars = ax.bar(names, values, color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(title=title, ylim=(0, max(values) * 1.2))
    fig.suptitle("Measured held-out inference resources · excludes model loading")
    fig.savefig(assets / "inference-resources.png", bbox_inches="tight")
    plt.close(fig)
    rows = []
    for name, run in zip(names, runs):
        m = run["metrics"]
        low, high = m["accuracy_at_1_ci95"]
        rows.append(
            f"| {name} | {percent(m['accuracy_at_1'])} | {percent(low)}–{percent(high)} | {percent(m['pass_at_8'])} | {percent(m['majority_at_8'])} |"
        )
    table = (
        "| Frozen finalist | Accuracy@1 | Wilson 95% CI | Pass@8 | Majority@8 |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(rows)
    )
    delta = paired["differences"][0]["accuracy_at_1"]
    gain = 100 * delta["difference"]
    ci = [100 * x for x in delta["paired_bootstrap_ci95"]]
    p = delta["mcnemar_exact_p_two_sided"]
    statistical = (
        f"GRPO minus Base: **{gain:+.2f} percentage points**; {delta['corrected_questions']} improved, "
        f"{delta['regressed_questions']} worsened, {n - delta['discordant_questions']} unchanged first-answer outcomes. "
        f"Two-sided exact paired McNemar p={p:.6g}. Paired empirical bootstrap 95% interval: [{ci[0]:+.2f}, {ci[1]:+.2f}] percentage points."
    )
    interpretation = (
        "The exact paired test rejects equal marginal correctness at the conventional 5% threshold for this task and sample set. It does not establish broad reasoning improvement."
        if p < 0.05
        else "The exact paired test does not provide conventional 5% significance evidence of improved first-answer accuracy. This does not establish equivalence."
    )
    efficiency = "| Finalist | Extraction success | Truncation | Mean tokens | Seconds / batch | Seconds / sample | Peak allocated GiB |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for name, run in zip(names, runs):
        m = run["metrics"]
        efficiency += f"| {name} | {percent(m['answer_extraction_success'])} | {percent(m['truncation_rate'])} | {m['average_generated_tokens']:.2f} | {m['seconds_per_question']:.3f} | {m['seconds_per_response']:.3f} | {m['peak_vram_bytes'] / 2**30:.3f} |\n"
    error_table = "| Operational label | Base (all samples) | Base + GRPO (all samples) |\n|---|---:|---:|\n"
    for label in errors[names[0]]["all_samples"]:
        error_table += f"| {label.replace('_', ' ')} | {errors[names[0]]['all_samples'][label]} | {errors[names[1]]['all_samples'][label]} |\n"
    adapter_sha = next(iter(manifest["adapters"].values()))
    report = f"""# MiniDeepSeek final research report

## Abstract

MiniDeepSeek tests whether binary verifiable rewards can improve a 0.6B language model on synthetic arithmetic using a 6 GB laptop GPU. Seven exploratory validation conditions led to two frozen finalists: unmodified Qwen3-0.6B and its 16-update LoRA GRPO adapter. They were evaluated once on {n} unique held-out questions, eight samples per question. {statistical} {interpretation}

This is a small-model GRPO research experiment, not a reproduction of the full DeepSeek-R1 training system.

## Research question and background

Can short, local GRPO training improve final-answer accuracy without requiring a learned reward model? GRPO samples a group of responses and centers their verifiable rewards to form relative advantages. Groups with identical rewards supply no within-group correctness signal. This implementation uses TRL 0.23.1, group reward scaling and loss_type=grpo, with beta=0 (no reference-policy KL penalty). The binary reward gives 1 for a safely parsed, mathematically equivalent final value and 0 otherwise. Neither verbosity nor plausible intermediate text earns reward. See the [versioned TRL documentation](https://huggingface.co/docs/trl/v0.23.1/grpo_trainer).

## Model, hardware and checkpoint identities

The unmodified checkpoint is `Qwen/Qwen3-0.6B`, revision `{manifest["model_revision"]}`. Base means the requested unmodified release; its [model card](https://huggingface.co/Qwen/Qwen3-0.6B) describes post-training, so it is not a pretraining-only checkpoint. The selected adapter is `outputs/phase2/grpo-base16`, SHA256 `{adapter_sha}`. No finalist was retrained or tuned during Phase 3.

Native Windows 11, AMD Ryzen 7 7435HS, approximately 31.8 GiB system RAM and NVIDIA RTX 4050 Laptop GPU (6141 MiB). Python 3.12.8; torch 2.8.0+cu128; transformers 4.56.2; TRL 0.23.1; PEFT 0.17.1. BF16 and SDPA were used. The [frozen manifest](evidence/{root.name}/manifest.json) records every environment version and checkpoint/file hash. The private original manifest SHA256 is `{evidence["manifest_sha256"]}`; the public copy replaces the absolute adapter path with a relative path.

## Dataset, splits and separation

The existing synthetic-v2 data has 1200 unique training questions and 150 unique validation questions; exploratory evaluation used a fixed 100-question validation prefix. The reserved file contains 240 rows, with six duplicate questions: **all {n} first-occurrence unique questions** were selected before model outcomes. Duplicate rows were not additional independent observations. Nothing was added to or removed from the original dataset files.

The held-out set is addition, subtraction and multiplication with operands in the original 71–110 range. It is narrower than the nine-category training/validation generator. Exact and NFKC/case/whitespace-normalized overlap audits found no held-out question matches against any saved training or validation version. Repeated templates and operand structures still limit independence and distributional generalization. The test was previously read for schema/reference-integrity and split-exclusion checks; no saved test generations, reward training, model selection or tuning were found before this freeze. This is an audit of available artifacts, not proof about all possible historical access.

The generator reproduces expanded-v2 train, validation and test files byte-for-byte in the recorded local reconstruction check. The project-generated synthetic arithmetic is distinct from externally licensed benchmarks; no GSM8K data was used for these reported results. Raw datasets are excluded from publication.

## Training methodology and reward

Historical SFT budgets were 20, 50 and 100 actual optimizer updates using verified final-answer-only targets. SFT100 was selected by predeclared validation Accuracy@1, then Majority@8, then Pass@8, then smaller budget. Original SFT+GRPO used eight updates; new selected-SFT and unmodified-model GRPO conditions each used 16 actual updates after separate four-update signal pilots. LoRA rank8, alpha16, q_proj/v_proj targets, dropout0; AdamW; microbatch1, accumulation8; SFT learning rate1e-4, GRPO5e-6. The two GRPO initializations have equal update/sample budgets, not equal compute because response lengths differ.

The bounded rational-expression parser accepts final boxed or answer-tag values, fractions, decimals, signs, percentages and a small arithmetic AST. It rejects arbitrary names, calls, powers and malformed/multiple answer fields. There is no eval/exec path. Gold answers enter scoring, never prompts. Accepting a mathematically equivalent expression does not itself prove the model computed its value.

## Frozen experimental design

Both finalists used identical non-thinking chat prompts, temperature0.6, top_p0.95, top_k0, 256 completion tokens and batches of eight. Question i receives seed42+i independently for each model. Paired seeds reduce a source of variation but do not make token sequences or random trajectories identical. No outcome-dependent cap changes or reruns were performed.

Accuracy@1 is correctness of the first stochastic candidate, not greedy accuracy. Pass@8 is the observed presence of at least one correct candidate among eight. Majority@8 scores the unique modal parsed value; ties/no parsed values fail and invalid samples abstain. Parser success and numeric correctness are separate: an extracted value may be wrong. Completion truncation is measured using EOS-aware token IDs, independently of correctness.

The equal finite completion cap can still censor longer responses differently. Truncation rates below quantify the effect; an equal cap does not establish absence of length bias. Full outputs remain in ignored `{root.as_posix()}`. A durable exclusive start marker prevents accidental reruns of this final experiment.

## Exploratory validation evidence

| Model | Accuracy@1 | Pass@8 | Majority@8 |
|---|---:|---:|---:|
| Base | 79% | 96% | 91% |
| Original SFT20 | 52% | 63% | 53% |
| Original SFT + GRPO8 | 50% | 64% | 52% |
| SFT50 | 61% | 72% | 63% |
| SFT100 | 71% | 79% | 70% |
| SFT100 + GRPO16 | 70% | 79% | 69% |
| Base + GRPO16 | 83% | 98% | 90% |

These 5600 completions supported model selection; they are not independent confirmation. The +4-point first-answer gain for Base+GRPO was four corrections with no regressions, exact paired p=0.125. SFT100 improved 19 points over SFT20 but stayed below the unmodified model. Answer-only adaptation changed observed output length and format compliance. A narrow training distribution, limited updates and the supervised objective are plausible explanations for underperformance, but no ablation establishes a causal explanation. Base-initialized GRPO was selected for its observed validation accuracy and stronger mixed-reward support, not on final-test outcomes. The [Phase 2 report](phase2-experiment-report.md) preserves all seven conditions and their uncertainty.

![Exploratory validation accuracy](phase2-assets/model_accuracy.png)

## Final held-out results

{table}

![Final comparison](final-assets/{root.name}/final-comparison.png)

## Paired statistical analysis

{statistical}

{interpretation} The p-value is not the probability that GRPO is better. The primary inferential comparison is Accuracy@1; other metrics are descriptive, without multiplicity-adjusted claims. Pass@8 difference: {100 * paired["differences"][0]["pass_at_8"]["difference"]:+.2f} points. Majority@8 difference: {100 * paired["differences"][0]["majority_at_8"]["difference"]:+.2f} points.

Wilson intervals describe each accuracy proportion. The paired bootstrap resamples whole questions, retaining model pairing, with 5000 draws and seed42. Sparse discordance can make an empirical bootstrap optimistic or degenerate; it cannot invent unobserved failures. Exact McNemar conditions on discordant pairs and uses a two-sided binomial test ([reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)). Shared synthetic templates weaken the assumption that questions are independent draws. A single group of eight samples per question does not estimate variability over training or sampling seeds. Failing to reject is not evidence of equivalence; even a small p-value here would not establish general mathematical reasoning.

![Accuracy intervals](final-assets/{root.name}/accuracy-uncertainty.png)

## Reward diagnostics

During the selected Base+GRPO16 training, 9/16 groups had mixed correctness; 7/16 (43.75%) had zero reward variance. Mean reward was0.7265625. In SFT100+GRPO16, only4/16 were mixed and12/16 (75%) zero variance; mean reward0.625. Each group had eight genuine sampled outputs. Centered advantages can yield near-zero on-policy scalar loss despite nonzero gradients. Saved adapters changed and finite weights were verified. These are training observations, not held-out accuracy evidence. [Reward analysis](grpo-reward-analysis.md) includes measured variance plots.

## GPU efficiency

{efficiency}

Latency synchronizes CUDA and includes decoding but excludes model loading; seconds/sample is batch time divided by eight, not interactive latency. GPU memory is torch peak allocated, not total process usage, cache reservation or total device memory. Base+GRPO16 training used about1.678GiB peak allocated and366.45 seconds including save on this laptop. Historical SFT100 inference memory covers only its resumed final11 questions; that partial peak is not compared as a complete measurement.

![Inference resources](final-assets/{root.name}/inference-resources.png)

## Observable failure analysis

{error_table}

Labels are mutually exclusive for incorrect completions: truncation first, then extraction/format patterns, then parsed numeric mismatch. Arithmetic error means an incorrect extracted final value on an arithmetic question; it does not identify an unseen internal step. A plain number without the required field is labeled formatting failure. A malformed/unsupported boxed or answer-tag field is an unsupported answer pattern; other missing fields are extraction failures. Independent truncation and parsing metrics can overlap, unlike this table. Operation-selection and multi-step-reasoning categories remain zero because final-value mismatches alone do not establish those causes. This test contains only single-operation arithmetic, so it cannot meaningfully test multi-step reasoning.

Representative failures are reviewed privately in `failure-review.json`; [the aggregate evidence](evidence/{root.name}/summary.json) publishes counts, not raw test completions. No explanation of hidden reasoning is inferred from prose.

![Error categories](final-assets/{root.name}/error-analysis.png)

## Reproducibility and verification

See [reproducibility](reproducibility.md) for Windows setup, controlled commands, checkpoints and one-run safeguards. Frozen source snapshots, original input hashes, exact generation commands, synchronized timings and complete raw responses remain local. The public aggregate JSON records hashes of those raw outputs, allowing authorized artifact holders to verify them. Checkpoints are deliberately not redistributed, so the public repository supports procedural reconstruction, not downloading this exact private adapter. A later training reconstruction may differ numerically across kernels/drivers.

Actual check outcomes and publication status are recorded in [verification](verification.md). The final eval itself loaded both checkpoints and performed real GPU inference. No retraining occurred after test exposure.

## Limitations and conclusion

Read the final-test evidence separately from the exploratory validation table. The final set is larger than a12-question pilot, but still consists of only {n} template-based arithmetic questions. Its restricted operation/operand range, question dependence, single training seed, single stochastic sample group, fixed completion cap and semantic answer checker limit the claim. The name MiniDeepSeek does not imply reproduction of DeepSeek-R1's data, architecture, scale or training recipe.

{interpretation} The defensible deliverable is a measured local SFT/GRPO pipeline with a frozen, auditable held-out comparison. Future studies should preregister multiple training and evaluation seeds, include independently sourced/licensed multi-step tasks, use disjoint template families and test cap sensitivity on a new validation set. The used final set must not become a tuning set for another claimed confirmation.
"""
    Path("docs/final-research-report.md").write_text(report, encoding="utf-8")
    # The uniquely named report is never overwritten; the top-level document is its public entry point.
    snapshot = re.sub(
        r"\]\((?!https?://)([^)]+)\)", lambda m: "](../../" + m[1] + ")", report
    )
    (public / "final-research-report.md").write_text(snapshot, encoding="utf-8")
    private_snapshot = re.sub(
        r"\]\((?!https?://)([^)]+)\)", lambda m: "](../../docs/" + m[1] + ")", report
    )
    (root / "final-research-report.md").write_text(private_snapshot, encoding="utf-8")
    Path("docs/final-results.md").write_text(
        "# Frozen held-out results\n\n"
        + table
        + "\n\n"
        + statistical
        + "\n\n"
        + interpretation
        + "\n",
        encoding="utf-8",
    )
    portfolio = f"""# MiniDeepSeek portfolio case study

## The problem I investigated

I wanted to test whether verifiable rewards could improve a small language model using hardware I could run locally. MiniDeepSeek trains Qwen3-0.6B with LoRA SFT and eight-response GRPO on a native Windows laptop with a 6 GB RTX 4050. The difficult part was separating formatting changes, sampling effects and model selection from evidence of better answers.

## What was built

The project has deterministic synthetic data generation, split audits, a bounded rational-answer checker, actual SFT/GRPO training, reward-variance diagnostics and a local Gradio comparison dashboard. Seven conditions produced 5,600 validation completions. The two selected finalists were then frozen by checkpoint, source, data and protocol hashes before a single held-out evaluation on {n} unique questions, eight candidates/model ({n * 16} held-out completions).

## Measured outcomes

Validation favored the unmodified-model GRPO initialization: 83% Accuracy@1 versus 79% for the unmodified model, exact paired p=0.125. SFT100 improved over SFT20 but remained below the unmodified model. Those results guided selection and weren't treated as independent confirmation.

The frozen held-out result was {percent(runs[0]["metrics"]["accuracy_at_1"])} for Base and {percent(runs[1]["metrics"]["accuracy_at_1"])} for Base+GRPO. {statistical} {interpretation}

{table}

## Engineering decisions and tradeoffs

LoRA kept local training feasible. A binary exact-value reward avoided a learned judge but limited evaluation to supported rational answers. Reward-group logs exposed zero-variance updates; final evaluation separated wrong extracted values from parse failures and truncation. CPU-only CI avoids accidental GPU training. Models, raw completions and private datasets stay out of GitHub.

The final dataset has 234 unique questions from 240 rows and only single-operation arithmetic. Shared templates, one seed and finite output lengths limit what the result says about broader reasoning. I would use a new independently held-out multi-step benchmark and multiple seeds before making a broader claim. See the [full research report](final-research-report.md) and [limitations](limitations.md).

## Three résumé bullets

- Built a reproducible Qwen3-0.6B fine-tuning pipeline with PEFT LoRA, supervised training and eight-response GRPO on a 6 GB RTX 4050 laptop GPU, with verified adapter updates and reward-variance diagnostics.
- Compared seven configurations using 5,600 validation completions, then froze two finalists for a {n}-question held-out evaluation; reported Accuracy@1, Pass@8, Majority@8, paired confidence intervals and exact McNemar statistics.
- Implemented safe rational-answer rewards, dataset contamination checks, EOS-aware truncation metrics and a Gradio evidence dashboard; documented the observed held-out accuracy difference of {gain:+.2f} points with p={p:.6g}, without overstating generalization.

## LinkedIn project announcement draft — not posted

I completed MiniDeepSeek, a local experiment in fine-tuning Qwen3-0.6B with supervised learning and GRPO on a 6 GB laptop GPU.

I compared seven configurations on 100 validation questions, then froze the two finalists before evaluating 234 unique held-out arithmetic questions with eight samples per model. Final Accuracy@1: {percent(runs[0]["metrics"]["accuracy_at_1"])} for the unmodified model and {percent(runs[1]["metrics"]["accuracy_at_1"])} after GRPO. The exact paired comparison gave p={p:.6g}. {interpretation}

The project includes a safe numeric reward checker, reward-variance diagnostics, reproducible manifests, GPU measurements and a local dashboard. It is a small-model research experiment, not a reproduction of the full DeepSeek-R1 system. The report discusses failed SFT conditions and uncertainty alongside the measured outcomes.

Repository: https://github.com/Shahidali78/minideepseek

## Short interview explanation

“I tested whether a small model could improve on arithmetic using binary verifiable rewards on a laptop. I implemented LoRA SFT and GRPO with eight sampled answers per question, audited the data and measured whether updates actually changed the adapter. The validation study selected GRPO initialized from the unmodified model. I then froze everything and ran the held-out comparison once. The first-answer accuracies were {percent(runs[0]["metrics"]["accuracy_at_1"])} and {percent(runs[1]["metrics"]["accuracy_at_1"])}, with exact paired p={p:.6g}. I distinguish that narrow result from broad reasoning ability, and I keep uncertainty, truncation and extraction failures visible.”

## Repository description and topics

Description: Local Qwen3-0.6B LoRA SFT/GRPO research on a 6 GB GPU, with frozen paired evaluation, verifiable rewards and an evidence dashboard.

Five topics: `reinforcement-learning`, `grpo`, `llm-finetuning`, `lora`, `mathematical-reasoning`.
"""
    Path("docs/portfolio-case-study.md").write_text(portfolio, encoding="utf-8")
    save_json(
        Path("docs/publication-metadata.json"),
        {
            "repository": "Shahidali78/minideepseek",
            "description": "Local Qwen3-0.6B LoRA SFT/GRPO research on a 6 GB GPU, with frozen paired evaluation, verifiable rewards and an evidence dashboard.",
            "topics": [
                "reinforcement-learning",
                "grpo",
                "llm-finetuning",
                "lora",
                "mathematical-reasoning",
            ],
        },
    )
    readme_path = Path("README.md")
    readme = readme_path.read_text(encoding="utf-8")
    readme = re.sub(
        r"<!-- FINAL_RESULTS_START -->.*?<!-- FINAL_RESULTS_END -->",
        "<!-- FINAL_RESULTS_START -->\n"
        + table
        + "\n\n"
        + statistical
        + "\n\n"
        + interpretation
        + f"\n\nBoth models used {n} unique held-out questions and eight paired-seed samples per question. "
        f"The original 240-row file was preserved; six repeated questions were excluded before outcomes. "
        f"See [the final report](docs/final-research-report.md) and [aggregate evidence](docs/evidence/{root.name}/summary.json).\n"
        f"\n![Frozen held-out comparison](docs/final-assets/{root.name}/final-comparison.png)\n<!-- FINAL_RESULTS_END -->",
        readme,
        flags=re.S,
    )
    readme_path.write_text(readme, encoding="utf-8")
    print(json.dumps({"public_evidence": str(public), "comparison": paired}, indent=2))


if __name__ == "__main__":
    main()
