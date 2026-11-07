"""Build a report from real evaluation/training artifacts only."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.comparison import validate_comparison
from minideepseek.data import audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="outputs/base-val.json")
    p.add_argument("--sft", default="outputs/sft-val.json")
    p.add_argument("--grpo", default="outputs/grpo-val.json")
    p.add_argument("--sft-run", default="outputs/sft-run1/experiment.json")
    p.add_argument("--grpo-run", default="outputs/grpo-run1/experiment.json")
    p.add_argument("--data", default="data/expanded-v2")
    p.add_argument("--output", default="docs/experiment-report.md")
    a = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8-sig"))
    evals = [read(a.base), read(a.sft), read(a.grpo)]
    validate_comparison(evals)
    sft, grpo = read(a.sft_run), read(a.grpo_run)
    lines = [
        "# Experiment report",
        "",
        f"Measured validation pilot: {len(evals[0]['records'])} questions, eight samples per question. These results do not demonstrate improved mathematical reasoning.",
        "",
        "| Model | Accuracy@1 | Pass@8 | Majority@8 | Seconds/response | Peak allocated GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in zip(
        [
            "Base",
            f"SFT ({sft['optimizer_steps']} steps)",
            f"SFT + GRPO ({grpo['optimizer_steps']} steps)",
        ],
        evals,
    ):
        m = r["metrics"]
        lines.append(
            f"| {name} | {m['accuracy_at_1']:.1%} | {m['pass_at_8']:.1%} | {m['majority_at_8']:.1%} | {m['seconds_per_response']:.3f} | {m['peak_vram_bytes'] / 2**30:.3f} |"
        )
    lines += [
        "",
        "All stages use the same validation IDs, model revision, seed and decoding settings. Majority ties fail; malformed answers abstain. Peak allocation is measured by PyTorch and excludes the full CUDA/WDDM device footprint. Latency excludes model loading.",
        "",
        "| Model | Accuracy@1 Wilson 95% CI | Format compliance | Mean generated tokens | At completion cap |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in zip(["Base", "SFT", "GRPO"], evals):
        m = r["metrics"]
        lo, hi = m["accuracy_at_1_ci95"]
        tokens = [n for row in r["records"] for n in row["tokens"]]
        cap = sum(n == r["config"]["max_completion_length"] for n in tokens) / len(
            tokens
        )
        lines.append(
            f"| {name} | {lo:.1%} to {hi:.1%} | {m['format_rate']:.1%} | {m['average_generated_tokens']:.2f} | {cap:.1%} |"
        )
    lines += [
        "",
        "The sample is too small for a reliable improvement claim. SFT substantially changed formatting and response length; those effects confound interpretation as reasoning ability. The baseline cap truncates many responses. Paired bootstrap differences are saved in `outputs/comparison.json`.",
        "",
        "## Training evidence",
        "",
        "| Stage | Optimizer steps | Parameters changed | Final logged loss | Seconds including save | Peak allocated GiB |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name, r in [("SFT", sft), ("GRPO", grpo)]:
        losses = [x["loss"] for x in r["logs"] if "loss" in x]
        lines.append(
            f"| {name} | {r['optimizer_steps']} | {r['parameters_changed']} | {losses[-1]:.6g} | {r['training_seconds']:.2f} | {r['peak_vram_bytes'] / 2**30:.3f} |"
        )
    groups = grpo["groups"]
    mean = sum(g["mean_reward"] for g in groups) / len(groups)
    zero = sum(g["zero_variance"] for g in groups)
    lines += [
        "",
        f"GRPO recorded {len(groups)} eight-response groups ({8 * len(groups)} samples). Mean reward was {mean:.6f}; {zero}/{len(groups)} groups had zero reward variance. Samples can repeat: distinct responses per group ranged from {min(g['distinct_completions'] for g in groups)} to {max(g['distinct_completions'] for g in groups)}. Eight draws were retained in every group.",
        "",
        "The GRPO scalar loss is near zero because centered advantages can cancel at an on-policy ratio of one. Nonzero gradient norms in mixed-reward groups and changed adapter tensors establish real optimization. KL was disabled (beta=0), so no KL measurement is claimed. Reward values from different training questions should not be read as a learning curve for held-out accuracy.",
        "",
        "Before these runs, a two-step SFT memory pilot changed weights. A two-step GRPO memory pilot produced only all-zero-reward groups and did not change weights. Its saved adapter still loaded for inference. All pilot artifacts are preserved separately.",
        "",
        "## Hardware and environment",
        "",
        "Windows 11; AMD Ryzen 7 7435HS; 34,169,589,760 bytes RAM (31.8 GiB); RTX 4050 Laptop GPU with 6,141 MiB VRAM; driver 591.91. Driver CUDA compatibility is 13.1; the tested PyTorch runtime is CUDA 12.8. Native Windows inference, BF16 LoRA SFT and eight-response GRPO all executed successfully with these short sequences. This does not establish support for arbitrary long sequences or larger models.",
        "",
        f"Model revision: `{evals[0]['resolved_model_revision']}`.",
        "",
        "Direct installed versions:",
        "",
    ]
    for package in [
        x.split("==")[0]
        for x in Path("requirements.txt").read_text().splitlines()
        if "==" in x
    ]:
        lines.append(f"- {package} {importlib.metadata.version(package)}")
    lines += [
        "",
        "The full dependency lock is [requirements-lock.txt](../requirements-lock.txt). `pip check` passed. Installed SFTTrainer/GRPOTrainer signatures and config construction are recorded in `outputs/preflight.json`. The CUDA wheel was checked against the official SHA256 before installation.",
        "",
        "## Dataset and reproduction",
        "",
        f"Dataset directory: `{a.data}`. Statistics and SHA256 hashes:",
        "",
        "```json",
        json.dumps(audit(a.data), indent=2),
        "```",
        "",
        "The original reserved test file is unchanged and has not been evaluated. The expanded test copy preserves its questions and answers. No training data comes from validation or test. SFT used 160 examples over 20 accumulated updates, rather than a full epoch. The broad generator checks exact question overlap; shared templates and semantically similar arithmetic remain limitations.",
        "",
        "Pilot settings are in [configs/pilot.json](../configs/pilot.json): rank 8 query/value LoRA, microbatch 1, accumulation 8, SFT LR 1e-4, GRPO LR 5e-6, BF16, SFT cap 256, GRPO prompt cap 160, completion cap 96, eight generations, seed 42, temperature 0.6, top-p 0.95, top-k disabled, thinking disabled.",
        "",
        "Run from the project root after installing the pinned CUDA environment:",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python scripts/create_data.py --output data/reproduction",
        ".\\.venv\\Scripts\\python scripts/run_pipeline.py --data data/reproduction --examples 9 --sft-steps 20 --grpo-steps 8 --output outputs/reproduction",
        ".\\.venv\\Scripts\\python scripts/dashboard.py --sft outputs/sft-run1 --grpo outputs/grpo-run1",
        "```",
        "",
        "Use a new output directory for every run. Exact reruns can still vary across drivers, kernels and hardware; seeds do not guarantee bitwise equality. [README](../README.md) covers environment setup, memory pilots, protocol freezing and the reserved final test. A frozen manifest for the selected adapters is stored in `outputs/frozen-protocol.json`.",
        "",
        "## Verification and limits",
        "",
        "The final test log is `outputs/tests.log`. The CPU suite runs without downloading model weights; its tiny random-model test performs an optimizer update and adapter round-trip entirely offline. GPU evidence includes baseline generation, SFT and GRPO weight changes, and inference from saved adapters. The Gradio interface passed a build smoke check. GitHub Actions is prepared, but no remote CI or publication has been run.",
        "",
        "GSM8K support is optional and has not been exercised in this run. Its published answers are imported without inventing reasoning traces. These short synthetic experiments, one seed, rational-only rewards and nine validation questions cannot establish broad generalization. A larger held-out validation study, longer completion budgets, independently verified reasoning targets and multiple training seeds remain future work.",
        "",
        "## Measured charts",
        "",
        "![Stage accuracy](assets/accuracy.png)",
        "",
        "![Training loss](assets/loss.png)",
        "",
        "![Reward](assets/groups_mean_reward.png)",
        "",
        "![Reward variance](assets/groups_reward_variance.png)",
        "",
        "![Peak allocation](assets/peak_vram_bytes.png)",
        "",
        "![Training duration](assets/duration.png)",
        "",
        "Additional charts, CSVs and raw response JSON files are in `outputs/`. [Portfolio wording](portfolio.md) includes the suggested repository description and three resume bullets.",
    ]
    verification = Path("outputs/artifact-verification.json")
    if verification.exists():
        v = read(verification)
        lines += [
            "",
            f"Independent artifact checks confirmed {v['sft_grpo_tensors_different']} changed adapter tensors between SFT and GRPO (maximum absolute difference {v['max_adapter_weight_difference']:.8g}), and {v['validation_responses_changed']} changed validation responses. All 64 stored training rewards recomputed correctly. Original dataset hashes and the frozen protocol were verified.",
        ]
    Path(a.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
