"""Local measured-results dashboard with separate interactive predictions."""

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.core import make_prompt
from minideepseek.dashboard import (
    adapter_choices,
    available,
    measured_results,
    score_candidates,
)
from minideepseek.runtime import load_model, settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/phase2-eval.json")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--sft")
    p.add_argument("--grpo")
    a = p.parse_args()
    import gradio as gr
    import torch

    cfg = settings(a.config)
    choices = adapter_choices()
    if a.sft:
        choices["Selected SFT (Phase 2)"] = a.sft
    if a.grpo:
        choices["GRPO from selected SFT"] = a.grpo
    choices["SFT 50 (historical)"] = "outputs/phase2/sft50"

    def final_results():
        summaries = sorted(Path("docs/evidence").glob("*/summary.json"))
        if not summaries:
            return {"status": "Frozen held-out evaluation has not been exported yet."}
        evidence = json.loads(summaries[-1].read_text())
        return {
            "run_id": evidence["run_id"],
            "unique_questions": evidence["n_unique_questions"],
            "metrics": evidence["metrics"],
            "paired_comparison": evidence["paired_comparison"],
            "warning": "One stochastic sample set on template-based arithmetic; see the report for uncertainty and limitations.",
        }

    def gpu_status():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,memory.free",
                    "--format=csv",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "GPU status unavailable. CPU inference may be slow; measured results remain viewable."

    def generate(question, stage, reference, candidates):
        if not question.strip():
            return {"error": "Enter a mathematical question."}
        adapter = choices[stage]
        if not available(adapter):
            return {"error": f"{stage} checkpoint is unavailable."}
        try:
            score_candidates([], reference)
        except (ValueError, ZeroDivisionError):
            return {"error": "Reference must be a supported numeric expression."}
        model = None
        try:
            model, tok = load_model(cfg, adapter)
            model.eval()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            prompt = tok.apply_chat_template(
                make_prompt(question),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=cfg.thinking,
            )
            batch = tok(prompt, return_tensors="pt").to(model.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                y = model.generate(
                    **batch,
                    num_return_sequences=int(candidates),
                    max_new_tokens=cfg.max_completion_length,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=0,
                    pad_token_id=tok.pad_token_id,
                )
            responses = tok.batch_decode(
                y[:, batch["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return {
                "kind": "Interactive predictions, not benchmark measurements",
                "model": stage,
                "candidates": score_candidates(responses, reference),
                "batch_seconds": time.monotonic() - started,
                "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available()
                else None,
            }
        except torch.OutOfMemoryError:
            return {
                "error": "GPU memory exhausted. Use one candidate or close other GPU workloads."
            }
        except (OSError, ValueError) as exc:
            return {"error": f"Model could not be loaded or used: {exc}"}
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with gr.Blocks(
        analytics_enabled=False,
        title="MiniDeepSeek | Local GRPO Research",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# MiniDeepSeek\n### Small-model GRPO research on a 6 GB laptop GPU\n"
            "Compare frozen evidence, inspect uncertainty, or sample your own mathematical question. "
            "This is a local research experiment, not a reproduction of the full DeepSeek-R1 training system."
        )
        with gr.Tab("Try a question"):
            question = gr.Textbox(
                label="Mathematical question", value="Calculate 12 + 7."
            )
            gr.Examples(
                [
                    ["Calculate 12 + 7."],
                    ["What is 25 percent of 80?"],
                    ["Solve for x: 2*x + 4 = 18."],
                ],
                inputs=question,
            )
            with gr.Row():
                stage = gr.Dropdown(list(choices), value="Base", label="Model")
                candidates = gr.Radio([1, 8], value=8, label="Candidate responses")
            reference = gr.Textbox(
                label="Optional reference answer (used only for scoring)"
            )
            response = gr.JSON(
                label="Interactive predictions · extracted answers and optional scores"
            )
            gr.Button("Generate", variant="primary").click(
                generate, [question, stage, reference, candidates], response
            )
            memory = gr.Textbox(
                label="Live GPU memory (MiB)", value=gpu_status(), interactive=False
            )
            gr.Button("Refresh GPU status").click(gpu_status, outputs=memory)
            gr.Markdown(
                "Missing adapters return an availability message. If memory runs out, try one candidate or close other GPU workloads. Interactive samples do not update benchmark scores."
            )
        with gr.Tab("Final held-out evidence"):
            final = gr.JSON(
                label="Frozen final comparison and paired uncertainty",
                value=final_results(),
            )
            gr.Button("Refresh final results").click(final_results, outputs=final)
            gr.Gallery(
                label="Actual final evaluation charts",
                value=[str(p) for p in Path("docs/final-assets").glob("*/*.png")],
                columns=2,
            )
        with gr.Tab("Exploratory validation"):
            measured = gr.JSON(
                label="Measured validation performance · used for selection",
                value=measured_results(),
            )
            gr.Button("Refresh validation results").click(
                measured_results, outputs=measured
            )
            gr.Gallery(
                label="Measured validation charts",
                value=[str(p) for p in Path("docs/phase2-assets").glob("*.png")],
                columns=2,
            )
        with gr.Tab("Methodology and limits"):
            gr.Markdown(
                "## Read the scores with their uncertainty\n"
                "Accuracy@1 is the first stochastic candidate. Pass@8 means at least one correct candidate. "
                "Majority@8 uses the unique modal parsed value; ties fail and invalid values abstain. "
                "Gold answers are used only for scoring. Both finalists share prompts, paired seeds, temperature 0.6, top-p 0.95, and a 256-token cap.\n\n"
                "The held-out set has 240 rows and 234 unique questions; repeated questions were excluded before outcomes. "
                "It covers synthetic addition, subtraction and multiplication. Shared templates, one training seed, one sample group and finite completion lengths limit generalization. "
                "Validation selected the finalists and is exploratory. A p-value is not the probability that one model is better. "
                "GPU peaks are allocated memory, not total device consumption; batch latency excludes loading.\n\n"
                "See docs/final-research-report.md, docs/reproducibility.md and docs/limitations.md in the repository."
            )
    if a.smoke:
        print(
            json.dumps(
                {
                    "status": "built",
                    "components": len(demo.get_config_file()["components"]),
                    "models": list(choices),
                }
            )
        )
        return
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
