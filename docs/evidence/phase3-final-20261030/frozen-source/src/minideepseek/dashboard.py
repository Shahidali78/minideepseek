"""Dashboard selection and scoring, independent of Gradio or model loading."""

import json
from pathlib import Path

from .core import correct, numeric, parse_final


def adapter_choices(selection="outputs/phase2/selection.json"):
    path = Path(selection)
    chosen = (
        json.loads(path.read_text())["selected_sft"]
        if path.exists()
        else "outputs/phase2/sft100"
    )
    rl_path = Path("outputs/phase2/rl-selection.json")
    rl = json.loads(rl_path.read_text()) if rl_path.exists() else {}
    return {
        "Base": None,
        "Original SFT": "outputs/sft-run1",
        "Selected SFT (Phase 2)": chosen,
        "Original GRPO": "outputs/grpo-run1",
        "GRPO from selected SFT": rl.get("sft", {}).get(
            "adapter", "outputs/phase2/grpo-sft16"
        ),
        "Base + GRPO": rl.get("base", {}).get("adapter", "outputs/phase2/grpo-base16"),
    }


def available(adapter):
    return adapter is None or all(
        (Path(adapter) / f).exists()
        for f in ["adapter_config.json", "adapter_model.safetensors"]
    )


def score_candidates(responses, reference=None):
    gold = (
        None if reference is None or not str(reference).strip() else numeric(reference)
    )
    return [
        {
            "candidate": i + 1,
            "response": text,
            "final_answer": str(parse_final(text))
            if parse_final(text) is not None
            else None,
            "correct": correct(text, gold) if gold is not None else None,
        }
        for i, text in enumerate(responses)
    ]


def measured_results(root="outputs/phase2"):
    rows = []
    for path in sorted(Path(root).glob("*val-v2.json")):
        result = json.loads(path.read_text())
        if result.get("status") == "complete":
            rows.append(
                {
                    "experiment": path.stem,
                    "peak_memory_complete": result.get("peak_memory_complete", True),
                    **result["metrics"],
                }
            )
    return rows
