"""Load a saved adapter and execute one inference token; no accuracy claim."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.core import make_prompt
from minideepseek.runtime import load_model, save_json, settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--config", default="configs/pilot.json")
    a = p.parse_args()
    import torch

    cfg = settings(a.config)
    model, tok = load_model(cfg, a.adapter)
    model.eval()
    batch = tok(
        tok.apply_chat_template(
            make_prompt("Calculate 2 + 2."),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg.thinking,
        ),
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        result = model.generate(
            **batch,
            max_new_tokens=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
        )
    assert result.shape[1] == batch["input_ids"].shape[1] + 1
    save_json(
        Path(a.adapter) / "reload-check.json",
        {"status": "loaded_and_generated", "new_tokens": 1},
    )
    print("Adapter loaded and generated successfully:", a.adapter)


if __name__ == "__main__":
    main()
