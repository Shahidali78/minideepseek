import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.data import audit
from minideepseek.runtime import hardware, save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apis", action="store_true")
    a = p.parse_args()
    report = {"hardware": hardware(), "data": audit("data")}
    if a.apis:
        import importlib.metadata

        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

        report["versions"] = {
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "trl", "peft", "accelerate", "datasets"]
        }
        report["apis"] = {
            x.__name__: str(inspect.signature(x))
            for x in [SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer]
        }
        # Instantiate exact configs used by this project, catching API drift before model download.
        SFTConfig(
            output_dir="outputs/api-check",
            max_length=256,
            completion_only_loss=True,
            bf16=False,
            fp16=False,
            report_to="none",
        )
        GRPOConfig(
            output_dir="outputs/api-check",
            num_generations=8,
            generation_batch_size=8,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_completion_length=96,
            bf16=False,
            fp16=False,
            report_to="none",
            use_vllm=False,
        )
    save_json("outputs/preflight.json", report)
    print(json.dumps(report, indent=2))
