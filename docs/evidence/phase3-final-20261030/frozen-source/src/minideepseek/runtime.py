"""Shared experiment configuration, model loading, and provenance."""

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .core import MODEL

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    model: str = MODEL
    revision: str = "main"
    seed: int = 42
    max_length: int = 256
    max_prompt_length: int = 160
    per_device_train_batch_size: int = 1
    save_steps: int = 25
    logging_steps: int = 1
    max_completion_length: int = 96
    learning_rate_sft: float = 0.0001
    learning_rate_grpo: float = 0.000005
    gradient_accumulation_steps: int = 8
    num_generations: int = 8
    temperature: float = 0.6
    top_p: float = 0.95
    beta: float = 0.0
    lora_r: int = 8
    thinking: bool = False

    def validate(self):
        import math

        for name in [
            "seed",
            "max_length",
            "max_prompt_length",
            "per_device_train_batch_size",
            "save_steps",
            "logging_steps",
            "max_completion_length",
            "gradient_accumulation_steps",
            "num_generations",
            "lora_r",
        ]:
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid integer setting: {name}")
        if self.seed >= 2**32:
            raise ValueError("seed must fit uint32")
        for name in [
            "learning_rate_sft",
            "learning_rate_grpo",
            "temperature",
            "top_p",
            "beta",
        ]:
            if not isinstance(getattr(self, name), (int, float)) or not math.isfinite(
                getattr(self, name)
            ):
                raise ValueError(f"invalid numeric setting: {name}")
        if type(self.thinking) is not bool or not self.model or not self.revision:
            raise ValueError("invalid model/chat settings")
        if self.num_generations != 8:
            raise ValueError("this experiment requires exactly eight generations")
        if (self.gradient_accumulation_steps * self.per_device_train_batch_size) % 8:
            raise ValueError("effective generation batch must be divisible by eight")
        if 8 % self.per_device_train_batch_size:
            raise ValueError(
                "microbatch must divide the eight-response generation batch"
            )
        if self.max_length < 32 or self.max_completion_length < 1:
            raise ValueError("invalid sequence length")
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or self.beta < 0:
            raise ValueError("invalid decoding/KL settings")
        if min(self.learning_rate_sft, self.learning_rate_grpo, self.lora_r) <= 0:
            raise ValueError("invalid optimizer/LoRA settings")
        return self


def settings(path=None):
    return Settings(
        **(json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {})
    ).validate()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def hardware():
    import psutil

    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
        )
        gpu_text, gpu_error = gpu.stdout.strip(), gpu.stderr.strip()
    except FileNotFoundError:
        gpu_text, gpu_error = "", "nvidia-smi not found"
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "ram_bytes": psutil.virtual_memory().total,
        "gpu": gpu_text,
        "gpu_error": gpu_error,
    }
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as k:
            result["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except ImportError:
        pass
    try:
        import torch

        result.update(
            torch=torch.__version__,
            cuda_runtime=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
        )
    except ImportError:
        result["torch"] = "not installed"
    return result


def load_model(cfg, adapter=None, train=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    torch.set_num_threads(1)
    set_seed(cfg.seed)
    source = cfg.model
    local = ROOT / "outputs/model"
    if (local / "source.json").exists():
        cached = json.loads((local / "source.json").read_text())
        if cached["model"] == cfg.model and cfg.revision in [
            "main",
            cached["revision"],
        ]:
            source = str(local)
    tok = AutoTokenizer.from_pretrained(source, revision=cfg.revision)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=cfg.revision, dtype=dtype, attn_implementation="sdpa"
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = not train
    return model, tok


def provenance(model, cfg, data_path):
    from .data import digest

    return {
        "config": asdict(cfg),
        "resolved_model_revision": resolved_revision(model, cfg),
        "dataset_sha256": digest(data_path),
        "hardware": hardware(),
    }


def resolved_revision(model, cfg):
    commit = getattr(model.config, "_commit_hash", None)
    if commit:
        return commit
    path = ROOT / "outputs/model/source.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["model"] == cfg.model and cfg.revision in [
            "main",
            cached["revision"],
        ]:
            return cached["revision"]
    return cfg.revision
