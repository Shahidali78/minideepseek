"""Real TRL training with adapter-update and reload verification."""

import argparse
import time
from pathlib import Path

from .core import reward_fn
from .data import grpo_records, sft_records, training_rows
from .diagnostics import completion_info, group_diagnostics, summarize_groups
from .runtime import ROOT, load_model, provenance, save_json, settings


def run(stage):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/pilot.json"))
    p.add_argument("--data", default=str(ROOT / "data"))
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--output", default=f"outputs/{stage}-pilot")
    p.add_argument("--sft-adapter")
    p.add_argument("--from-base", action="store_true")
    p.add_argument("--resume-from")
    p.add_argument("--max-seconds", type=float, default=1200)
    p.add_argument("--log-responses", action="store_true")
    a = p.parse_args()
    if a.max_seconds <= 0:
        p.error("max-seconds must be positive")
    if a.from_base and a.sft_adapter:
        p.error("choose either --from-base or --sft-adapter")
    if a.steps < 1:
        p.error("steps must be positive")
    if stage == "grpo" and not (a.sft_adapter or a.from_base or a.resume_from):
        p.error("GRPO requires --sft-adapter or explicit --from-base")
    if Path(a.output).exists():
        p.error("output already exists; choose a new run directory")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    cfg = settings(a.config)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU pilot requires CUDA-enabled PyTorch")
    rows = training_rows(a.data)
    model, tok = load_model(cfg, a.resume_from or a.sft_adapter, train=True)
    if not (a.sft_adapter or a.resume_from):
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    before = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    if not before or any("lora_" not in n for n in before):
        raise RuntimeError("only intended LoRA parameters may be trainable")
    for row in sft_records(rows):
        prompt = tok.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg.thinking,
        )
        if (
            stage == "sft"
            and len(
                tok(prompt + row["completion"][0]["content"] + tok.eos_token)[
                    "input_ids"
                ]
            )
            > cfg.max_length
        ):
            raise ValueError("SFT target would be truncated; increase max_length")
        if stage == "grpo" and len(tok(prompt)["input_ids"]) > cfg.max_prompt_length:
            raise ValueError(
                "GRPO question would be truncated; increase max_prompt_length"
            )
    model.enable_input_require_grads()
    bf16 = torch.cuda.is_bf16_supported()
    common = dict(
        output_dir=a.output,
        max_steps=a.steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_steps=min(cfg.save_steps, a.steps),
        report_to=["tensorboard"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=not bf16,
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=0,
        optim="adamw_torch",
        save_total_limit=2,
    )
    telemetry = []

    stop_reason = None

    class Monitor(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            nonlocal stop_reason
            if time.monotonic() - start > a.max_seconds:
                stop_reason = "runtime_limit"
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            import subprocess

            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            telemetry.append(
                {
                    "step": state.global_step,
                    "elapsed_seconds": time.monotonic() - start,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_utilization_and_memory": gpu.stdout.strip(),
                    **(logs or {}),
                }
            )

    group_stats = []

    def monitored_reward(completions, answer, **kwargs):
        rewards = reward_fn(completions, answer)
        if len(rewards) % 8:
            raise ValueError("incomplete eight-response reward group")
        for i in range(0, len(rewards), 8):
            prompts = kwargs.get("prompts", [])
            if prompts and len({str(x) for x in prompts[i : i + 8]}) != 1:
                raise ValueError("reward group mixes different questions")
            responses = [
                c[0]["content"] if isinstance(c, list) else c
                for c in completions[i : i + 8]
            ]
            ids = kwargs.get("completion_ids")
            infos = (
                [
                    completion_info(x, {tok.eos_token_id}, cfg.max_completion_length)
                    for x in ids[i : i + 8]
                ]
                if ids is not None
                else None
            )
            diagnostic = group_diagnostics(
                responses,
                answer[i : i + 8],
                [x["tokens"] for x in infos] if infos else None,
                [x["truncated"] for x in infos] if infos else None,
            )
            diagnostic["optimizer_step"] = getattr(
                kwargs.get("trainer_state"), "global_step", None
            )
            if a.log_responses:
                diagnostic["responses"] = responses
            group_stats.append(diagnostic)
            save_json(
                Path(a.output) / "reward-diagnostics.json",
                {"groups": group_stats, "summary": summarize_groups(group_stats)},
            )
        return rewards

    if stage == "sft":
        args = SFTConfig(
            **common,
            learning_rate=cfg.learning_rate_sft,
            max_length=cfg.max_length,
            completion_only_loss=True,
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            processing_class=tok,
            train_dataset=Dataset.from_list(
                [
                    {
                        "prompt": tok.apply_chat_template(
                            r["prompt"],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=cfg.thinking,
                        ),
                        "completion": r["completion"][0]["content"] + tok.eos_token,
                    }
                    for r in sft_records(rows)
                ]
            ),
            callbacks=[Monitor()],
        )
    else:
        args = GRPOConfig(
            **common,
            learning_rate=cfg.learning_rate_grpo,
            num_generations=8,
            generation_batch_size=8,
            max_completion_length=cfg.max_completion_length,
            max_prompt_length=cfg.max_prompt_length,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=0,
            beta=cfg.beta,
            use_vllm=False,
            remove_unused_columns=False,
            scale_rewards="group",
            loss_type="grpo",
        )
        trainer = GRPOTrainer(
            model=model,
            args=args,
            processing_class=tok,
            train_dataset=Dataset.from_list(
                [
                    {
                        "prompt": tok.apply_chat_template(
                            r["prompt"],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=cfg.thinking,
                        ),
                        "answer": r["answer"],
                    }
                    for r in grpo_records(rows)
                ]
            ),
            reward_funcs=monitored_reward,
            callbacks=[Monitor()],
        )
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    metadata = provenance(model, cfg, Path(a.data) / "train.jsonl")
    metadata.update(
        n_training_questions=len(rows),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    )
    try:
        trainer.train(resume_from_checkpoint=a.resume_from)
        torch.cuda.synchronize()
        changed = any(
            not torch.equal(before[n], p.detach().cpu())
            for n, p in model.named_parameters()
            if n in before
        )
        if not changed:
            print(
                "No parameter change: inspect zero-variance reward groups; no learning claim.",
                flush=True,
            )
        trainer.save_model(a.output)
        tok.save_pretrained(a.output)
        # Reload the saved tensors onto a new adapter around the same base model.
        from safetensors.torch import load_file

        saved = load_file(str(Path(a.output) / "adapter_model.safetensors"))
        if not saved or not all(torch.isfinite(v).all() for v in saved.values()):
            raise RuntimeError("invalid saved adapter")
        elapsed = time.monotonic() - start
        result = {
            **metadata,
            "stage": stage,
            "status": "trained"
            if changed
            else "optimizer_steps_without_parameter_change",
            "optimizer_steps": trainer.state.global_step,
            "parameters_changed": changed,
            "training_seconds": elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "logs": telemetry,
            "groups": group_stats,
            "sft_adapter": a.sft_adapter,
            "from_base": a.from_base,
            "resume_from": a.resume_from,
            "stop_reason": stop_reason,
            "reward_summary": summarize_groups(group_stats),
            "training_arguments": args.to_dict(),
        }
        save_json(Path(a.output) / "experiment.json", result)
    except Exception as e:
        save_json(
            Path(a.output) / "failure.json",
            {
                **metadata,
                "status": "failed",
                "error": repr(e),
                "logs": telemetry,
                "groups": group_stats,
            },
        )
        raise
    # Full inference reload in a separate process is done by evaluate.py.
    print(
        f"Saved {stage} adapter after {trainer.state.global_step} optimizer steps: {a.output}"
    )
