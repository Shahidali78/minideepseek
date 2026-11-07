import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.core import (
    PARSER_VERSION,
    correct,
    make_prompt,
    parse_final,
    question_metrics,
)
from minideepseek.data import audit, digest, load_rows
from minideepseek.diagnostics import completion_info, select_validation
from minideepseek.runtime import load_model, provenance, save_json, settings


def wilson(k, n):
    z = 1.96
    den = 1 + z * z / n
    center = (k / n + z * z / (2 * n)) / den
    width = z * ((k / n * (1 - k / n) + z * z / (4 * n)) / n) ** 0.5 / den
    return [center - width, center + width]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--adapter")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--data", default="data")
    p.add_argument("--max-examples", type=int, default=30)
    p.add_argument("--output", default="outputs/base-val.json")
    p.add_argument("--frozen-protocol")
    p.add_argument("--generation-batch-size", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-seconds", type=float, default=1800)
    a = p.parse_args()
    if a.max_seconds <= 0:
        p.error("max-seconds must be positive")
    cfg = settings(a.config)
    path = Path(a.data) / f"{a.split}.jsonl"
    if Path(a.output).exists():
        p.error("output already exists")
    if a.max_examples < 1:
        p.error("max-examples must be positive")
    if a.split == "test":
        if not a.frozen_protocol:
            p.error("reserved test requires --frozen-protocol")
        from minideepseek.protocol import verify_frozen

        verify_frozen(a.frozen_protocol, a.config, a.data, a.adapter)
    audit(a.data)
    rows = load_rows(path)
    rows = select_validation(rows, a.max_examples)
    import torch
    from transformers import set_seed

    model, tok = load_model(cfg, a.adapter)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    metadata = provenance(model, cfg, path)
    if a.adapter:
        metadata["adapter_sha256"] = digest(
            Path(a.adapter) / "adapter_model.safetensors"
        )
    metadata["evaluation_protocol"] = {
        "version": 2,
        "parser": PARSER_VERSION,
        "generation_batch_size": a.generation_batch_size,
        "selection": "first-unique-prefix",
        "cpu_threads": torch.get_num_threads(),
    }
    partial = Path(a.output + ".partial.json")
    records = []
    prior_peak = 0
    metadata["peak_memory_complete"] = True
    if partial.exists():
        if not a.resume:
            p.error("partial run exists; use --resume")
        previous = json.loads(partial.read_text())
        for field in [
            "config",
            "dataset_sha256",
            "adapter_sha256",
            "evaluation_protocol",
        ]:
            if previous.get(field) != metadata.get(field):
                raise ValueError("resume protocol changed: " + field)
        records = previous["records"]
        prior_peak = previous.get("peak_vram_bytes_so_far", 0)
        metadata["peak_memory_complete"] = previous.get(
            "peak_memory_complete", "peak_vram_bytes_so_far" in previous
        )
        metadata["resumed_questions"] = len(records)
        if [r["id"] for r in records] != [r["id"] for r in rows[: len(records)]]:
            raise ValueError("resume questions changed")
    run_start = time.monotonic()
    for i, row in enumerate(rows):
        if i < len(records):
            continue
        if time.monotonic() - run_start > a.max_seconds:
            break
        set_seed(cfg.seed + i)
        text = tok.apply_chat_template(
            make_prompt(row["question"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg.thinking,
        )
        batch = tok(text, return_tensors="pt").to(model.device)
        responses = []
        tokens = []
        truncations = []
        outcomes = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.monotonic()
        # Eight samples, generated in fixed microbatches; protocol records batch size.
        for j in range(0, 8, a.generation_batch_size):
            with torch.inference_mode():
                output = model.generate(
                    **batch,
                    max_new_tokens=cfg.max_completion_length,
                    do_sample=True,
                    num_return_sequences=a.generation_batch_size,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=0,
                    pad_token_id=tok.pad_token_id,
                )
            eos = model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos)
            for sequence in output:
                completion = sequence[batch["input_ids"].shape[-1] :].tolist()
                info = completion_info(completion, eos, cfg.max_completion_length)
                text = tok.decode(
                    completion[: info["tokens"]], skip_special_tokens=True
                )
                tokens.append(info["tokens"])
                truncations.append(info["truncated"])
                responses.append(text)
                parsed = parse_final(text)
                outcomes.append(
                    "parse_failure"
                    if parsed is None
                    else ("correct" if correct(text, row["answer"]) else "incorrect")
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "responses": responses,
                "answers": [
                    str(parse_final(x)) if parse_final(x) is not None else None
                    for x in responses
                ],
                "truncated": truncations,
                "outcomes": outcomes,
                "tokens": tokens,
                "seconds": time.monotonic() - start,
                **question_metrics(responses, row["answer"]),
            }
        )
        save_json(
            partial,
            {
                **metadata,
                "adapter": a.adapter,
                "records": records,
                "status": "partial",
                "peak_vram_bytes_so_far": max(
                    prior_peak, torch.cuda.max_memory_allocated()
                )
                if torch.cuda.is_available()
                else None,
            },
        )
        print(f"evaluated {i + 1}/{len(rows)}", flush=True)
    if not records:
        raise RuntimeError("no completed questions")
    metrics = {
        k: sum(r[k] for r in records) / len(records)
        for k in ["accuracy_at_1", "pass_at_8", "majority_at_8", "format_rate"]
    }
    metrics.update(
        n_questions=len(records),
        answer_extraction_success=sum(
            o != "parse_failure" for r in records for o in r["outcomes"]
        )
        / (8 * len(records)),
        invalid_answer_rate=sum(
            o == "parse_failure" for r in records for o in r["outcomes"]
        )
        / (8 * len(records)),
        parsed_incorrect_rate=sum(
            o == "incorrect" for r in records for o in r["outcomes"]
        )
        / (8 * len(records)),
        truncation_rate=sum(sum(r["truncated"]) for r in records) / (8 * len(records)),
        average_generated_tokens=sum(sum(r["tokens"]) for r in records)
        / (8 * len(records)),
        seconds_per_question=sum(r["seconds"] for r in records) / len(records),
        seconds_per_response=sum(r["seconds"] for r in records) / (8 * len(records)),
        peak_vram_bytes=max(prior_peak, torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else None,
    )
    for field in ["accuracy_at_1", "pass_at_8", "majority_at_8"]:
        metrics[field + "_ci95"] = wilson(sum(r[field] for r in records), len(records))
    save_json(
        a.output,
        {
            **metadata,
            "adapter": a.adapter,
            "status": "complete" if len(records) == len(rows) else "runtime_limited",
            "requested_questions": len(rows),
            "split": a.split,
            "metrics": metrics,
            "records": records,
        },
    )
    with (
        Path(a.output).with_suffix(".csv").open("w", newline="", encoding="utf-8") as f
    ):
        fields = [
            "id",
            "accuracy_at_1",
            "pass_at_8",
            "majority_at_8",
            "format_rate",
            "seconds",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
