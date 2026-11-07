from fractions import Fraction

import pytest

from minideepseek.core import correct, numeric, parse_final, reward_fn
from minideepseek.diagnostics import (
    completion_info,
    group_diagnostics,
    select_validation,
    summarize_groups,
)


@pytest.mark.parametrize(
    "text,value",
    [
        ("36.0", 36),
        ("72/2", 36),
        ("(70+2)/2", 36),
        ("25%", Fraction(1, 4)),
        (r"25\%", Fraction(1, 4)),
        ("-12.5%", Fraction(-1, 8)),
        ("$36$", 36),
        (r"\(36\)", 36),
        ("0.1+0.2", Fraction(3, 10)),
    ],
)
def test_equivalences(text, value):
    assert numeric(text) == value
    assert correct("<answer>" + text + "</answer>", str(value))


@pytest.mark.parametrize(
    "text",
    [
        "2**10000",
        "1//2",
        '__import__("os")',
        "[36]",
        "36 or 37",
        "True",
        "sum([36])",
        "9e9",
        "9% + 1",
        "1/0",
        "1?2",
        "(" * 25 + "1" + ")" * 25,
    ],
)
def test_unsafe_expressions(text):
    with pytest.raises((ValueError, ZeroDivisionError)):
        numeric(text)
    assert parse_final("<answer>" + text + "</answer>") is None


def test_group_statistics():
    g = group_diagnostics(
        [r"\boxed{2}"] * 4 + [r"\boxed{3}"] * 3 + ["bad"],
        ["2"] * 8,
        [10] * 8,
        [False] * 7 + [True],
    )
    assert (
        g["correct_count"] == 4
        and g["invalid_count"] == 1
        and g["truncated_count"] == 1
    )
    assert not g["zero_variance"] and abs(g["advantage_mean"]) < 1e-8
    assert summarize_groups([g])["mixed_correctness_fraction"] == 1
    z = group_diagnostics([r"\boxed{2}"] * 8, ["2"] * 8)
    assert z["advantages"] == [0] * 8 and z["zero_variance"]
    with pytest.raises(ValueError):
        group_diagnostics(["bad"] * 7, ["2"] * 8)
    with pytest.raises(ValueError):
        group_diagnostics(["bad"] * 8, ["1", "2"] * 4)


def test_exact_truncation():
    assert completion_info([1, 2, 9, 9], {9}, 4) == {
        "tokens": 3,
        "ended_with_eos": True,
        "truncated": False,
    }
    assert completion_info([1, 2, 3, 4], {9}, 4)["truncated"]
    assert not completion_info([1, 2, 3, 9], {9}, 4)["truncated"]


def test_question_selection():
    rows = [
        {"id": "a", "question": "a"},
        {"id": "b", "question": " A "},
        {"id": "c", "question": "b"},
    ]
    assert select_validation(rows, 2) == [rows[0], rows[2]]
    assert select_validation(rows, 2) == select_validation(rows, 2)


def test_no_speculative_reward():
    assert reward_fn([r"\boxed{36} or \boxed{37}"], ["36"]) == [0]
    assert reward_fn(["Question mentions 36."], ["36"]) == [0]
    assert reward_fn(["<answer>72/2</answer>"], ["36"]) == [1]


def test_dashboard_logic(tmp_path):
    from minideepseek.dashboard import adapter_choices, available, score_candidates

    assert available(None) and not available(tmp_path / "missing")
    import json

    p = tmp_path / "selection.json"
    p.write_text(json.dumps({"selected_sft": "my-checkpoint"}))
    assert adapter_choices(p)["Selected SFT (Phase 2)"] == "my-checkpoint"
    assert score_candidates(["<answer>72/2</answer>"], "36")[0]["correct"]
    assert score_candidates(["<answer>36</answer>"])[0]["correct"] is None
    assert score_candidates(["36"], "36")[0]["final_answer"] is None
    with pytest.raises(ValueError):
        score_candidates([], "malformed")


def test_comparison_protocol():
    from minideepseek.comparison import validate_comparison

    r = {
        "config": {},
        "dataset_sha256": "x",
        "resolved_model_revision": "a",
        "split": "val",
        "records": [],
        "evaluation_protocol": {"generation_batch_size": 8},
    }
    with pytest.raises(ValueError):
        validate_comparison(
            [r, {**r, "evaluation_protocol": {"generation_batch_size": 1}}]
        )


def test_sft_response_mask_without_download(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from trl import SFTConfig, SFTTrainer

    backend = Tokenizer(
        models.WordLevel(
            {"<pad>": 0, "<eos>": 1, "<unk>": 2, "question": 3, "answer": 4},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=5,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=SFTConfig(
            output_dir=str(tmp_path),
            max_length=32,
            use_cpu=True,
            bf16=False,
            fp16=False,
            report_to="none",
            completion_only_loss=True,
        ),
        train_dataset=Dataset.from_list(
            [{"prompt": "question ", "completion": "answer"}]
        ),
    )
    batch = trainer.data_collator([trainer.train_dataset[0]])
    assert batch["labels"][0, 0].item() == -100
    assert (batch["labels"][0, 1:] != -100).all()


def test_cpu_checkpoint_resume(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments

    config = Qwen3Config(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    data = [{"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]}] * 4

    def make(steps):
        return Trainer(
            model=Qwen3ForCausalLM(config),
            train_dataset=data,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                max_steps=steps,
                per_device_train_batch_size=1,
                use_cpu=True,
                bf16=False,
                fp16=False,
                save_steps=1,
                report_to="none",
                optim="adamw_torch",
            ),
        )

    first = make(1)
    first.train()
    assert (tmp_path / "checkpoint-1/optimizer.pt").exists()
    second = make(2)
    second.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1"))
    assert (
        second.state.global_step == 2
        and (tmp_path / "checkpoint-2/trainer_state.json").exists()
    )


def test_gold_column_never_changes_prompt():
    from minideepseek.data import grpo_records

    a = grpo_records([{"question": "Calculate 18 + 18.", "answer": "36"}])[0]
    b = grpo_records([{"question": "Calculate 18 + 18.", "answer": "999999"}])[0]
    assert a["prompt"] == b["prompt"]
    assert "999999" not in str(b["prompt"])


def test_accuracy_interval():
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_metrics", "scripts/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lo, hi = module.wilson(0, 100)
    assert abs(lo) < 1e-12 and 0.03 < hi < 0.04
    lo, hi = module.wilson(100, 100)
    assert 0.96 < lo < 0.97 and abs(hi - 1) < 1e-12


def test_malformed_completion_wrappers():
    outputs = [
        [],
        [None],
        ["bad"],
        [{"role": "user", "content": "<answer>36</answer>"}],
        None,
        {},
        [{"role": "assistant", "content": None}],
    ]
    assert reward_fn(outputs, ["36"] * len(outputs)) == [0.0] * len(outputs)


@pytest.mark.parametrize(
    "args", [["--max-seconds", "0"], ["--from-base", "--sft-adapter", "missing"]]
)
def test_invalid_training_cli_stops_before_gpu(monkeypatch, args):
    import sys

    from minideepseek.training import run

    monkeypatch.setattr(sys, "argv", ["train_grpo", *args])
    with pytest.raises(SystemExit) as error:
        run("grpo")
    assert error.value.code == 2


def test_exact_paired_small_discordance():
    from minideepseek.comparison import paired_interval

    a = [{"x": False}] * 4 + [{"x": True}] * 96
    b = [{"x": True}] * 100
    r = paired_interval(a, b, "x")
    assert r["corrected_questions"] == 4 and r["regressed_questions"] == 0
    assert r["mcnemar_exact_p_two_sided"] == 0.125
    assert paired_interval(b, b, "x")["mcnemar_exact_p_two_sided"] == 1
