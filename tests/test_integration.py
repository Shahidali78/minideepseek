"""No network: tiny random model checks PEFT serialization and inference."""

import pytest


def test_tiny_adapter_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    base = Qwen3ForCausalLM(config)
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=0.01
    )
    model(tokens, labels=tokens).loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        expected = model(tokens).logits.clone()
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    torch.manual_seed(42)
    reloaded = PeftModel.from_pretrained(Qwen3ForCausalLM(config), tmp_path).eval()
    with torch.no_grad():
        actual = reloaded(tokens).logits
    assert torch.allclose(expected, actual, atol=1e-6)


def test_installed_trl_api():
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    from trl import GRPOConfig, SFTConfig

    a = SFTConfig(
        output_dir="outputs/test-api",
        max_length=256,
        completion_only_loss=True,
        bf16=False,
        fp16=False,
        report_to="none",
    )
    b = GRPOConfig(
        output_dir="outputs/test-api",
        num_generations=8,
        generation_batch_size=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_completion_length=96,
        use_vllm=False,
        bf16=False,
        fp16=False,
        report_to="none",
    )
    assert a.max_length == 256 and b.num_generations == 8
