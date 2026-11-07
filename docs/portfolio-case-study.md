# MiniDeepSeek portfolio case study

## The problem I investigated

I wanted to test whether verifiable rewards could improve a small language model using hardware I could run locally. MiniDeepSeek trains Qwen3-0.6B with LoRA SFT and eight-response GRPO on a native Windows laptop with a 6 GB RTX 4050. The difficult part was separating formatting changes, sampling effects and model selection from evidence of better answers.

## What was built

The project has deterministic synthetic data generation, split audits, a bounded rational-answer checker, actual SFT/GRPO training, reward-variance diagnostics and a local Gradio comparison dashboard. Seven conditions produced 5,600 validation completions. The two selected finalists were then frozen by checkpoint, source, data and protocol hashes before a single held-out evaluation on 234 unique questions, eight candidates/model (3744 held-out completions).

## Measured outcomes

Validation favored the unmodified-model GRPO initialization: 83% Accuracy@1 versus 79% for the unmodified model, exact paired p=0.125. SFT100 improved over SFT20 but remained below the unmodified model. Those results guided selection and weren't treated as independent confirmation.

The frozen held-out result was 85.90% for Base and 83.76% for Base+GRPO. GRPO minus Base: **-2.14 percentage points**; 2 improved, 7 worsened, 225 unchanged first-answer outcomes. Two-sided exact paired McNemar p=0.179688. Paired empirical bootstrap 95% interval: [-4.70, +0.00] percentage points. The exact paired test does not provide conventional 5% significance evidence of improved first-answer accuracy. This does not establish equivalence.

| Frozen finalist | Accuracy@1 | Wilson 95% CI | Pass@8 | Majority@8 |
|---|---:|---:|---:|---:|
| Base | 85.90% | 80.86%–89.78% | 99.15% | 96.58% |
| Base + GRPO | 83.76% | 78.50%–87.93% | 98.72% | 96.58% |

## Engineering decisions and tradeoffs

LoRA kept local training feasible. A binary exact-value reward avoided a learned judge but limited evaluation to supported rational answers. Reward-group logs exposed zero-variance updates; final evaluation separated wrong extracted values from parse failures and truncation. CPU-only CI avoids accidental GPU training. Models, raw completions and private datasets stay out of GitHub.

The final dataset has 234 unique questions from 240 rows and only single-operation arithmetic. Shared templates, one seed and finite output lengths limit what the result says about broader reasoning. I would use a new independently held-out multi-step benchmark and multiple seeds before making a broader claim. See the [full research report](final-research-report.md) and [limitations](limitations.md).

## Three résumé bullets

- Built a reproducible Qwen3-0.6B fine-tuning pipeline with PEFT LoRA, supervised training and eight-response GRPO on a 6 GB RTX 4050 laptop GPU, with verified adapter updates and reward-variance diagnostics.
- Compared seven configurations using 5,600 validation completions, then froze two finalists for a 234-question held-out evaluation; reported Accuracy@1, Pass@8, Majority@8, paired confidence intervals and exact McNemar statistics.
- Implemented safe rational-answer rewards, dataset contamination checks, EOS-aware truncation metrics and a Gradio evidence dashboard; documented the observed held-out accuracy difference of -2.14 points with p=0.179688, without overstating generalization.

## LinkedIn project announcement draft — not posted

I completed MiniDeepSeek, a local experiment in fine-tuning Qwen3-0.6B with supervised learning and GRPO on a 6 GB laptop GPU.

I compared seven configurations on 100 validation questions, then froze the two finalists before evaluating 234 unique held-out arithmetic questions with eight samples per model. Final Accuracy@1: 85.90% for the unmodified model and 83.76% after GRPO. The exact paired comparison gave p=0.179688. The exact paired test does not provide conventional 5% significance evidence of improved first-answer accuracy. This does not establish equivalence.

The project includes a safe numeric reward checker, reward-variance diagnostics, reproducible manifests, GPU measurements and a local dashboard. It is a small-model research experiment, not a reproduction of the full DeepSeek-R1 system. The report discusses failed SFT conditions and uncertainty alongside the measured outcomes.

Repository: https://github.com/Shahidali78/minideepseek

## Short interview explanation

“I tested whether a small model could improve on arithmetic using binary verifiable rewards on a laptop. I implemented LoRA SFT and GRPO with eight sampled answers per question, audited the data and measured whether updates actually changed the adapter. The validation study selected GRPO initialized from the unmodified model. I then froze everything and ran the held-out comparison once. The first-answer accuracies were 85.90% and 83.76%, with exact paired p=0.179688. I distinguish that narrow result from broad reasoning ability, and I keep uncertainty, truncation and extraction failures visible.”

## Repository description and topics

Description: Local Qwen3-0.6B LoRA SFT/GRPO research on a 6 GB GPU, with frozen paired evaluation, verifiable rewards and an evidence dashboard.

Five topics: `reinforcement-learning`, `grpo`, `llm-finetuning`, `lora`, `mathematical-reasoning`.
