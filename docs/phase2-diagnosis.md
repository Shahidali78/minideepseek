# Phase 2 diagnosis

The original nine-question experiment cannot establish an accuracy regression or a general reasoning gain. Its paired SFT-minus-base Accuracy@1 interval was -44.4 to +22.2 percentage points.

## Observed evidence

The base model reached its 96-token cap on 30/72 samples. Several saved first responses compute the right answer (624, 80, 295) but stop before completing the answer box. This is evaluation censoring, rather than evidence that those calculations were wrong. EOS termination was not recorded, so the old cap-hit statistic is only a truncation proxy.

SFT trained on final-answer-only labels. Thinking was disabled consistently in the Qwen chat template for training and evaluation. Mean response length fell from 73.74 to 8.06 tokens, and valid-format rate rose from 58.3% to 98.6%. Its 20-step loss fell from 2.3344 to 0.3247, but held-out accuracy fell from 4/9 to 3/9. Learning the answer format does not demonstrate learning the computation.

The old parser rejects multiple boxes, including repeated identical boxes, and expressions such as `1+2`. This prevents guessing across several answers but also rejects some valid mathematical expressions. An inspected percentage example had two boxes containing the same *incorrect* value 50.88 (reference 49.92); relaxing that extraction would not make it correct. A versioned re-score of all historical raw responses will quantify parser effects without changing historical reports.

GRPO had six zero-variance groups out of eight. Only two groups had mixed correctness and nonzero gradients. The mean training reward was 0.515625. Some groups produced eight identical strings; eight stochastic draws do not guarantee unique strings. Centered advantages can give near-zero scalar loss while gradients remain nonzero. All 112 adapter tensors differ from SFT, with a maximum absolute difference of 1.5452504e-05, and three of 72 validation responses changed. Aggregate scores did not change.

LoRA trains query/value adapter matrices only (1,146,880 parameters); frozen base weights are excluded from the optimizer. BF16, microbatch 1, accumulation 8, AdamW, completion-only masking and gradient checkpointing executed successfully. The saved adapters loaded for actual inference. KL was explicitly disabled.

## Hypotheses to test

Twenty answer-only SFT steps may bias this model toward short guesses at the expense of reasoning. Eight low-learning-rate GRPO updates, only two informative groups and a narrow synthetic distribution may explain the absence of recovery. Neither explanation is established by nine validation examples. More training may help or harm.

Phase 2 holds validation prompts, model revision and decoding fixed across models, increases the completion budget, records EOS-aware truncation separately from parse failures and incorrect answers, and evaluates a deterministic 100-question prefix of the existing validation data if the smoke benchmark permits. Training budgets and selection use validation only. The reserved final test is never evaluated.

The historical re-score is now saved in `outputs/phase2/historical-rescore-v2.json`. The new checker recovered **zero** previously unparseable historical samples. Correct completion counts stayed 38/72 for the base model and 32/72 for each adapter. Parser improvements therefore do not explain the original aggregate regression.

Here "base" means the unmodified requested checkpoint, not a pretraining-only release. The [Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B) lists both pretraining and post-training, and documents distinct thinking and non-thinking modes. This study keeps non-thinking mode fixed, so it does not test a full R1-style long-thinking protocol. The decoding settings are controlled experimental choices, rather than a claim to reproduce every vendor-recommended sampling setting.

An additional observed final-answer faithfulness error appears in the new base evaluation (`val-00005`): the response correctly computes `36/66 = 6/11`, then ends with `\boxed{6}`. The parser correctly extracts 6 and marks it wrong. This is not an extractor implementation failure. It shows that final-answer correctness conflates calculation, instruction following and faithful serialization; a wrong boxed value does not establish that every intermediate reasoning step was wrong.
