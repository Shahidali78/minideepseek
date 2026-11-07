# What these results can support

MiniDeepSeek is a small-model GRPO research experiment, not a reproduction of the full DeepSeek-R1 training system. It uses Qwen3-0.6B, LoRA and binary final-answer rewards on a laptop. The unmodified Qwen checkpoint already received post-training.

Validation results were used to choose budgets and finalists. The observed79% to83% Accuracy@1 change on100 validation questions had exact paired p=0.125. That observation alone is not conclusive evidence of improvement. Only the separately frozen held-out comparison supplies final-test evidence; read its effect size, paired discordances and intervals together.

The reserved set has240 rows but only234 unique questions. The predeclared analysis excludes repeated questions from independent counting. It covers addition, subtraction and multiplication in a restricted operand range, with synthetic templates shared across splits. Exact and normalized question separation prevents direct question reuse; it does not remove template dependence, commutative/semantic similarity or model pretraining familiarity with arithmetic.

There is one training seed and one eight-candidate sample set per question. Paired seeds don't produce identical stochastic trajectories after models diverge. Accuracy@1 uses the first sampled answer, not greedy decoding. Pass@8 benefits from sampling and cannot be read as single-answer reliability. Majority@8 uses the unique modal parsed answer, with ties failing and parse failures abstaining.

A common256-token completion cap can still affect models differently. Truncation and extraction failure are reported independently from parsed numeric errors. A model may compute a correct intermediate number and produce a wrong final field; scoring evaluates the final field. Safe equivalence checking can also accept an unevaluated arithmetic expression. Neither final correctness nor plausible prose proves a hidden reasoning mechanism.

Zero-variance reward groups supply no relative correctness signal. Small LoRA updates and equal update counts don't imply equal compute, especially when response lengths differ. The observed SFT underperformance has several plausible explanations, but the study lacks ablations to identify a cause.

Wilson and paired-bootstrap intervals assume a meaningful sampling population. Template dependence weakens that assumption. Sparse discordances can make bootstrap intervals optimistic; McNemar p-values are not probabilities that a model is superior. A nonsignificant comparison does not prove equivalence, and a significant comparison on this narrow set would not establish broad reasoning.

GPU figures are peak torch-allocated memory, not total VRAM consumption. Latency excludes loading and amortizes batches of eight; it is not interactive request latency. Historical SFT100 evaluation memory was recorded only for the resumed final11 questions. Native Windows success with these short sequences doesn't establish support for larger models or long contexts.

Raw completions and exact trained adapters remain local. Public hashes and aggregate reports support audit, but procedural reconstruction is not an exact checkpoint download. Model license terms apply separately from the source license. The used final set must not become a tuning target for a later confirmation claim.
