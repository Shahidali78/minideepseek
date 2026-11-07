# Phase 2 portfolio summary

Studied final-answer SFT, output censoring and sparse group-relative rewards in an already-post-trained small model. The study kept original artifacts intact, separated validation tuning from the reserved test, and compared candidate correctness support rather than treating lower loss as better reasoning.

The highest observed validation Accuracy@1 was 83.0%; see the paired intervals and selection caveats in the [research report](phase2-experiment-report.md). This does not establish general mathematical reasoning improvement.

Suggested repository description

Reproducible Qwen3-0.6B experiments on a 6 GB GPU: controlled math validation, LoRA SFT budgets, eight-candidate GRPO, safe numeric rewards and reward-variance diagnostics.

Three resume-ready achievements

- Ran matched 100-question, eight-sample evaluations across seven Qwen3-0.6B configurations and controlled SFT budget comparisons on a native Windows RTX 4050 6 GB.
- Implemented safe rational-expression and percentage rewards, EOS-aware truncation diagnostics, group-relative advantage statistics and paired uncertainty estimates without using the reserved final test.
- Built versioned experiment automation, resumable evaluation and checkpoints, artifact-integrity verification, offline tests and a Gradio dashboard for candidate-level model comparison.

Research-paper direction

A controlled study of format collapse, output censoring and reward-gradient starvation in small-model RL. Compare verified explanatory targets with answer-only SFT, multiple completion budgets and equal-compute inference voting; repeat across seeds and an independently reserved benchmark.
