# Phase 2 interim measurements

Snapshot during the 30 October 2026 run. The base-initialized GRPO training/evaluation is pending. This is not the final Phase 2 report.

All six completed conditions below use the same 100 validation questions, eight samples, seed rule and 256-token output cap. The reserved final test has not been evaluated.

| Model | Accuracy@1 | Pass@8 | Majority@8 |
|---|---:|---:|---:|
| Unmodified model | 79% | 96% | 91% |
| Original SFT 20 | 52% | 63% | 53% |
| Original SFT + GRPO 8 | 50% | 64% | 52% |
| SFT 50 | 61% | 72% | 63% |
| SFT 100 | 71% | 79% | 70% |
| SFT 100 + GRPO 16 | 70% | 79% | 69% |

The selected SFT budget is 100 steps. Its Accuracy@1 difference versus original SFT20 is +19 percentage points, paired 95% interval [+9,+29]. Versus the unmodified model the difference is -8 points, interval [-17,+1]. Selection used validation, so this is not a final-test generalization claim.

GRPO from the selected SFT adapter changed weights but gave -1 point Accuracy@1, paired interval [-3,0]. Four of sixteen training groups had mixed correctness; mean reward was 0.625. No RL gain is established by these measurements.

SFT100 evaluation resumed after 89 saved questions. Its recorded inference peak is only the resumed final segment; earlier peak memory was not retained. Stored per-question responses, seeds and timings were preserved.

The CPU suite passed 83 tests before the final resume-metadata changes. Final verification will be repeated after completion. Original datasets, adapters and reports are protected by a 96-file hash manifest.

See [diagnosis](phase2-diagnosis.md) and [GRPO analysis](grpo-reward-analysis.md). The final report will include the seventh condition, all charts, integrity checks and the full comparison.
