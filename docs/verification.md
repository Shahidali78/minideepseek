# Recorded verification

The frozen GPU evaluations completed on 30 October 2026. Local publication verification completed on 7 November 2026 at 17:33 UTC. These are executed checks, not a checklist of intended work.

| Check | Actual outcome |
|---|---|
| CPU pytest | 101 passed; CUDA hidden with CUDA_VISIBLE_DEVICES=-1 |
| Python compilation | Passed for src, scripts and tests |
| Dependency consistency | pip check passed |
| Dashboard build | Passed; seven model choices |
| Real dashboard screenshot | Rendered in a separate headless Chrome profile and visually inspected |
| Dataset reconstruction | Expanded-v2 train/validation/test reproduced byte-for-byte |
| Historical preservation | All 96 protected files unchanged |
| Frozen identities | Model, adapter, datasets, validation outputs and source snapshot hashes verified |
| Final-score integrity | All 3744 saved completions re-scored; saved answers/outcomes/metrics matched |
| Model loading | Both frozen finalists actually loaded and generated on the local GPU |
| Publication allowlist scan | Passed; no detected credential shapes or raw evaluation JSON fields |
| GitHub authentication | Confirmed Shahidali78 outside the network sandbox |
| Remote CI | Passed: 96 tests passed, 5 optional-dependency tests skipped on the minimal offline runner |

Private logs and raw outputs remain in ignored `outputs/phase3-final-20261030`. [Aggregate verification evidence](evidence/phase3-final-20261030/verification.json) records the local results. The credential scan covers known formats and cannot prove the absence of every possible secret. Initial Git history was empty; only the explicitly reviewed source/documentation allowlist is eligible for the first commit.

The final evaluation itself is the real model-loading smoke test; no extra GPU generation or finalist retraining was needed. CI uses offline lightweight tests, with optional dependency tests skipped when those packages aren't installed.

## Public repository verification

Published [Shahidali78/minideepseek](https://github.com/Shahidali78/minideepseek) as a public source/documentation repository. The initial reviewed source commit is `ff2c6391c57fac7e34d19bb21855b6fa06795cdd`. All 110 remote files matched the local Git blob identities at that commit. The publication excluded datasets, model checkpoints, adapters and raw private evaluation outputs. The initial one-commit Git history scan found no known credential patterns; subsequent publication-status changes receive the same scan.

The [first GitHub Actions run](https://github.com/Shahidali78/minideepseek/actions/runs/42397034508) completed successfully on 7 November 2026: **96 passed, 5 skipped**. The five skips are optional-dependency tests on the lightweight CI environment; the full local environment passed all 101 tests. CI ran Python compilation and pytest without model downloads or training. The [workflow page](https://github.com/Shahidali78/minideepseek/actions/workflows/ci.yml) shows checks for later documentation commits.

[Publication status evidence](publication-status.json) records the initial verified commit and run. No force-push, paid resource creation, checkpoint publication or LinkedIn posting occurred.
