# GRPO reward analysis

The original eight-step GRPO run produced 64 responses. Six of eight groups had identical rewards (75% zero variance); two groups contained both correct and incorrect answers. Mean reward was 0.515625, and mean correct responses per group was 4.125. Five groups contained at least one correct answer (62.5%). Only the two mixed groups supplied a correctness policy gradient. Three all-correct groups still had zero advantages.

The preceding two-step memory pilot had two all-zero groups and no parameter change. The eight-step run had nonzero gradient norms in its mixed groups, 112 changed adapter tensors relative to SFT and three changed validation responses. Near-zero scalar loss does not prove a stalled optimizer: at the on-policy ratio of one, centered advantages can cancel the loss value without canceling its gradient.

Phase 2 diagnostics store eight extracted answers, binary rewards, sample reward standard deviation, identical-reward status, correctness/invalid counts, EOS-aware truncation counts, completion lengths and recomputed group-normalized advantages. The epsilon is 1e-4, matching the installed TRL group normalization. These diagnostics are saved after every generation group, including before a later failure. Default logs exclude prompts and generated response bodies; `--log-responses` explicitly enables bodies for a controlled debugging run.

The reward remains binary correctness: there is no length bonus or formatting bonus. A bounded AST whitelist handles rational arithmetic and percentages without `eval`, code execution, function calls, powers or arbitrary names. The final answer must be explicitly boxed or enclosed in one answer tag. Multiple speculative answer fields are rejected. Gold answers are passed to the reward function in a separate dataset column, never inserted into the generation prompt. Validity of the reasoning trace is not established by a correct final answer.

The predeclared Phase 2 rule is a four-step GRPO pilot from the selected SFT adapter and another from the unmodified model. If fewer than 25% of groups have mixed correctness, a separate pilot changes only training temperature to 0.9. A 16-step run is allowed only when that signal criterion is met. Evaluation remains at temperature 0.6 for every model. KL remains disabled to keep the initialization comparison small and avoid confounding SFT initialization with a penalty against the underlying base weights. The installed PEFT/TRL reference with adapters disabled would be the underlying model, not a frozen copy of the SFT adapter.

Phase 2 group summaries and charts will be added to the experiment report after the actual runs complete. No future reward improvement is assumed. Reward changes across different training questions are not a held-out learning curve.

The normalized objective and zero-variance behavior follow the [versioned TRL GRPO documentation](https://huggingface.co/docs/trl/v0.23.1/grpo_trainer).

## Phase 2 validation candidate support

Under the matched 100-question, eight-sample, 256-token evaluation, the unmodified model had 45% mixed-correctness groups and 55% zero-variance groups. The original SFT adapter had 20% mixed groups and 80% zero variance; original GRPO had 21% mixed and 79% zero variance. At least one correct candidate appeared in 96%, 63% and 64% of groups respectively. Average correct candidates were 6.62, 4.23 and 4.17 out of eight.

These are observed **validation** candidate statistics, not new training rewards. They support the hypothesis that answer-only SFT reduced candidate correctness support and the frequency of useful group-relative signals. The new GRPO training diagnostics are needed to test whether this also occurs on sampled training questions. The base-initialized GRPO comparison is motivated by these measurements.
