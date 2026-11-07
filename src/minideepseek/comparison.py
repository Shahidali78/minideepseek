"""Comparable measured evaluations and paired bootstrap differences."""

import random


def validate_comparison(evals):
    if not evals:
        raise ValueError("no evaluation results")
    first = evals[0]
    for run in evals[1:]:
        if run.get("evaluation_protocol") != first.get("evaluation_protocol"):
            raise ValueError("different evaluation protocols")
        for k in ["config", "dataset_sha256", "resolved_model_revision", "split"]:
            if run[k] != first[k]:
                raise ValueError(f"incomparable evaluations: {k}")
        if [r["id"] for r in run["records"]] != [r["id"] for r in first["records"]]:
            raise ValueError("different evaluation questions")


def paired_interval(base, other, field, seed=42, repetitions=5000):
    deltas = [float(b[field]) - float(a[field]) for a, b in zip(base, other)]
    if not deltas:
        raise ValueError("empty evaluation")
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
        for _ in range(repetitions)
    )
    result = {
        "difference": sum(deltas) / len(deltas),
        "paired_bootstrap_ci95": [
            samples[int(0.025 * repetitions)],
            samples[int(0.975 * repetitions)],
        ],
    }
    if all(a[field] in (0, 1) and b[field] in (0, 1) for a, b in zip(base, other)):
        from math import comb

        improved = sum(d > 0 for d in deltas)
        regressed = sum(d < 0 for d in deltas)
        n = improved + regressed
        p = min(
            1.0,
            2 * sum(comb(n, k) for k in range(min(improved, regressed) + 1)) / (2**n),
        )
        result.update(
            corrected_questions=improved,
            regressed_questions=regressed,
            discordant_questions=n,
            mcnemar_exact_p_two_sided=p,
        )
    return result


def comparison(evals):
    validate_comparison(evals)
    fields = [
        "accuracy_at_1",
        "pass_at_8",
        "majority_at_8",
        "seconds_per_response",
        "peak_vram_bytes",
    ]
    table = []
    for run in evals:
        table.append(
            {
                "model": run["adapter"] or "base",
                **{k: run["metrics"][k] for k in fields},
            }
        )
    differences = []
    for run in evals[1:]:
        differences.append(
            {
                "model": run["adapter"],
                "versus": evals[0]["adapter"] or "base",
                **{
                    k: paired_interval(evals[0]["records"], run["records"], k)
                    for k in fields[:3]
                },
            }
        )
    return {
        "table": table,
        "differences": differences,
        "n_questions": len(evals[0]["records"]),
        "warning": "Bootstrap intervals on a small pilot can be degenerate; do not infer general reasoning gains.",
    }
