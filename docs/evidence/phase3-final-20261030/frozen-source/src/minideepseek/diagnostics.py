"""EOS-aware completion diagnostics and deterministic evaluation selection."""

from .core import correct, numeric, parse_final
from .data import key


def select_validation(rows, count):
    if count < 1:
        raise ValueError("positive question count required")
    unique = {}
    for row in rows:
        unique.setdefault(key(row["question"]), row)
    return list(unique.values())[:count]


def completion_info(ids, eos_ids, cap):
    for i, token in enumerate(ids):
        if token in eos_ids:
            return {"tokens": i + 1, "truncated": False, "ended_with_eos": True}
    return {"tokens": len(ids), "truncated": len(ids) >= cap, "ended_with_eos": False}


def group_diagnostics(responses, references, lengths=None, truncated=None):
    if len(responses) != 8 or len(references) != 8:
        raise ValueError("group requires eight samples")
    if len({numeric(a) for a in references}) != 1:
        raise ValueError("mixed references")
    parsed = [parse_final(r) for r in responses]
    rewards = [float(correct(r, g)) for r, g in zip(responses, references)]
    mean = sum(rewards) / 8
    std = (sum((r - mean) ** 2 for r in rewards) / 7) ** 0.5
    advantages = [(r - mean) / (std + 1e-4) for r in rewards]
    return {
        "extracted_answers": [str(p) if p is not None else None for p in parsed],
        "rewards": rewards,
        "mean_reward": mean,
        "reward_std": std,
        "reward_variance": std**2,
        "zero_variance": std == 0,
        "correct_count": int(sum(rewards)),
        "invalid_count": sum(p is None for p in parsed),
        "truncated_count": sum(truncated) if truncated is not None else None,
        "completion_lengths": lengths,
        "advantages": advantages,
        "advantage_mean": sum(advantages) / 8,
        "advantage_std": (sum(a * a for a in advantages) / 7) ** 0.5,
        "distinct_completions": len(set(responses)),
    }


def summarize_groups(groups):
    if not groups:
        return {"n_groups": 0}
    n = len(groups)
    return {
        "n_groups": n,
        "zero_variance_fraction": sum(g["zero_variance"] for g in groups) / n,
        "any_correct_fraction": sum(g["correct_count"] > 0 for g in groups) / n,
        "mixed_correctness_fraction": sum(0 < g["correct_count"] < 8 for g in groups)
        / n,
        "average_correct_count": sum(g["correct_count"] for g in groups) / n,
        "mean_reward": sum(g["mean_reward"] for g in groups) / n,
    }
