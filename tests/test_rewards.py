from fractions import Fraction

import pytest

from minideepseek.core import correct, numeric, parse_final, question_metrics, reward_fn


@pytest.mark.parametrize(
    "raw,value",
    [
        ("36", 36),
        ("-1/2", Fraction(-1, 2)),
        (".5", Fraction(1, 2)),
        ("1,234.50", Fraction(2469, 2)),
        ("2/-4", Fraction(-1, 2)),
    ],
)
def test_numeric(raw, value):
    assert numeric(raw) == value
    assert parse_final("work \\boxed{" + raw + "}") == value


@pytest.mark.parametrize(
    "text",
    [
        "36 somewhere",
        "\\boxed{1} and \\boxed{2}",
        "\\boxed{1/0}",
        "\\boxed{nan}",
        "\\boxed{1e3}",
        "\\boxed{1,2}",
        '\\boxed{__import__("os")}',
        "\\boxed{3} <answer>3</answer>",
        "<answer>3</answer><answer>",
        "\\boxed{3} \\boxed{",
        "<answer>3 kg</answer>",
        "<answer></answer>",
        "<answer>3</answer></answer>",
    ],
)
def test_exploits(text):
    assert parse_final(text) is None
    assert not correct(text, "3")


def test_rewards():
    assert reward_fn(["\\boxed{3}", "\\boxed{2}"], ["3", "3"]) == [1.0, 0.0]
    assert reward_fn(
        [[{"role": "assistant", "content": "<answer>1/2</answer>"}]], ["0.5"]
    ) == [1.0]
    assert not correct("\\boxed{3}", "invalid")
    with pytest.raises(ValueError):
        reward_fn(["x"], [])


def test_metrics():
    r = question_metrics(["\\boxed{0.5}"] * 8, "1/2")
    assert r["accuracy_at_1"] and r["pass_at_8"] and r["majority_at_8"]
    assert not question_metrics(["\\boxed{1}"] * 4 + ["\\boxed{2}"] * 4, "1")[
        "majority_at_8"
    ]
    assert not question_metrics(["bad"] * 8, "1")["majority_at_8"]
    with pytest.raises(ValueError):
        question_metrics([], "1")


def test_requires_eight():
    with pytest.raises(ValueError, match="eight predictions"):
        question_metrics(["\\boxed{1}"] * 7, "1")


@pytest.mark.parametrize(
    "text",
    [r"\boxed{\frac{1}{2}}", r"\boxed{-\dfrac{-1}{-2}}", r"<answer>1 / 2</answer>"],
)
def test_latex_fraction(text):
    assert parse_final(text) == (
        Fraction(-1, 2) if "-\\dfrac" in text else Fraction(1, 2)
    )


@pytest.mark.parametrize(
    "text",
    [
        r"\boxed{\frac{1}{0}}",
        r"\boxed{\frac{1}{2}+1}",
        r"\boxed{\frac{1}{2}}\boxed{3}",
        r"\boxed{{1}}",
    ],
)
def test_invalid_latex(text):
    assert parse_final(text) is None
