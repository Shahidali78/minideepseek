import re
from collections import Counter
from fractions import Fraction

MODEL = "Qwen/Qwen3-0.6B"
INSTRUCTION = (
    "Solve this math problem. End with exactly one final answer "
    "in the form: \\boxed{number}."
)


PARSER_VERSION = "rational-expression-v2"


def numeric(value):
    """Bounded rational arithmetic via an AST whitelist; never execute code."""
    import ast

    raw = str(value).strip().replace("\u2212", "-")
    if len(raw) > 120:
        raise ValueError("answer too long")
    if raw.startswith("$") and raw.endswith("$"):
        raw = raw[1:-1].strip()
    if raw.startswith(r"\(") and raw.endswith(r"\)"):
        raw = raw[2:-2].strip()
    raw = raw.replace(r"\%", "%")
    latex = re.fullmatch(
        r"([+-]?)\\(?:frac|dfrac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}", raw
    )
    if latex:
        sign, a, b = latex.groups()
        return (-1 if sign == "-" else 1) * Fraction(int(a), int(b))
    if "," in raw:
        if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?", raw):
            raise ValueError("invalid thousands separator")
        raw = raw.replace(",", "")
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1].strip()
    if not re.fullmatch(r"[\d.+*/()\s-]+", raw):
        raise ValueError("unsupported numeric expression")
    depth = 0
    for char in raw:
        depth += (char == "(") - (char == ")")
        if depth > 8:
            raise ValueError("expression too deep")
    try:
        tree = ast.parse(raw, mode="eval")
    except (SyntaxError, RecursionError) as error:
        raise ValueError("invalid expression") from error
    if len(list(ast.walk(tree))) > 32:
        raise ValueError("expression too complex")

    def visit(node, depth=0):
        if depth > 8:
            raise ValueError("expression too deep")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(raw, node)
            if not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", token):
                raise ValueError("invalid literal")
            result = Fraction(token)
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            result = visit(node.operand, depth + 1) * (
                1 if isinstance(node.op, ast.UAdd) else -1
            )
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            a, b = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                result = a + b
            elif isinstance(node.op, ast.Sub):
                result = a - b
            elif isinstance(node.op, ast.Mult):
                result = a * b
            else:
                result = a / b
        else:
            raise ValueError("unsupported expression")
        if max(result.numerator.bit_length(), result.denominator.bit_length()) > 512:
            raise ValueError("number too large")
        return result

    return visit(tree.body) / (100 if percent else 1)


def parse_final(text: str):
    if not isinstance(text, str):
        return None
    boxes = re.findall(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    tags = re.findall(r"<answer>\s*([^<>]*)\s*</answer>", text)
    # Reject multiple answers, mixed formats and dangling answer markers.
    if len(boxes) + len(tags) != 1:
        return None
    if text.count("\\boxed") != len(boxes) or text.count("<answer>") != len(tags):
        return None
    if text.count("</answer>") != len(tags):
        return None
    try:
        return numeric((boxes + tags)[0])
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def correct(text, answer):
    try:
        parsed = parse_final(text)
        return parsed is not None and parsed == numeric(answer)
    except (ValueError, ZeroDivisionError, TypeError):
        return False


def make_prompt(question):
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": question},
    ]


def reward_fn(completions, answer, **kwargs):
    if len(completions) != len(answer):
        raise ValueError("completion/reference lengths differ")
    out = []
    for completion, gold in zip(completions, answer):
        if isinstance(completion, list):
            completion = (
                completion[0].get("content", "")
                if len(completion) == 1
                and isinstance(completion[0], dict)
                and completion[0].get("role", "assistant") == "assistant"
                else ""
            )
        out.append(float(correct(completion, gold)))
    return out


def question_metrics(responses, answer):
    if len(responses) != 8:
        raise ValueError("Accuracy@1/Pass@8/Majority@8 require eight predictions")
    parsed = [parse_final(x) for x in responses]
    counts = Counter(x for x in parsed if x is not None)
    # Ties count as incorrect instead of choosing a favorable answer.
    winners = (
        [x for x, n in counts.items() if n == max(counts.values())] if counts else []
    )
    return {
        "accuracy_at_1": correct(responses[0], answer),
        "pass_at_8": any(correct(x, answer) for x in responses),
        "majority_at_8": len(winners) == 1 and winners[0] == numeric(answer),
        "format_rate": sum(x is not None for x in parsed) / len(parsed),
    }
