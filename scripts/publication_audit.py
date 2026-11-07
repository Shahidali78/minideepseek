"""Review the exact allowlisted public file set without printing secret values."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT_FILES = {
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    ".gitignore",
    "pyproject.toml",
    "requirements.txt",
    "requirements-training.txt",
    "requirements-lock.txt",
}
PUBLIC_DIRS = ("src", "scripts", "tests", "configs", "docs", ".github")
EXTENSIONS = {".py", ".md", ".json", ".png", ".svg", ".yml", ".yaml", ".sha256", ".txt"}
FORBIDDEN = {".jsonl", ".safetensors", ".bin", ".pt", ".pth", ".pem", ".key", ".log"}
SECRET_PATTERNS = [
    re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}"),
]


def candidates():
    paths = [Path(p) for p in ROOT_FILES if Path(p).is_file()]
    for folder in PUBLIC_DIRS:
        paths.extend(
            p
            for p in Path(folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in EXTENSIONS
        )
    return sorted(paths, key=lambda p: p.as_posix())


def inspect(paths):
    issues, reviewed = [], []
    for path in paths:
        data = path.read_bytes()
        if (
            path.is_symlink()
            or path.suffix in FORBIDDEN
            or path.stat().st_size > 5_000_000
        ):
            issues.append(
                {"path": str(path), "reason": "symlink, forbidden artifact or >5MB"}
            )
        if path.suffix not in (".png",):
            text = data.decode("utf-8-sig")
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                issues.append(
                    {
                        "path": str(path),
                        "reason": "credential-shaped content; value withheld",
                    }
                )
            if path.suffix == ".json":
                obj = json.loads(text)

                def private_fields(value):
                    if isinstance(value, dict):
                        return any(
                            k in {"responses", "records", "completions"}
                            or private_fields(v)
                            for k, v in value.items()
                        )
                    return isinstance(value, list) and any(
                        private_fields(v) for v in value
                    )

                if private_fields(obj):
                    issues.append(
                        {
                            "path": str(path),
                            "reason": "raw evaluation fields in public JSON",
                        }
                    )
        reviewed.append(
            {
                "path": path.as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return reviewed, issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/publication-audit.json")
    parser.add_argument("--git", default="git")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    paths = candidates()
    if args.staged:
        result = subprocess.run(
            [args.git, "diff", "--cached", "--name-only", "--diff-filter=ACM", "-z"],
            capture_output=True,
            check=True,
        )
        staged = [Path(p) for p in result.stdout.decode().split("\0") if p]
        allowed = {p.as_posix() for p in paths}
        if any(p.as_posix() not in allowed for p in staged):
            raise SystemExit(
                "staging includes a file outside the reviewed publication allowlist"
            )
        for path in staged:
            index = subprocess.run(
                [args.git, "show", ":" + path.as_posix()],
                capture_output=True,
                check=True,
            )
            if index.stdout != path.read_bytes():
                raise SystemExit(
                    "staged content differs from reviewed working file: "
                    + path.as_posix()
                )
        paths = staged
    reviewed, issues = inspect(paths)
    history_commits = None
    if args.history:
        commits = subprocess.run(
            [args.git, "rev-list", "--all"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
        history_commits = len(commits)
        pattern = r"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|(AKIA|ASIA)[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{24,}|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY)"
        for commit in commits:
            result = subprocess.run(
                [args.git, "grep", "--quiet", "-I", "-E", pattern, commit],
                capture_output=True,
            )
            if result.returncode == 0:
                issues.append(
                    {
                        "commit": commit,
                        "reason": "credential-shaped content in Git history; value withheld",
                    }
                )
            elif result.returncode != 1:
                raise RuntimeError("Git history scan failed")
        messages = subprocess.run(
            [args.git, "log", "--all", "--format=%B"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if any(pattern.search(messages) for pattern in SECRET_PATTERNS):
            issues.append(
                {
                    "reason": "credential-shaped content in commit messages; value withheld"
                }
            )
    output = {
        "files": reviewed,
        "file_count": len(reviewed),
        "total_bytes": sum(r["bytes"] for r in reviewed),
        "issues": issues,
        "history_commits_scanned": history_commits,
        "scope": "explicit source/docs/config/test allowlist; no model files, raw data or outputs",
        "secret_scan_limit": "Known credential shapes only; not a guarantee against all secret formats",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k != "files"}, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
