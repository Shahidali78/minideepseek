"""Download only the single requested 0.6B model; record immutable revision."""

import json
from pathlib import Path

import requests

repo = "Qwen/Qwen3-0.6B"
s = requests.Session()
r = s.get(f"https://huggingface.co/api/models/{repo}", timeout=60)
r.raise_for_status()
info = r.json()
revision = info["sha"]
out = Path("outputs/model")
out.mkdir(parents=True, exist_ok=True)
allowed = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
}
for item in info["siblings"]:
    name = item["rfilename"]
    if name not in allowed:
        continue
    path = out / name
    if path.exists():
        continue
    print("Downloading", name, flush=True)
    r = s.get(
        f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
        stream=True,
        timeout=120,
    )
    r.raise_for_status()
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)
    temp.rename(path)
(out / "source.json").write_text(
    json.dumps({"model": repo, "revision": revision}, indent=2)
)
print("Complete", revision, flush=True)
