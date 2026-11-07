import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.comparison import comparison
from minideepseek.runtime import save_json

p = argparse.ArgumentParser()
p.add_argument("evaluations", nargs="+")
p.add_argument("--output", default="outputs/comparison.json")
a = p.parse_args()
result = comparison(
    [json.loads(Path(path).read_text(encoding="utf-8-sig")) for path in a.evaluations]
)
save_json(a.output, result)
with Path(a.output).with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=result["table"][0].keys())
    w.writeheader()
    w.writerows(result["table"])
print(json.dumps(result, indent=2))
