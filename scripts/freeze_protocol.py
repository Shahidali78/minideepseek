import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.protocol import freeze
from minideepseek.runtime import save_json

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/pilot.json")
p.add_argument("--data", default="data")
p.add_argument("--adapters", nargs="+", required=True)
p.add_argument("--output", default="outputs/frozen-protocol.json")
a = p.parse_args()
if Path(a.output).exists():
    raise FileExistsError(a.output)
save_json(a.output, freeze(a.config, a.data, a.adapters))
