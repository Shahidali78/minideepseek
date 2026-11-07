"""Install dependency closure while the separately downloaded CUDA wheel transfers."""

import importlib.metadata as m
import subprocess
import sys

from packaging.requirements import Requirement

for iteration in range(6):
    needed = set()
    for dist in m.distributions():
        for raw in dist.requires or []:
            r = Requirement(raw)
            if r.name.lower() == "torch" or (r.marker and not r.marker.evaluate()):
                continue
            try:
                installed = m.version(r.name)
            except m.PackageNotFoundError:
                installed = None
            if installed is None or installed not in r.specifier:
                needed.add(str(r))
    if not needed:
        break
    print("Installing dependency closure:", sorted(needed), flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-c",
            "requirements.txt",
            *sorted(needed),
        ],
        check=True,
    )
else:
    raise RuntimeError("dependency closure did not converge")
