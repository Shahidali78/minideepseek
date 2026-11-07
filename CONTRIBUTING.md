# Contributing

Keep measured evidence separate from proposals. Include the command, configuration, seed, data hash and actual check outcomes with a change that affects experiments. Use a new output directory; don't overwrite historical datasets, adapters, manifests or reports.

Run `python -m pytest -q` and `python -m compileall -q src scripts tests` before proposing code changes. Tests should use synthetic fixtures and CPU stubs or tiny random models. Don't download checkpoints or launch training in CI.

Answer-checker changes need equivalence and rejection tests. Never execute model-provided expressions. Changes to sampling, prompts, metric definitions or data selection require a new protocol identity. Don't present scores obtained under different protocols as isolated training gains.

Do not commit `.env` files, credentials, raw evaluation completions, caches, datasets or model/checkpoint files. Public evidence should contain aggregates and hashes only. Keep examples illustrative rather than exposing held-out question/answer pairs.

The existing final set has been used for evaluation. Don't tune a model on it and then call another run independent confirmation. New confirmatory work needs a new held-out set and a protocol frozen before outcomes. Contributions must respect the separate model and dependency licenses.
