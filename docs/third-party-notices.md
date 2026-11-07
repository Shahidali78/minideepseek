# Data and model notices

Project source and original documentation are offered under the repository MIT license. That license does not relicense third-party models, libraries or datasets.

Qwen/Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca` is published under Apache2.0. The [revision-specific model license](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE) and [model card](https://huggingface.co/Qwen/Qwen3-0.6B) are authoritative. Weights, tokenizer assets and trained adapters are excluded from this repository. Any later distribution must review and preserve the relevant license and notices. No separate authorization to publish checkpoints is assumed.

Reported data is generated synthetic arithmetic, preserved locally from the starter and reproduced by the supplied deterministic generator. The generator contains no copied benchmark question collection. Dataset files are not redistributed here; their hashes, counts, provenance and overlap audits are recorded in manifests. Shared mathematical templates don't establish independent task families.

`create_data.py --gsm8k` is an optional, unused path. No reported result in this study uses GSM8K. Anyone enabling that path must review the dataset's then-current license and terms before use or redistribution; no claim of a license audit for that unused external dataset is made here.

The source depends on separately licensed packages including PyTorch, Transformers, TRL, PEFT, Gradio and Matplotlib. Package versions are recorded in the requirements files and frozen environment manifest. Installing a dependency does not transfer its licensing obligations to the MIT license for this project's source.
