# Reproduce the procedure

The recorded run used Windows 11, Python 3.12.8, an RTX 4050 Laptop GPU with 6 GB VRAM, torch 2.8.0+cu128 and the versions in `requirements-lock.txt`. No cloud resources are needed. A GPU with BF16 support was used; CPU tests do not establish CPU training performance.

From the repository directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements-training.txt
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m pip check
```

`requirements-lock.txt` records the measured Windows environment; it is not a portable Linux CUDA lock. Don't replace NVIDIA drivers as part of setup. Check available memory with `nvidia-smi`. Close competing GPU workloads before running inference. Allocated memory in a report excludes driver, display and cache overhead, so it is not a sufficient minimum-device-size estimate.

## Model identity

Use Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`. The supplied `download_model.py` resolves the current remote revision and records it; **for this frozen study use the pinned download command below**, not an unpinned new snapshot:

```powershell
.\.venv\Scripts\python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', local_dir='outputs/model')"
.\.venv\Scripts\python -c "import json; from pathlib import Path; Path('outputs/model/source.json').write_text(json.dumps({'model':'Qwen/Qwen3-0.6B','revision':'c1899de289a04d12100db370d81485cdf75e47ca'}))"
```

Model access/download is separate from this source repository. Its Apache 2.0 license and notices apply; see [third-party notices](third-party-notices.md). The private trained adapter is identified by SHA256 in the final manifest and is not included in the public repository.

## Data reconstruction and validation

```powershell
.\.venv\Scripts\python scripts/create_data.py --output data/expanded-v2
```

The generator refuses to overwrite an existing directory. If `data/test.jsonl` exists, its reserved rows are preserved. If absent in a fresh checkout, the documented original synthetic test protocol is reconstructed using seed3. Training uses seed42, validation seed43, with held-out questions excluded. The final reconstruction audit must compare generated hashes to the published manifest before interpreting a run as a replication. The recorded local reconstruction reproduced all three expanded-v2 files byte-for-byte. Re-running a generator does not create independent evidence.

## Exploratory evaluation and training

```powershell
.\.venv\Scripts\python scripts/evaluate.py --config configs/phase2-eval.json --data data/expanded-v2 --max-examples 100 --output outputs/my-base-val.json
.\.venv\Scripts\python scripts/train_sft.py --help
.\.venv\Scripts\python scripts/train_grpo.py --help
.\.venv\Scripts\python scripts/phase2_experiments.py --output-root outputs/my-validation-study
.\.venv\Scripts\python scripts/phase2_report.py --root outputs/my-validation-study
```

The sequential Phase2 runner records exact training commands, pilots, budgets and the SFT selection rule. It also expects the preserved original20-step SFT and8-step GRPO adapters for historical comparison; a fresh checkout does not include them. Reconstruct those with `run_pipeline.py` in a new output directory, then point the historical comparison paths at the resulting adapters or retain the published historical aggregates. Don't fabricate missing checkpoints. The [historical report](experiment-report.md) and [Phase2 report](phase2-experiment-report.md) document their different evaluation protocols.

The selected final adapter was trained directly from the unmodified model with `--from-base`, 16 updates, seed42, eight completions per group and the Phase2 config. No retraining is needed to analyze existing reports. Reconstructed training is a new experiment; seeds do not guarantee identical weights across devices or library kernels.

## Frozen held-out run

The recorded run is `outputs/phase3-final-20261030`. Its manifest fixes model and adapter files, code hashes, data hashes, software, hardware, prompts, parser, sampling and all metric definitions. The public manifest redacts the absolute local adapter path and retains the original private-manifest hash.

```powershell
# Inspect/freeze only. Run only after the correct existing finalists are available.
.\.venv\Scripts\python scripts/phase3_final.py --root outputs/a-new-preregistered-study
# Explicitly generate the one reserved evaluation for that frozen study.
.\.venv\Scripts\python scripts/phase3_final.py --root outputs/a-new-preregistered-study --execute
.\.venv\Scripts\python scripts/phase3_report.py --root outputs/a-new-preregistered-study
```

**Do not rerun the used final test to improve a reported score.** These commands describe a fresh preregistered study, not permission to tune on this project's used test. Once final outcomes are visible, any new model selection requires a new independent held-out set. The recorded test used all234 unique questions from240 rows, first occurrence order, paired seed42+i and eight candidates/model. Both models have the same256-token cap. The start marker blocks accidental repetitions; do not delete it to rerun. If a run is interrupted, preserve partial outputs and document the interruption rather than silently replacing evidence.

The final-report exporter refuses to overwrite an existing uniquely named public evidence directory. Reports retain original source hashes: a later documentation/dashboard commit does not change the evaluation snapshot.

## Tests and dashboard

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
.\.venv\Scripts\python -m pytest -q
Remove-Item Env:CUDA_VISIBLE_DEVICES
.\.venv\Scripts\python -m compileall -q src scripts tests
.\.venv\Scripts\python scripts/dashboard.py --smoke
.\.venv\Scripts\python scripts/dashboard.py
```

Gradio binds to127.0.0.1 with sharing disabled. Interactive samples are marked separately from measured results. Eight candidates can require more memory than one; errors should return useful messages. CI does not download a model, start training or access the reserved test.
