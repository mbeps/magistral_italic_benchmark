# Magistral ITALIC Benchmark

This repository contains tooling and evaluation code to benchmark Magistral-family models on the ITALIC multiple-choice dataset.
It supports both standard (non-reasoning) evaluation and explicit reasoning (chain-of-thought) evaluation modes via the provided notebooks.

> Note: these notebooks and the benchmark in this repository do not use vLLM — inference is performed using the typical Transformers/PyTorch stack and relies on quantization (bitsandbytes) where configured.

## Requirements

- Python 3.10 to 3.12
- GPU recommended (CUDA-enabled) for reasonable performance when running quantized models. Quantization (4-bit / QLoRA workflows) relies on CUDA and bitsandbytes for efficient inference.

If you need GPU acceleration, install a PyTorch wheel that matches your CUDA runtime.

## Stack

- transformers — model and tokenizer utilities
- torch — core deep learning runtime (install the wheel that matches your CUDA runtime if you need GPU)
- bitsandbytes — 4-bit quantization runtime used for efficient inference
- PEFT (peft) — utilities for merging and evaluating QLoRA adapters
- pandas — data handling and tabulation
- numpy — numerical utilities
- tqdm — progress bars
- python-dotenv — load environment variables from a `.env` file
- datasets — dataset utilities and I/O
- scikit-learn — evaluation utilities and metrics
- trl — training / SFT utilities (optional)
- jupyter — interactive notebooks
- tiktoken — tokenization utilities (when used)

## Example dependency install (minimal)

Install a core set of runtime packages (adjust PyTorch / CUDA wheel to match your system):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install transformers torch peft bitsandbytes pandas numpy tqdm python-dotenv datasets scikit-learn trl jupyter tiktoken
```

Notes:
- `bitsandbytes` and GPU-enabled `torch` may require special wheel selection to match your CUDA driver.
- If you plan to use QLoRA adapters, `peft` is required.

## Set up

1. Create and activate a virtual environment (see example above).
2. Install the dependencies listed in the previous section.
3. Place environment values (for example, `HF_TOKEN` for Hugging Face access) in a `.env` file at the repo root. The benchmark loader uses `python-dotenv` to load these values.

Example `.env` (optional):

```env
# HF_TOKEN=ghp_...
# Any other custom config keys used by the notebooks or scripts
```

## Notebooks and evaluation modes

This project provides interactive notebooks and code to run two evaluation modes:

- `standard.ipynb` — standard (non-reasoning) multiple-choice evaluation.
- `reasoning.ipynb` — explicit reasoning (chain-of-thought) evaluation where the model is prompted to produce a reasoning trace before an answer.

Both notebooks are located at the repository root and are the recommended way to reproduce the experiments and inspect intermediate outputs.

## Outputs

- Detailed results CSV: `results/<variant>/*.csv`
- Summary JSON: `results/<variant>/*_summary.json`

The `run_benchmark()` flow in `magistral_benchmark/benchmark.py` saves:
- a detailed CSV (`*_results.csv`) with per-question predictions
- a JSON summary (`*_summary.json`) containing model metadata, dataset info, and aggregated metrics

## Configuration

- Set environment variables in a `.env` file (for example `HF_TOKEN`) if required for model/tokenizer downloads.
- The main benchmark configuration is provided by the `MagistralBenchmarkConfig` class in `magistral_benchmark/config.py` (used by the `MagistralBenchmark` runner).

## Notes & tips

- This project uses quantization and the Transformers + bitsandbytes route for efficient inference rather than vLLM. If you previously used examples that rely on vLLM, note that those are not used here because quantized models in this repo are loaded with the Transformers/BitsAndBytes configuration.
- If you run into GPU memory issues, try reducing batch size or use smaller / more heavily quantized models.
- When merging QLoRA adapters, the code will attempt to merge the adapter into the base model and save a merged checkpoint; ensure you have sufficient disk space.

## References
- [ITALIC: An Italian Culture-Aware Natural Language Benchmark](https://aclanthology.org/2025.naacl-long.68.pdf)