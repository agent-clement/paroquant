# Reproduction Scripts

These are the scripts to reproduce most results in the paper. The environment for all the experiments (except for baselines and reasoning tasks) can be set up with `pip install -e ".[gpu]"` from the repo root (see [`pyproject.toml`](../pyproject.toml)). The docker image for this environment is `ghcr.io/z-lab/paroquant:latest`.

We use pseudo-quantized models for all experiments, except for experiments on AWQ where we use [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) to run real-quantized models. ParoQuant's pseudo-quantized models used in the experiments can be downloaded from the `pseudo` directory at [`z-lab/paroquant-checkpoints`](https://huggingface.co/z-lab/paroquant-checkpoints).

## Optimization

To quantize and optimize a model with ParoQuant:

```
./experiments/optimize/4bit.sh <model> [<num_shards>]
```

`num_shards` is 1 by default and may need to be adjusted for large models to accommodate memory constraints.
You can also decrease `--batch-size` and increase `--gradient-accumulation-steps` within the script to reduce peak training memory usage. Please keep accumulation steps × batch size consistent.

We adjust the batch size, learning rate, and number of training samples for LLaMA-3-70B. Please use [`experiments/optimize/4bit_70b.sh`](./optimize/4bit_70b.sh) instead for LLaMA-3-70B.

The optimized checkpoints will be saved to `./output/<model_name>`. To create a Hugging Face model with the checkpoints, use [`paroquant/cli/convert.py`](../paroquant/cli/convert.py) (`--mode pseudo` for pseudo-quantized, `--mode real` for INT4).

## FM&M Additions

The local optimization flow now keeps the model in its runtime dtype instead of forcing FP16 during optimization. For Qwen 3.5 VLM models this means optimization runs in BF16 end to end, which avoids the earlier BF16/FP16 mismatch in the optimizer while preserving the existing FP16-serving path during conversion.

Calibration datasets also support weighted mixes and Hugging Face dataset specs. Built-in aliases such as `wikitext2`, `c4`, `redpajama`, and `pileval` still work, and weights are optional. When weights are omitted, they default to `1.0`.

Examples:

```bash
python -m paroquant.cli.optimize \
  --model Qwen/Qwen3.5-0.8B \
  --datasets \
    "wikitext2?weight=0.1" \
    "c4?weight=0.1" \
    "redpajama?weight=0.1" \
    "hf://your-org/your-domain-dataset?split=train&text_key=text&weight=0.7" \
  --val-dataset "hf://your-org/your-domain-dataset?split=validation&text_key=text"
```

The `text_key` query parameter names the dataset column that contains the text to tokenize. If the column is already named `text`, it can be omitted. Hugging Face dataset specs also support `name=<config>`, `revision=<rev>`, and optional `trust_remote_code=true|false`.

For serving, continue to convert with explicit FP16 buffers unless you are intentionally testing another export dtype:

```bash
python -m paroquant.cli.convert \
  --model Qwen/Qwen3.5-0.8B \
  --result-dir output/Qwen3.5-0.8B \
  --output-path models/Qwen3.5-0.8B-PARO \
  --dtype float16
```

## Baselines

Scripts to obtain models quantized by baseline methods presented in the paper are in the [`baselines`](./baselines) directory. These models are used for perplexity and downstream task evaluation. Please refer to each script for its usage.

## Downstream Tasks

To evaluate downstream tasks, use [`tasks/reasoning.sh`](./tasks/reasoning.sh) for reasoning tasks and [`tasks/non_reasoning.sh`](./tasks/non_reasoning.sh) for non-reasoning tasks. Please note that they only support pseudo-quantized and AWQ-quantized models.

To run non-reasoning tasks:

```
./experiments/tasks/non_reasoning.sh <model>
```

We use a separate environment for reasoning tasks. To run reasoning tasks:

```
conda env create -f ./experiments/tasks/reasoning/environment.yml
conda activate paroquant-eval

./experiments/tasks/reasoning.sh <model> <seed> [<task0>, <task1>, ...]
```

> The docker image for this environment is `ghcr.io/z-lab/paroquant:eval-reasoning`.

The seeds we use in our paper are 42 for MMLU-Pro and 42, 0, 1 for other tasks.

## Throughput

Use [`throughput/bench.sh`](./throughput/bench.sh) to benchmark the decoding throughput of a real-quantized ParoQuant model:

```
./experiments/throughput/bench.sh <model>
```

## Ablation Studies

Scripts for the ablation studies are in the [`ablations`](./ablations/) directory. Their usage is similar to the ParoQuant optimization script.

## Plots

We provide the scripts to create the plots in the paper in the [`plots`](./plots/) directory. Some scripts require optimized checkpoints of linear layers, which can be downloaded from [`z-lab/paroquant-checkpoints`](https://huggingface.co/z-lab/paroquant-checkpoints).
