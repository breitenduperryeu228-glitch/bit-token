# bit-token

Cross-model multilingual benchmark studying **`Token/s` vs `UTF-8 Bit/s`** as
generation-throughput metrics.

This repository is a **code + data** mirror of the working project
[`token`](https://github.com/...). It contains every executable artifact and
the raw benchmark outputs, but no papers, figures, editor caches, or compiled
files.

## What's in this repo

```
bit-token/
├── monitor.sh                          # Periodic progress reporter
└── benchmark/
    ├── __init__.py
    ├── configs/
    │   └── config.yaml                 # Models, languages, params
    ├── datasets/
    │   ├── __init__.py
    │   ├── samples.jsonl               # Aligned FLORES-200 samples
    │   ├── dataset_info.json
    │   └── _cache/
    │       └── flores200_devtest.parquet
    ├── prompts/
    │   ├── experiment_a.json           # Fixed-token prompts
    │   └── experiment_b.json           # Fixed-semantic-length prompts
    ├── scripts/
    │   ├── __init__.py
    │   ├── ollama_client.py            # Thin wrapper around ollama-python
    │   ├── run_benchmark.py            # Main entry — full matrix runner
    │   ├── calculate_metrics.py        # UTF-8 bytes/bits + derived metrics
    │   ├── analyze_results.py          # Spearman, Kendall, ANOVA, effect sizes
    │   ├── visualize.py                # Figure 1–7
    │   ├── visualize_thinking_bias.py
    │   ├── visualize_vram.py
    │   ├── report.py                   # Final REPORT.md generator
    │   ├── progress_report.py
    │   ├── watch_progress.py
    │   ├── compare_q4_q8.py
    │   ├── model_orchestrator.py
    │   ├── host_telemetry.py
    │   ├── analyze_thinking_bias.py
    │   ├── start_benchmark.sh
    │   ├── stop_benchmark.sh
    │   ├── resume.sh
    │   ├── status_alt.sh
    │   ├── stop_alt.sh
    │   ├── q8_run.sh
    │   └── post_run_q8.sh
    ├── results/
    │   └── raw/                        # 176 020 per-run JSONs (~813 MB)
    └── logs/
        └── *.log                       # 21 runtime log files
```

## Quick start

```bash
# 1. Build dataset (downloads FLORES-200 once, caches locally)
python3 -m benchmark.datasets.build_dataset --config benchmark/configs/config.yaml

# 2. Launch the full benchmark in the background (RESUMES by default)
./benchmark/scripts/start_benchmark.sh
./benchmark/scripts/start_benchmark.sh --clean     # wipe then start fresh
./benchmark/scripts/start_benchmark.sh --no-resume # overwrite
./benchmark/scripts/start_benchmark.sh --models qwen3:0.6b,lfm2.5-350m
./benchmark/scripts/start_benchmark.sh --samples 3 --repeats 3

# 3. Pause / resume at any time
./benchmark/scripts/stop_benchmark.sh
./benchmark/scripts/start_benchmark.sh    # auto-resumes

# 4. Watch live progress
python3 -m benchmark.scripts.watch_progress --interval 5

# 5. After benchmark finishes — analysis chain
python3 -m benchmark.scripts.calculate_metrics
python3 -m benchmark.scripts.analyze_results
python3 -m benchmark.scripts.visualize
python3 -m benchmark.scripts.report
```

## Hardware baseline

* GPU: NVIDIA RTX 4060 (8 GB VRAM)
* RAM: 15 GB
* Ollama: 0.23.0
* Models invoked through the local Ollama daemon (`http://localhost:11434`)

## Repo size

* ~837 MB total
* ~176 000 files (most of which are per-run benchmark JSONs in
  `benchmark/results/raw/`)

If you only need the code, clone with `--depth 1` and ignore `benchmark/results/raw/`
locally after pulling.

## License

See upstream `token` project.
