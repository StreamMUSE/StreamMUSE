# NLL Compute Core (NLL 计算核心)

This module provides the core computational tools for calculating the Negative Log-Likelihood (NLL) of MIDI files under a given model. 
本模块提供了核心计算工具，用于执行指定模型下 MIDI 文件的负对数似然（NLL）的正向推理与原始数据保留。

> **⚠️ ARCHITECTURAL NOTICE (架构修改通知) ⚠️**
> All evaluation aggregation, statistics (mean/median/variance), and heatmap plotting logic have been moved to the `eval` repository's `eval_toolkit` to maintain separation of concerns and reduce bloated dependencies in the core engine. **This directory should only handle raw tensor computations and basic JSON dataset dumping.**
> 所有的评估聚合、统计指标（中位数/平均值/方差）及热力图绘制逻辑均已迁移至 `eval` 仓库的 `eval_toolkit` 中。这是为了保证环境职责清晰，避免核心引擎的依赖变得臃肿。**当前目录仅负责最源头的张量计算与基础的 JSON 数据导出。**

## Structure (模块结构)

- **`core.py`**: Core mathematical tensor logic and single-file NLL calculation.`cal_nll`
  *(核心张量逻辑与单文件 NLL 计算 `cal_nll`)*
- **`batch.py`**: Batch processing logic for computing raw chunked NLL metrics across entire directories.
  *(目录级别批量计算原始 NLL 指标的逻辑处理)*
- **`run_nll_from_manifest.py`**: Generates massive raw JSON metrics from model checkpoints.
  *(根据配置清单批量生成模型推理检测点的数据)*
- **`runners/`**: Executable scripts for computational execution. *(可执行的核心计算触发脚本)*

## How to use (工作流对接)

1. **Compute (计算阶)**: 
   Use scripts from this module (e.g., `batch.py` or the runners) within the `StreamMUSE` workspace to run the heavy AI model inference and output raw JSON metric logs.
   *(在 StreamMUSE 项目中使用本模块的工具跑繁重的 AI 模型推理，并输出包含各个文件的基础分析数据的原始 JSON。)*

2. **Aggregate & Plot (聚合与可视化阶段)**: 
   Switch your workspace to the `eval` repository. Use the `eval_toolkit` (e.g., `stats.py` and `csv_exporter.py`) to parse the generated JSON logs, compute means and variances, and plot matrices/heatmaps.
   *(切换至 `eval` 仓库。使用在那里的 `eval_toolkit` 去解析刚才生成的原始日志，并负责统计加权评分、导出 CSV 以及画图。)*

## Executing the Compute Pipeline (执行计算案例)

Executable scripts are kept inside the `runners/` directory to preserve module purity. Usage instructions are available for each via the `--help` flag.  
*(独立的可执行脚本放在 `runners/` 目录中。可通过 `--help` 标志查看。)*

### 1. Evaluate an entire directory (评估生成的数据目录)
Calculate chunked NLLs for a folder of generated `.mid` files and save raw tensors/JSON.
*(推导出一个目录下所有 MIDI 文件每片段的 NLL 数据，生成原始 JSON 记录文件。)*

```bash
python -m nll_compute.runners.run_cal_nll \
  --midi_dir /path/to/midi/dir \
  --ckpt_path /path/to/model.ckpt \
  --save_json_path output/results.json \
  --window 384 --offset 128
```
or 

```bash
uv run -m nll_compute.runners.run_cal_nll \
  --midi_dir /path/to/midi/dir \
  --ckpt_path /path/to/model.ckpt \
  --save_json_path output/results.json \
  --window 384 --offset 128
```

*(Note: Use `run_nll_from_manifest.py` for testing large grids of combinations.)*
