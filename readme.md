# LLM-Evaluation Project

## 概述

该项目使用指定模型，基于CP和MCP两种提示方式，在给定数据集上进行 k-shot 推理，并计算其准确率。

关于本次项目的数据分析、结论、思考，请参见 `LLM_Evaluation_research_note.md`

## 文件结构

- `main.py`: 主脚本，处理参数解析并调用其他模块。
- `model_inference.py`: 模型加载和推理逻辑。
- `data_processing.py`: 数据加载和示例提取。
- `evaluation.py`: 评估和准确率计算。
- `config.py`: 定义可用的模型和数据集。
- `aggregate_and_visualize.py`: 合并并可视化评测结果。
- `results/aggregate_results.xlsx`: 所有评测结果汇总表。
- `requirements.txt`: 列出所需的 Python 库。
- `README.md`: 项目说明和使用说明。
- `LLM_Evaluation_research_note.md`: 研究结论等。

## 使用方法

1. 安装依赖库：

    ```bash
    pip install -r requirements.txt
    ```

2. 运行 `main.py`：

    ```bash
    python main.py --model_name qwen/Qwen2-1.5B-Instruct --k 0 --dataset_name CommonQA
    ```

3. 或者，可以选择使用bash脚本运行实验，在此前需要调整脚本中的参数

   ```bash
   chmod +x run_experiments.sh
   ./run_experiments.sh
   ```

## 参数说明

- `--model_name`: 要使用的模型名称。
- `--k`: k-shot 示例的数量。
- `--dataset_name`: 要使用的数据集名称。
