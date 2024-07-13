#!/bin/bash

# 输出日志文件的目录
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 模型、k-shot和数据集的配置
MODELS=("qwen/Qwen2-0.5B-Instruct" "qwen/Qwen2-1.5B-Instruct")
K_VALUES=(0 1 2 5)
# K_VALUES=(0 1 2 5)
DATASET_NAME="CommonQA"

# 循环执行实验
for model in "${MODELS[@]}"; do
  for k in "${K_VALUES[@]}"; do
    MODEL_NAME="${model//\//-}"  # 替换斜杠
    echo "Running $model with $k-shot on $DATASET_NAME"
    LOG_FILE="$LOG_DIR/${MODEL_NAME}_${DATASET_NAME}_${k}-shot.log"
    python main.py --model_name "$model" --k $k --dataset_name "$DATASET_NAME" &> "$LOG_FILE"
    echo "Finished $model with $k-shot. Output logged to $LOG_FILE"
  done
done
