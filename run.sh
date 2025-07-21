#!/bin/bash

# 預設設定
stage=0                 # 從哪個階段開始
model="cnn"             # 預設模型

# 如果有傳入參數，更新 stage 和 model
if [ $# -ge 1 ]; then
  stage=$1
fi

if [ $# -ge 2 ]; then
  model=$2
fi

# === Stage 1: 訓練模型 ===
if [ $stage -le 1 ]; then
  echo "🔧 Stage 1: Training model [$model]..."
  python run_training.py --model "$model"
fi

# === Stage 2: 測試模型 ===
if [ $stage -le 2 ]; then
  echo "📊 Stage 2: Evaluating model [$model]..."
  python auto_eval.py --model "$model"
fi