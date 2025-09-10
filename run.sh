#!/bin/bash

# === 預設參數 ===
stage=0                 # 從哪個階段開始執行
model="transformer"     # 預設模型名稱（可選 cnn, lstm, transformer）

# === 處理輸入參數 ===
if [ $# -ge 1 ]; then
  stage=$1
fi

if [ $# -ge 2 ]; then
  model=$2
fi

pt_file="trained_model_${model}.pt"
onnx_output="model_${model}"  # 最終會生成 model_${model}.engine

# === Stage 1: 訓練模型 ===
if [ $stage -eq 1 ]; then
  echo "🔧 Stage 1: Training model [$model]..."
  python run_training.py --model "$model"
fi

# === Stage 2: 轉 ONNX / FP16 / TensorRT Engine ===
if [ $stage -eq 2 ]; then
  echo "🔄 Stage 2: Converting [$pt_file] to ONNX, FP16, and TensorRT engine..."
  python convert_to_onnx_fp32_fp16.py \
    --pt_path "$pt_file" \
    --model "$model" \
    --output_prefix "$onnx_output"
fi

# === Stage 3: 測試模型 BER + 時間 ===
if [ $stage -eq 3 ]; then
  echo "📊 Stage 3: Evaluating model [$model]..."
  python auto_eval.py --model "$model"
fi
