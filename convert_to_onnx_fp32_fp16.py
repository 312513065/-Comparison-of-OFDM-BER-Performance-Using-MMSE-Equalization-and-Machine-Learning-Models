import torch
import argparse
import onnx
from onnxconverter_common import float16
import subprocess
import os
from model import OFDMNet_CNN, OFDMNet_LSTM, OFDMNet_Transformer

def export_onnx_and_fp16(pt_path, output_prefix, model_name, trtexec_path="trtexec"):
    model_dict = {
        "cnn": OFDMNet_CNN,
        "lstm": OFDMNet_LSTM,
        "transformer": OFDMNet_Transformer
    }

    if model_name not in model_dict:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model_dict[model_name]()
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 64, 5)  # Change this based on your model input shape

    onnx_fp32_path = f"{output_prefix}.onnx"
    onnx_fp16_path = f"{output_prefix}_fp16.onnx"
    engine_path = f"{output_prefix}_model_fp16.engine"

    # Step 1: Export FP32 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_fp32_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print(f"✅ Exported FP32 model to {onnx_fp32_path}")

    # Step 2: Convert to FP16 ONNX
    model_fp32 = onnx.load(onnx_fp32_path)
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
    onnx.save(model_fp16, onnx_fp16_path)
    print(f"✅ Converted and saved FP16 model to {onnx_fp16_path}")

    # Step 3: Build TensorRT engine using trtexec
    trtexec_cmd = [
        trtexec_path,
        f"--onnx={onnx_fp32_path}",
        f"--saveEngine={engine_path}",
        "--explicitBatch",
        "--fp16",
        "--verbose",
        "--minShapes=input:1x64x5",
        "--optShapes=input:32x64x5",
        "--maxShapes=input:64x64x5"
    ]

    print(f"🚀 Building TensorRT engine with: {' '.join(trtexec_cmd)}")

    try:
        subprocess.run(trtexec_cmd, check=True)
        print(f"✅ Saved TensorRT engine to {engine_path}")
    except subprocess.CalledProcessError as e:
        print("❌ trtexec failed.")
        print("🔍 Error output:")
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--model", type=str, choices=["cnn", "lstm", "transformer"], required=True, help="Model type")
    parser.add_argument("--output_prefix", type=str, required=True, help="Output prefix for .onnx/.engine files")
    parser.add_argument("--trtexec_path", type=str, default="trtexec", help="Path to trtexec executable")
    args = parser.parse_args()

    export_onnx_and_fp16(args.pt_path, args.output_prefix, args.model, trtexec_path=args.trtexec_path)
