import numpy as np
import torch
import argparse
from ofdm_utils import generate_dataset
from qam_mapping import Demapping, PS, mapping_table
from config import K, pilotValue
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from model import OFDMNet_CNN, OFDMNet_LSTM, OFDMNet_Transformer

providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = ort.InferenceSession("model_transformer_fp16.onnx", providers=providers)
print(providers)
print("✅ Using ONNX providers:", ort_session.get_providers())


def generate_zc_sequence(u, N_ZC):
    n = np.arange(N_ZC)
    zc_seq = np.exp(-1j * np.pi * u * n * (n + 1) / N_ZC)
    return zc_seq

zc_pilot = generate_zc_sequence(u=1, N_ZC=17)
def run_tensorrt_engine(engine_path, X_val, batch_size=32):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    pred_all = []
    h2d_times, compute_times, d2h_times = [], [], []

    # ❗Dummy warm-up batch（不計入 benchmark）
    warmup_input = np.zeros((batch_size, 64, 5), dtype=np.float32)
    context.set_binding_shape(0, warmup_input.shape)
    input_shape = context.get_binding_shape(0)
    output_shape = context.get_binding_shape(1)

    d_input = cuda.mem_alloc(trt.volume(input_shape) * warmup_input.itemsize)
    d_output = cuda.mem_alloc(trt.volume(output_shape) * warmup_input.itemsize)
    bindings = [int(d_input), int(d_output)]

    cuda.memcpy_htod(d_input, warmup_input)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(np.empty(output_shape, dtype=np.float32), d_output)
    # 🔥 Warm-up 結束！

    # ✅ 正式推論
    for i in range(0, len(X_val), batch_size):
        x = X_val[i:i + batch_size].astype(np.float32)
        if x.shape[0] != batch_size:
            pad = np.zeros((batch_size - x.shape[0], *x.shape[1:]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)

        # Set dynamic shape again (in case it's different)
        context.set_binding_shape(0, x.shape)
        input_shape = context.get_binding_shape(0)
        output_shape = context.get_binding_shape(1)

        d_input = cuda.mem_alloc(trt.volume(input_shape) * x.itemsize)
        d_output = cuda.mem_alloc(trt.volume(output_shape) * x.itemsize)
        bindings = [int(d_input), int(d_output)]

        # Host to Device
        t0 = time.time()
        cuda.memcpy_htod(d_input, x)
        t1 = time.time()
        h2d_times.append((t1 - t0) * 1000)

        # Inference
        t2 = time.time()
        context.execute_v2(bindings)
        t3 = time.time()
        compute_times.append((t3 - t2) * 1000)

        # Device to Host
        output = np.empty(output_shape, dtype=np.float32)
        t4 = time.time()
        cuda.memcpy_dtoh(output, d_output)
        t5 = time.time()
        d2h_times.append((t5 - t4) * 1000)

        pred_all.append(output)

    return np.concatenate(pred_all, axis=0), sum(h2d_times), sum(compute_times), sum(d2h_times)

def run_traditional_equalizer(X_val, Y_val, pilotCarriers, dataCarriers, allCarriers):
    ber_total = 0
    total_bits = 0
    for i in range(len(X_val)):
        X_i = X_val[i]
        Y_true = Y_val[i]
        rx_freq = X_i[:, 0] + 1j * X_i[:, 1]
        pilots = rx_freq[pilotCarriers]
        Hest_at_pilots = pilots / zc_pilot
        Hest_abs = np.interp(allCarriers, pilotCarriers, np.abs(Hest_at_pilots))
        Hest_phase = np.interp(allCarriers, pilotCarriers, np.angle(Hest_at_pilots))
        Hest = Hest_abs * np.exp(1j * Hest_phase)
        rx_equal = rx_freq / Hest
        rx_data = rx_equal[dataCarriers]
        QAM_true = Y_true[:, 0] + 1j * Y_true[:, 1]
        bits_est, _ = Demapping(rx_data)
        bits_true, _ = Demapping(QAM_true)
        bits_est = PS(bits_est)
        bits_true = PS(bits_true)
        ber_total += np.sum(bits_est != bits_true)
        total_bits += len(bits_est)
    return ber_total / total_bits


def run_pytorch(model, X_val, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size)
    pred_all = []
    h2d_times, compute_times, d2h_times = [], [], []

    warmup_batches = 1
    with torch.no_grad():
        for i, (x_batch,) in enumerate(loader):
            # warm-up: 跳過前幾個 batch 的計時
            if i < warmup_batches:
                x_batch = x_batch.to(device)
                pred = model(x_batch)
                pred_np  = pred .cpu().numpy()
                pred_all.append(pred_np)
                continue
            t0 = time.time()
            x_batch = x_batch.to(device)
            t1 = time.time()
            h2d_times.append((t1 - t0) * 1000)
            
            t2 = time.time()
            pred = model(x_batch)
            t3 = time.time()
            compute_times.append((t3 - t2) * 1000)
           
            t4 = time.time()
            pred_np = pred.cpu().numpy()
            t5 = time.time()
            d2h_times.append((t5 - t4) * 1000)
            
            pred_all.append(pred_np)
    scale = (i + warmup_batches) / i
    print(h2d_times, compute_times, d2h_times)
    print(sum(h2d_times), sum(compute_times), sum(d2h_times))
    return np.concatenate(pred_all, axis=0), sum(h2d_times)*scale , sum(compute_times)*scale , sum(d2h_times)*scale


def run_onnx_fast(ort_session, X_val, batch_size=32):
    input_name = ort_session.get_inputs()[0].name
    pred_all = []
    h2d_times, compute_times, d2h_times = [], [], []

    for i in range(0, len(X_val), batch_size):
        x = X_val[i:i + batch_size].astype(np.float32)
        io_binding = ort_session.io_binding()

        t0 = time.time()
        x_tensor = torch.from_numpy(x).to("cuda")
        t1 = time.time()
        h2d_times.append((t1 - t0) * 1000)

        io_binding.bind_input(
            name=input_name,
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=x.shape,
            buffer_ptr=x_tensor.data_ptr()
        )

        io_binding.bind_output(
            name=ort_session.get_outputs()[0].name,
            device_type='cuda',
            device_id=0
        )

        t2 = time.time()
        ort_session.run_with_iobinding(io_binding)
        t3 = time.time()
        compute_times.append((t3 - t2) * 1000)

        t4 = time.time()
        output = io_binding.copy_outputs_to_cpu()[0]
        t5 = time.time()
        d2h_times.append((t5 - t4) * 1000)

        pred_all.append(output)

    return np.concatenate(pred_all, axis=0), sum(h2d_times), sum(compute_times), sum(d2h_times)


def evaluate_over_snr_with_timing(model_name):
    SNR_list = [ 5 , 10, 15, 20, 25]
    Time_h2d_tensorrt, Time_compute_tensorrt, Time_d2h_tensorrt = [], [], []
    Time_h2d_pytorch, Time_compute_pytorch, Time_d2h_pytorch = [], [], []
    Time_h2d_onnx, Time_compute_onnx, Time_d2h_onnx = [], [], []
    BER_pytorch, BER_onnx, BER_traditional  , BER_tensorrt= [], [], [] , []

    model_dict = {
        "cnn": OFDMNet_CNN,
        "lstm": OFDMNet_LSTM,
        "transformer": OFDMNet_Transformer
    }

    model = model_dict[model_name]()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(f"trained_model_{model_name}.pt", map_location=device))
    model.to(device)

    def compute_ber(pred_all, Y):
        ber_total = 0
        total_bits = 0
        for i in range(len(pred_all)):
            est = pred_all[i][:, 0] + 1j * pred_all[i][:, 1]
            true = Y[i][:, 0] + 1j * Y[i][:, 1]
            bits_est, _ = Demapping(est)
            bits_true, _ = Demapping(true)
            bits_est = PS(bits_est)
            bits_true = PS(bits_true)
            ber_total += np.sum(bits_est != bits_true)
            total_bits += len(bits_est)
        return ber_total / total_bits

    for snr in SNR_list:
        print(f"\n🚀 SNR = {snr}")
        X, Y, pilotCarriers, dataCarriers, allCarriers = generate_dataset(mapping_table, num_samples=320, SNR_range=(snr, snr))

        # Traditional equalizer
        ber_traditional = run_traditional_equalizer(X, Y, pilotCarriers, dataCarriers, allCarriers)
        BER_traditional.append(ber_traditional)

        # PyTorch
        pred_pytorch, h2d, compute, d2h = run_pytorch(model, X)
        ber_pytorch = compute_ber(pred_pytorch, Y)
        BER_pytorch.append(ber_pytorch)
        Time_h2d_pytorch.append(h2d)
        Time_compute_pytorch.append(compute)
        Time_d2h_pytorch.append(d2h)

        # ONNX
        pred_onnx, h2d, compute, d2h = run_onnx_fast(ort_session, X)
        ber_onnx = compute_ber(pred_onnx, Y)
        BER_onnx.append(ber_onnx)
        Time_h2d_onnx.append(h2d)
        Time_compute_onnx.append(compute)
        Time_d2h_onnx.append(d2h)


        pred_trt, h2d, compute, d2h = run_tensorrt_engine("model_transformer_model_fp16.engine", X)
        ber_trt = compute_ber(pred_trt, Y)
        BER_tensorrt.append(ber_trt)
        Time_h2d_tensorrt.append(h2d)
        Time_compute_tensorrt.append(compute)
        Time_d2h_tensorrt.append(d2h)
       
        print(f"✅ Traditional BER: {ber_traditional:.5f}")
        print(f"✅ PyTorch     BER: {ber_pytorch:.5f}")
        print(f"✅ ONNX        BER: {ber_onnx:.5f}")
        print(f"✅ TensorRT     BER: {ber_trt:.5f}")
    # Plot timeline
    def plot_timeline(h2d_list, compute_list, d2h_list, label, filename, color_compute):
        plt.figure(figsize=(10, 4))
        y_pos = np.arange(len(SNR_list))
        plt.barh(y_pos, h2d_list, color='gold', edgecolor='black', label='H2D')
        plt.barh(y_pos, compute_list, left=h2d_list, color=color_compute, edgecolor='black', label='Compute')
        plt.barh(y_pos, d2h_list, left=np.add(h2d_list, compute_list), color='limegreen', edgecolor='black', label='D2H')
        plt.yticks(y_pos, [f"SNR={snr}" for snr in SNR_list])
        plt.xlabel("Time (ms)")
        plt.title(f"Timeline Breakdown: {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    # Plot BER comparison
    def plot_ber():
        plt.figure(figsize=(8, 5))
        
        plt.plot(SNR_list, BER_onnx, marker='s', label="ONNX", color="tomato")
        plt.plot(SNR_list, BER_traditional, marker='^', linestyle='--', label="Traditional", color="gray")
        plt.plot(SNR_list, BER_pytorch, marker='o', label="PyTorch", color="dodgerblue")
        plt.yscale('log')
        plt.xlabel("SNR (dB)")
        plt.ylabel("Bit Error Rate (log scale)")
        plt.title("BER vs. SNR")
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("ber_comparison.png", dpi=300)
        plt.show()

    plot_timeline(Time_h2d_pytorch, Time_compute_pytorch, Time_d2h_pytorch, "PyTorch", "timeline_pytorch_measured.png", 'dodgerblue')
    plot_timeline(Time_h2d_onnx, Time_compute_onnx, Time_d2h_onnx, "ONNX", "timeline_onnx_measured.png", 'tomato')
    plot_timeline(Time_h2d_tensorrt, Time_compute_tensorrt, Time_d2h_tensorrt, "TensorRT", "timeline_tensorrt_measured.png", 'orange')
    plot_ber()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transformer")
    args = parser.parse_args()
    evaluate_over_snr_with_timing(args.model)
