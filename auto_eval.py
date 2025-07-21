import numpy as np
import torch
import argparse
from ofdm_utils import generate_dataset
from qam_mapping import Demapping, PS, mapping_table
from config import K, pilotValue
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 匯入所有模型
from model import OFDMNet_CNN, OFDMNet_LSTM, OFDMNet_Transformer

# channel estimation
def run_traditional_equalizer(X_val, Y_val, pilotCarriers, dataCarriers, allCarriers):
    ber_total = 0
    total_bits = 0
    for i in range(len(X_val)):
        X_i = X_val[i]
        Y_true = Y_val[i]
        rx_freq = X_i[:, 0] + 1j * X_i[:, 1]
        pilots = rx_freq[pilotCarriers]
        Hest_at_pilots = pilots / pilotValue
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

def run_nn_equalizer(model, X_val, Y_val):
    model.eval()
    device = next(model.parameters()).device
    ber_total = 0
    total_bits = 0
    with torch.no_grad():
        for i in range(len(X_val)):
            x = torch.tensor(X_val[i:i+1], dtype=torch.float32).to(device)
            pred = model(x).cpu().numpy()[0]
            est = pred[:,0] + 1j * pred[:,1]
            true = Y_val[i][:,0] + 1j * Y_val[i][:,1]
            bits_est, _ = Demapping(est)
            bits_true, _ = Demapping(true)
            bits_est = PS(bits_est)
            bits_true = PS(bits_true)
            ber_total += np.sum(bits_est != bits_true)
            total_bits += len(bits_est)
    return ber_total / total_bits

def evaluate_over_snr(model_name):
    SNR_list = [5, 10, 15, 20, 25]
    BER_nn = []
    BER_trad = []

    # 選擇模型
    model_dict = {
        "cnn": OFDMNet_CNN,
        "lstm": OFDMNet_LSTM,
        "transformer": OFDMNet_Transformer
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_dict[model_name]().to(device)
    model.load_state_dict(torch.load(f"trained_model_{model_name}.pt", map_location=device))

    for snr in SNR_list:
        print(f"🔄 SNR = {snr}")
        X, Y, pilotCarriers, dataCarriers, allCarriers = generate_dataset(mapping_table, num_samples=300, SNR_range=(snr, snr))
        BER1 = run_traditional_equalizer(X, Y, pilotCarriers, dataCarriers, allCarriers)
        BER2 = run_nn_equalizer(model, X, Y)
        BER_trad.append(BER1)
        BER_nn.append(BER2)
        print(f"   ➤ Traditional BER: {BER1:.5f}, Neural Net BER: {BER2:.5f}")

    # Plot
    plt.figure(figsize=(8,5))
    plt.semilogy(SNR_list, BER_nn, marker='o', label=f'{model_name.upper()}')
    plt.semilogy(SNR_list, BER_trad, marker='s', linestyle='--', label='Traditional ZF')
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ber_vs_snr_{model_name}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", help="Model type: cnn | lstm | transformer")
    args = parser.parse_args()
    evaluate_over_snr(args.model)