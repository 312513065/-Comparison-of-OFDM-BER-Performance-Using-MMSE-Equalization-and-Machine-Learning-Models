# Comparison-of-OFDM-BER-Performance-Using-MMSE-Equalization-and-Machine-Learning-Models

This project benchmarks lightweight deep learning models (CNN, LSTM, Transformer) for OFDM system EER (Equal Error Rate) analysis. It compares their performance with traditional MMSE equalization under synthetic Rayleigh channels

---

## 🧠 Project Highlights

- 📶 Supports synthetic Rayleigh and DeepMIMO ray-tracing datasets
- 🧹 Implements 3 model types: 3D CNN, LSTM, Transformer
- 🧪 Compares with MMSE baseline under realistic noise & distortion
- 📊 Includes visualizations: heatmaps, per-antenna comparisons, loss curves
- ⚙️ Benchmarks CUDA inference time — practical deployment focus

---

## 🖐 System Model

We estimate the MIMO channel matrix **H** using pilot signals **x**, with the received signal **y**:

\[
y = H \cdot x + n
\]

Where:

- `x`: known pilot symbol, shape (N_tx × L)
- `y`: received signal at RX, shape (N_rx × L)
- `H`: channel tensor, shape (N_rx × N_tx × L × 2) where last dim is [real, imag]
- `n`: additive white Gaussian noise

Each training sample represents a single CSI frame, consisting of `L=8` pilot subcarriers (not time steps), following real-world CSI-RS usage in OFDM.

---

## 🧪 Channel Data & Augmentations

- **Synthetic Rayleigh Channel**
  - Each element h_{i,j} ~ CN(0, 1)
  - i.i.d. fading with no TX/RX correlation
- **DeepMIMO (O1_60)**
  - Geometry-based ray-tracing CSI from real-world layout
  - Includes TX/RX correlation and multi-path structure

**Augmentations**
- Random SNR between 10–30 dB
- Random Zadoff-Chu pilot root index
- Optional IQ imbalance
- Optional 1-bit quantization noise

---

## 🧠 Models Implemented

| Model               | Description                                     |
|---------------------|-------------------------------------------------|
| `SimpleCSINet3D`     | 3D CNN over (rx, tx, pilot), ~80K params        |
| `LSTMCSINet`         | Models pilot as a sequence, uses LSTM layers    |
| `TransformerCSINet`  | Uses attention across subcarriers               |

Input:
