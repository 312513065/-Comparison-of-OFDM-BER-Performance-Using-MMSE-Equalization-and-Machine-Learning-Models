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

We estimate the wireless channel **h** using OFDM pilot signals **x**, with the received signal **y**:

\[
y = h * x + n
\]

Where:
- `x`: known OFDM symbol after 64-point IFFT, consisting of 16-QAM modulated data and 9 pilot subcarriers
- `y`: received signal at the receiver, after passing through the 3-tap multipath channel and additive white Gaussian noise (AWGN)
- `h`: channel impulse response, shape (3,), a complex-valued 3-tap channel 
- `n`: additive white Gaussian noise

### Signal Generation:
- 16-QAM modulation generates baseband symbols.
- Symbols are mapped onto 64 OFDM subcarriers, with 9 designated as pilot subcarriers.
- A 64-point IFFT converts to time domain.
- A cyclic prefix is added before transmission.

### 🧪 Channel Data & Augmentations
- The signal passes through a 3-tap channel **h** (frequency-selective fading).
- Additive white Gaussian noise **n** is added at the receiver, with SNR randomly chosen between 5 dB and 25 dB.
- The received signal **y** is processed by MMSE or machine learning-based equalization (CNN / LSTM / Transformer)



## 🧠 Models Implemented

| Model               | Description                                     |
|---------------------|-------------------------------------------------|
| `SimpleCSINet3D`     | 3D CNN over (rx, tx, pilot), ~80K params        |
| `LSTMCSINet`         | Models pilot as a sequence, uses LSTM layers    |
| `TransformerCSINet`  | Uses attention across subcarriers               |

Input:
