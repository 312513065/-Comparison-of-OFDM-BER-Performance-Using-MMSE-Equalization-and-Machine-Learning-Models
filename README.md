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

$$
y = h \cdot x + n
$$

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


---

## 🧠 Models Implemented

| Model             | Description                                                                                     | Parameter Count      |
|------------------|-------------------------------------------------------------------------------------------------|-----------------------|
| **SimpleCSINet3D**       | 3D CNN that processes input as a volume across (Rx, Tx, pilots). Fast and effective for spatial features. | 🏆 **18,490**          |
| **LSTMCSINet**           | Treats pilots as sequential input. Captures frequency correlation. Best when channel response has structure across L. | 235,552               |
| **TransformerCSINet**    | Uses self-attention across pilot subcarriers. Learns global relationships among antennas and subcarriers. | 537,760               |

**Input Format**: Each model receives input of shape `(batch, 4, N_rx, N_tx, L)`,  
where the 4 channels represent `[x_real, x_imag, y_real, y_imag]`.  
Real and imaginary parts of pilot and received signals are stacked along the channel dimension.

**Output Format**: The model outputs a predicted channel tensor \( \hat{H} \) of shape `(batch, N_rx, N_tx, L, 2)`,  
where the last dimension contains the real and imaginary components of the estimated channel matrix.

**Conclusion**: Among all models, the 3D CNN has the lowest parameter count (18K) while achieving the best trade-off between accuracy and latency.  
This lightweight nature makes it ideal for real-time CSI estimation and hardware deployment, especially under edge-device constraints.
### Input:

All models take 

input:
<pre> x_input: (batch, 64, 5) # 4 channels = [ y_real, y_imag, pilot_mask, pilot_real_value  , pilot_img_value ]  </pre> 

output:
<pre> y_output: (batch, 55, 2) # 2 channels = [ symbol_real, symbol_imag ]  </pre>




## 📊 Results & Visualizations

Below we present training results on both **Rayleigh** and **DeepMIMO** datasets for each model. We include:

- **Loss Curves**: Training vs. validation MSE loss across epochs. The dotted line indicates MMSE baseline.  
- **Heatmaps**: Visual comparison of true vs. predicted channel magnitudes for a selected sample.  
- **Residual Histograms**: Distribution of element-wise difference between predicted and true channels.

| Dataset   | Model       | Loss Curve                                  | MODEL BER vs MMSE BER                                  |                  
|-----------|-------------|----------------------------------------------|----------------------------------------------|
| Rayleigh  | CNN         | ![](results/rayleigh_cnn_loss.png)          | ![](results/rayleigh_cnn_heatmap.png)        | 
| Rayleigh  | LSTM        | ![](results/rayleigh_lstm_loss.png)         | ![](results/rayleigh_lstm_heatmap.png)       | 
| Rayleigh  | Transformer | ![](results/rayleigh_transformer_loss.png)  | ![](results/rayleigh_transformer_heatmap.png)| 

---

### 📁 Image Naming Convention

Please name your image files as:

- `rayleigh_cnn_loss.png`
- `rayleigh_cnn_heatmap.png`
- `rayleigh_cnn_hist.png`
- `rayleigh_lstm_loss.png`
- ...

Place all images under the `results/` directory.

---
---
