# Configuration parameters for OFDM system
K = 64              # Subcarriers
CP = K // 4         # Cyclic Prefix length
P = 8               # Number of pilot carriers
pilotValue = 3 + 3j # Pilot symbol
mu = 4              # 16-QAM: bits per symbol
epochs = 20
batch_size = 32
#SNRdb = 25