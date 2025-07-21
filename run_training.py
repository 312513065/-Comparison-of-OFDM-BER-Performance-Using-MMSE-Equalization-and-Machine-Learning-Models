import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import argparse
from qam_mapping import Demapping, PS, mapping_table
from ofdm_utils import generate_dataset
from model import OFDMNet_CNN, OFDMNet_LSTM, OFDMNet_Transformer
from config import epochs , batch_size
# Argument parser for model selection
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['cnn', 'lstm', 'transformer'], default='cnn', help='Choose model type')
args = parser.parse_args()

# Load dataset
X, Y, pilotCarriers, dataCarriers, allCarriers = generate_dataset(mapping_table)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)
X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=0.2)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size = batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size = batch_size)

# Model selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.model == 'cnn':
    model = OFDMNet_CNN().to(device)
elif args.model == 'lstm':
    model = OFDMNet_LSTM().to(device)
else:
    model = OFDMNet_Transformer().to(device)

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay = 1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.98, patience=10, verbose=True)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += criterion(pred, yb).item()
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}, Val Loss = {val_loss / len(val_loader):.4f}")

torch.save(model.state_dict(), f"trained_model_{args.model}.pt")
print(f"✅ 模型已儲存為 trained_model_{args.model}.pt")