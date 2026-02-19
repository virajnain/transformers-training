# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from model2 import ForecastingModel
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader

DATA_SIZE = 20000
dt = 0.01
theta = 1
sigma = 0.2

x = np.zeros((DATA_SIZE, 2))

for t in range(1, DATA_SIZE):
    dW = np.sqrt(dt) * np.random.randn(2)
    x[t] = x[t-1] - theta * x[t-1] * dt + sigma * dW

# Normalize each dimension separately
x_mean = x.mean(axis=0)  # shape (2,)
x_std = x.std(axis=0)    # shape (2,)
x_norm = (x - x_mean) / x_std

# Create dataset: input (seq_len, 2), output (2,)
seq_len = 100
X = np.array([x_norm[ii:ii+seq_len] for ii in range(0, x_norm.shape[0]-seq_len)])  # (-1, seq_len, 2)
Y = np.array([x_norm[ii+seq_len]    for ii in range(0, x_norm.shape[0]-seq_len)])  # (-1, 2)

# Training setup
device = "cpu"
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2

model = ForecastingModel(seq_len, embed_size=64, nhead=4, dim_feedforward=256, dropout=0.3, device=device)
model.to(device)
criterion = torch.nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, gamma=0.99)

# Chronological train/val split
dataset = TensorDataset(torch.Tensor(X).to(device), torch.Tensor(Y).to(device))
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
val_dataset   = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

# Training loop with early stopping
best_val_loss = float('inf')
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    train_loss = 0.0
    for xx, yy in train_loader:
        optimizer.zero_grad()

        if np.random.rand() < 0.3:
            xx = xx + torch.randn_like(xx) * 0.1

        out = model(xx)
        loss = criterion(out, yy)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xx, yy in val_loader:
            out = model(xx)
            loss = criterion(out, yy)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    scheduler.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"  → New best model saved (val_loss={val_loss:.6f})")

# Load best model
model.load_state_dict(torch.load('best_model.pt'))
print(f"Loaded best model with val_loss={best_val_loss:.6f}")

# Plot loss curves
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label="Train Loss")
ax.plot(val_losses,   label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Huber Loss")
ax.set_title("Training vs Validation Loss")
ax.legend()
fig.savefig("./img/ouloss_curves.png")

# Diagnostic: check pred vs target stats
model.eval()
with torch.no_grad():
    preds, actuals = [], []
    for xx, yy in val_loader:
        preds.extend(model(xx).cpu().numpy())
        actuals.extend(yy.cpu().numpy())
preds   = np.array(preds)
actuals = np.array(actuals)
print(f"Pred   mean: {preds.mean(axis=0)}, std: {preds.std(axis=0)}")
print(f"Target mean: {actuals.mean(axis=0)}, std: {actuals.std(axis=0)}")

# Prediction loop (autoregressive, in normalized space)
FORCAST_EXTENDED = 2 * DATA_SIZE
model.eval()
x_extended = list(x_norm)  # list of (2,) arrays

for ff in range(FORCAST_EXTENDED):
    xx = np.array(x_extended[len(x_extended)-seq_len:len(x_extended)])  # (seq_len, 2)
    with torch.no_grad():
        yy = model(torch.Tensor(xx).reshape(1, seq_len, 2).to(device))
    x_extended.append(yy.cpu().numpy().reshape(2,))

x_extended = np.array(x_extended)  # (DATA_SIZE + FORCAST_EXTENDED, 2)

# Denormalize
x_denorm = x_extended * x_std + x_mean

# Plot predictions for both dimensions
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for dim, ax in enumerate(axes):
    ax.plot(range(DATA_SIZE), x_denorm[:DATA_SIZE, dim], label="Training")
    ax.plot(range(DATA_SIZE, DATA_SIZE + FORCAST_EXTENDED),
            x_denorm[DATA_SIZE:DATA_SIZE+FORCAST_EXTENDED, dim],
            'r--', label="Predicted")
    ax.set_ylabel(f"x{dim+1}")
    ax.legend()
axes[0].set_title("2D OU Process Forecast")
axes[1].set_xlabel("Timestep")
fig.savefig("./img/ou2d_extended.png")