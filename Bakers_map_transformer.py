# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from forecasting_model import ForecastingModel
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader, random_split

DATA_SIZE = 20000
'''
dt = 0.01
theta = 1
sigma = 0.2

x = np.zeros(DATA_SIZE)

for t in range(1, DATA_SIZE):
    dW = np.sqrt(dt) * np.random.randn(1)
    x[t] = x[t-1] - theta*x[t-1]*dt + sigma*dW  # OU Process
'''

s = np.array([1.5, 0.5, 1.5, 0.5])

def torus_map(x, s):
    x1, x2 = x
    s1, s2, s3, s4 = s
    new_x1 = 2*x1 + (s1 + s2 * np.sin(2*x2) / 2) * np.sin(x1) - np.floor(x1 / np.pi) * 2 * np.pi
    new_x2 = (x2 + (s4 + s3 * np.sin(x1)) * np.sin(2*x2) + np.floor(x1 / np.pi) * 2*np.pi) / 2
    return np.array([new_x1 % (2*np.pi), new_x2 % (2*np.pi)])

# Generate trajectory
traj = np.zeros((DATA_SIZE, 2))
traj[0] = np.random.uniform(0, 2*np.pi, size=2)  # random initial condition
for t in range(1, DATA_SIZE):
    traj[t] = torus_map(traj[t-1], s)

# Use x^(1) component as your 1D signal (or both if you want multivariate)
x = traj[:, 0]


x_mean, x_std = x.mean(), x.std()
x_norm = (x - x_mean) / x_std

seq_len = 100
# Build X, Y from x_norm instead of x
X = np.array([x_norm[ii:ii+seq_len] for ii in range(0, x_norm.shape[0]-seq_len)]).reshape((-1, seq_len, 1))
Y = np.array([x_norm[ii+seq_len] for ii in range(0, x_norm.shape[0]-seq_len)]).reshape((-1, 1))


# Training Loop
device = "cpu"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2  # 20% for validation

model = ForecastingModel(seq_len, embed_size=64, nhead=2, dim_feedforward=256, dropout=0.3, device=device)
model.to(device)
criterion = torch.nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optimizer, gamma=0.99)

# Split dataset into train and validation
dataset = TensorDataset(torch.Tensor(X).to(device), torch.Tensor(Y).to(device))
# Replace random_split with a chronological split
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_dataset = torch.utils.data.Subset(dataset, range(train_size))
val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    train_loss = 0.0
    for xx, yy in train_loader:
        optimizer.zero_grad()

        if np.random.rand() < 0.3:  # 30% of the time
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


# Plot Loss Curves
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label="Train Loss")
ax.plot(val_losses, label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Huber Loss")
ax.set_title("Training vs Validation Loss")
ax.legend()
fig.savefig("./img/loss_curves.png")

model.eval()
with torch.no_grad():
    preds = []
    actuals = []
    for xx, yy in val_loader:
        preds.extend(model(xx).flatten().tolist())
        actuals.extend(yy.flatten().tolist())

print(f"Pred   mean: {np.mean(preds):.4f}, std: {np.std(preds):.4f}")
print(f"Target mean: {np.mean(actuals):.4f}, std: {np.std(actuals):.4f}")

# Prediction Loop
FORCAST_EXTENDED = 2*DATA_SIZE
# During prediction loop, denormalize each step
model.eval()
x_extended = list(x_norm)  # work in normalized space

for ff in range(FORCAST_EXTENDED):
    xx = np.array(x_extended[len(x_extended)-seq_len:len(x_extended)])
    yy = model(torch.Tensor(xx).reshape(1, xx.shape[0], 1).to(device))
    x_extended.append(yy.detach().cpu().numpy().reshape(1,)[0])

x_extended = np.array(x_extended)

# Denormalize — reverse of (x - mean) / std
x_denorm = x_extended * x_std + x_mean

# Plot Predictions
fig = plt.figure(figsize=(12, 6))
plt.plot(range(DATA_SIZE), x_denorm[:DATA_SIZE], label="Training")
plt.plot(range(DATA_SIZE, DATA_SIZE + FORCAST_EXTENDED), x_denorm[DATA_SIZE:DATA_SIZE+FORCAST_EXTENDED], 'r--', label="Predicted")
plt.legend()
fig.savefig("./img/papergraph.png")