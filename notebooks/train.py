import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gan_synthesis.datasets.dataset import Dataset
from gan_synthesis.mask_vae_models.decoder import Decoder
from gan_synthesis.mask_vae_models.encoder import Encoder
from gan_synthesis.mask_vae_models.vae import VAE, kl_divergence

# Enable live plotting
plt.ion()
fig, ax = plt.subplots()
(line1,) = ax.plot([], [], label="Train Loss")
(line2,) = ax.plot([], [], label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Live Training/Validation Loss")
ax.legend()

# Instantiate data loaders
dataset = Dataset()
train_set, test_set = dataset.split(0.8)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

# Instantiate models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)
model = VAE(encoder, decoder).to(device)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Tracking
train_losses = []
val_losses = []
val_accuracies = []

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    for _, seg in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False):
        seg_input = seg.to(torch.float32).to(device)
        seg_target = seg.to(device)
        print(seg_target.shape)

        recon, mu, logvar = model(seg_input)
        kld_loss = kl_divergence(mu, logvar)
        ce_loss = criterion(recon, seg_target)
        loss = ce_loss + 5e-3 * kld_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for _, seg in tqdm(test_loader, desc="Validation", leave=False):
            seg = seg.to(device)
            seg_input = seg.to(torch.float32).to(device)
            seg_target = seg.to(device)

            recon, mu, logvar = model(seg_input)
            kld_loss = kl_divergence(mu, logvar)
            ce_loss = criterion(recon, seg_target)
            loss = ce_loss + 5e-3 * kld_loss
            running_val_loss += loss.item()

            pred = torch.argmax(recon, dim=1)
            correct += (pred == seg_target).sum().item()
            total += seg.numel()

    avg_val_loss = running_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    acc = correct / total
    val_accuracies.append(acc)

    print(
        f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {acc:.4f}"
    )

    # Update live plot
    line1.set_xdata(range(1, len(train_losses) + 1))
    line1.set_ydata(train_losses)
    line2.set_xdata(range(1, len(val_losses) + 1))
    line2.set_ydata(val_losses)

    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)

# Keep final plot open
plt.ioff()
plt.show()
