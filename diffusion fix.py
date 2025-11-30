import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.amp import autocast, GradScaler
import math
from tqdm import tqdm

# ==================== CONFIG =====================
IMG_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 200
T = 200
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ================ UTILITY FUNCTIONS ================
def show_tensor_image(image_tensor, filename):
    image = image_tensor.detach().cpu().squeeze()
    image = (image * 0.5 + 0.5).clamp(0, 1)
    np_img = image.permute(1, 2, 0).numpy()
    plt.imsave(filename, np_img)

def get_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

# ================ FORWARD PROCESS ================
betas = get_beta_schedule(T).to(DEVICE)
alphas = 1 - betas
alpha_hat = torch.cumprod(alphas, dim=0)

def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
    return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise, noise

# ================ DATASET ================
class CarDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.lower().endswith(('png','jpg','jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

# ================ MODEL ===================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU())
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        x1 = self.enc1(x(x3 + x1))
        x2 = self.enc2(x1)
        x3 = self.dec1(x2)
        return self.out

# ================ EMA ===================
class EMA:
    def __init__(self, model, decay=0.995):
        self.ema_model = UNet().to(DEVICE)
        self.ema_model.load_state_dict(model.state_dict())
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema_model.state_dict().items():
                model_v = msd[k].detach()
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)

# ================ SAMPLING (REVERSE PROCESS) ================
@torch.no_grad()
def sample(model, img_size, steps):
    model.eval()
    x = torch.randn((1, 3, img_size, img_size)).to(DEVICE)
    for t in reversed(range(steps)):
        t_batch = torch.tensor([t], device=DEVICE)
        predicted_noise = model(x, t_batch)
        alpha = alphas[t]
        alpha_hat_t = alpha_hat[t]
        beta = betas[t]

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise) + torch.sqrt(beta) * noise
    return x

# ================ AMP Setup for MPS and CUDA ================
if DEVICE == "cuda":
    scaler = GradScaler()
    def autocast_context():
        return autocast(device_type="cuda")
elif DEVICE == "mps":
    scaler = GradScaler(enabled=False)
    def autocast_context():
        return autocast(device_type="mps", enabled=False)
else:
    scaler = GradScaler(enabled=False)
    def autocast_context():
        return autocast(enabled=False)

# ================ TRAINING ================
model = UNet().to(DEVICE)
ema = EMA(model)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

dataset = CarDataset("cars")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in pbar:
        batch = batch.to(DEVICE)
        batch = batch.contiguous(memory_format=torch.channels_last)

        t = torch.randint(0, T, (batch.size(0),), device=DEVICE).long()
        x_t, noise = forward_diffusion_sample(batch, t)

        with autocast_context():
            predicted_noise = model(x_t, t)
            loss = F.mse_loss(noise, predicted_noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema.update(model)

        pbar.set_description(f"Epoch {epoch} Step {step}: Loss = {loss.item():.4f}")

    # Save output image every 5 epochs
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        x_start = next(iter(dataloader)).to(DEVICE)
        t0 = torch.full((1,), T - 1, device=DEVICE, dtype=torch.long)
        x_t, _ = forward_diffusion_sample(x_start, t0)
        show_tensor_image(x_t[0], f"output_epoch_{epoch:03d}.png")

        sampled_img = sample(ema.ema_model, IMG_SIZE, T)
        show_tensor_image(sampled_img[0], f"generated_epoch_{epoch:03d}.png")

print("Training complete.")