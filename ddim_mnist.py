import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops.layers.torch import Rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt


def diffusion_schedule(time: torch.tensor):
    '''maps time in [0, 1] to noise-rate and signal-rate factors'''
    # cosine fade signal/noise mix from 95% to 2% signal
    start_angle = torch.tensor(0.95).arccos()
    end_angle = torch.tensor(0.02).arccos()
    diff = end_angle - start_angle
    
    # pythagorean identity ensures with preserve energy
    # sin(x) ** 2 + cos(x) ** 2 = 1 ** 2
    diffusion_angle = start_angle + time.mul(diff)
    signal_rate = diffusion_angle.cos()
    noise_rate = diffusion_angle.sin()
    return noise_rate, signal_rate

def time_embedding(time: torch.tensor, embedding_dim: int):
    '''positional encoding of time [0, 1]'''
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0, device=time.device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=time.device) * -emb)
    emb = emb * time.float()
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

def mlp(ins, hidden, outs):
    return nn.Sequential(
        nn.Linear(ins, hidden),
        nn.SiLU(),
        nn.Linear(hidden, outs),
    )

class ConditionedResidualMlp(nn.Module):
    '''see https://arxiv.org/pdf/2212.09748 for adaptive layer norm feature-wise modulation'''
    def __init__(self, noise_dimension, condition_dimension, latent_dimension, num_blocks):
        super().__init__()
        self.conditioning_layer = mlp(
            condition_dimension, 
            latent_dimension, 
            3*noise_dimension
        )
        self.mlp = mlp(
            noise_dimension, 
            latent_dimension, 
            noise_dimension
        )
        self.num_blocks = num_blocks
        
    def forward(self, x, condition):
        residual = x
        x = F.layer_norm(x, [x.shape[-1]])
        sss = self.conditioning_layer(condition)
        scale1, shift, scale2 = sss.chunk(3, dim=-1)
        x = self.mlp(scale1 * x + shift) * scale2
        return x/self.num_blocks + residual

class MlpStack(nn.Module):
    def __init__(self, noise_dimension, condition_dimension, num_blocks, latent_dimension):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConditionedResidualMlp(
                noise_dimension=noise_dimension, 
                condition_dimension=condition_dimension,
                latent_dimension=latent_dimension,
                num_blocks=num_blocks,
            ) 
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, time, z):
        cond = z + time_embedding(time, z.size(1))
        for block in self.blocks:
            x = block(x, cond)
        return x


def ema(mu, dx):
    '''exponential moving average'''
    return mu*0.99 + dx*0.01 if mu else dx

batch_size = 64
n_steps = 50_000
n_warmup = 500
n_cooldown = 500
device = 'mps'

ds = torchvision.datasets.MNIST(
    root=".", 
    download=True, 
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5),
        Rearrange("1 h w -> (h w)"), 
    ])
)
dl = torch.utils.data.DataLoader(
    dataset=ds, 
    batch_size=batch_size, 
    sampler=torch.utils.data.RandomSampler(
        data_source=ds, 
        replacement=False, 
        num_samples=batch_size*n_steps
    )
)
model = MlpStack(
    noise_dimension=28*28,
    condition_dimension=512,
    latent_dimension=1024,
    num_blocks=10,
).to(device)

cls_tokens = torch.randn(10, 512).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda x: torch.cat([
            torch.linspace(0, 1, n_warmup),
            torch.logspace(0, -2, n_steps-n_warmup-n_cooldown+1),
            torch.linspace(1, 0, n_cooldown),
        ])[x]
)

### TRAINING ###
loss_avg = None
for idx, (img, tar) in enumerate(dl):
    img = img.to(device)
    noise = torch.randn(img.shape, device=device)
    z = cls_tokens[tar]
    
    time = torch.rand(size=(len(img),1), device=device)
    noise_rate, signal_rate = diffusion_schedule(time)
    noised = signal_rate * img + noise_rate * noise
    pred = model(noised, time, z)
    loss = F.mse_loss(pred, noise)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    loss_avg = ema(loss_avg, loss.item())
    if idx % 50 == 0:
        print(f"{idx=:04d} {loss.item()=:.9f} {loss_avg=:.9f}")




### SAMPLING ###
num_images = 16
initial_noise = torch.randn((num_images, 28*28)).to(device)  # Start with Gaussian noise
diffusion_steps = 100
step_size = 1.0 / diffusion_steps

class_idxs = torch.randint(0, 10, size=(16, ))
z = cls_tokens[class_idxs]
noisy_images = initial_noise
with torch.no_grad():
    for time in tqdm(torch.linspace(1, 0, diffusion_steps, device=device)):
        time = time.expand(num_images, 1)
        noise_rate, signal_rate = diffusion_schedule(time)
        next_noise_rates, next_signal_rates = diffusion_schedule(time - step_size)
        
        pred_noise = model(noisy_images, time, z)
        pred_image = (noisy_images - noise_rate * pred_noise) / signal_rate
        noisy_images = (next_signal_rates * pred_image + next_noise_rates * pred_noise).clip(-1, 1)

fig, axs = plt.subplots(4, 4, figsize=(6, 6))
for ax, xt, idx in zip(axs.flatten(), noisy_images, class_idxs):
    ax.imshow(xt.reshape(28, 28).detach().cpu())
    ax.axis("off"); ax.set_title(idx.item())
fig.tight_layout()
