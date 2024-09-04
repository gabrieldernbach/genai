import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops.layers.torch import Rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt


def time_embedding(timestep, embedding_dim):
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0, device=timestep.device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timestep.device) * -emb)
    emb = emb * timestep.float()
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

def mlp(ins, hidden, outs):
    return nn.Sequential(
        nn.Linear(ins, hidden),
        nn.SiLU(),
        nn.Linear(hidden, outs),
    )

class ResidualBlock(nn.Module):
    '''see https://arxiv.org/pdf/2212.09748'''
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

class DenoisingMLP(nn.Module):
    def __init__(self, noise_dimension, condition_dimension, num_blocks, latent_dimension):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(
                noise_dimension=noise_dimension, 
                condition_dimension=condition_dimension,
                latent_dimension=latent_dimension,
                num_blocks=num_blocks,
            ) 
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, timestep, z):
        t_embedding = time_embedding(timestep, z.size(1))
        cond = z + t_embedding 
        for block in self.blocks:
            x = block(x, cond)
        return x

def ema(mu, dx):
    '''exponential moving average'''
    return mu*0.99 + dx*0.01 if mu else dx


batch_size = 512
n_steps = 8_000
n_warmup = 100
n_cooldown = 0 # not recommended 
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
model = DenoisingMLP(
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
            torch.logspace(0, -1, n_steps-n_warmup-n_cooldown+1),
            torch.linspace(1, 0, n_cooldown),
        ])[x]
)

### TRAIN ###
loss_avg = None
for idx, (img, tar) in enumerate(dl):
    img = img.to(device)
    noise = torch.randn(img.shape, device=device)
    cond = cls_tokens[tar]

    time = torch.randn(size=(len(img),1), device=device).sigmoid()
    noised = (1-time) * img + (0.001 + 0.999*time) * noise
    pred = model(noised, time, cond)
    loss = F.mse_loss(pred, noise.mul(0.999).sub(img))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    loss_avg = ema(loss_avg, loss.item())
    if idx % 50 == 0:
        print(f"{idx=:04d} {loss.item()=:.9f} {loss_avg=:.9f}")





### SAMPLE ###
num_images = 16
initial_noise = torch.randn((num_images, 28*28)).to(device)  # Start with Gaussian noise
diffusion_steps = 100
step_size = 1.0 / diffusion_steps

idxs = torch.randint(0, 10, size=(num_images, ))
cond = cls_tokens[idxs]
noisy_images = initial_noise

with torch.no_grad():
    for time in tqdm(torch.linspace(1, 0, diffusion_steps, device=device)):
        time = time.expand(num_images, 1)

        # runge-kutta-4
        k1 = model(noisy_images, time, cond)
        k2 = model(noisy_images - (step_size*k1)/2, time - step_size/2, cond)
        k3 = model(noisy_images - (step_size*k2)/2, time - step_size/2, cond)
        k4 = model(noisy_images - step_size*k3, time - step_size, cond)
        
        noisy_images = noisy_images - (step_size / 6) * (k1 + 2*k2 + 2*k3 + k4)
        noisy_images = noisy_images.clip(-1, 1) # images live in -1, 1


fig, axs = plt.subplots(4, 4, figsize=(6, 6))
for ax, xt, idx in zip(axs.flatten(), noisy_images, idxs):
    ax.imshow(xt.reshape(28, 28).detach().cpu(), vmin=-1, vmax=1)
    ax.axis("off"); ax.set_title(idx.item())
fig.tight_layout()
