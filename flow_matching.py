import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt

def sinusoidal_embedding(x, dim):
    # x should be vector of [num_samples] in [0, 1]
    frequencies = torch.linspace(
        torch.log(torch.tensor(1.0)),
        torch.log(torch.tensor(1000.0)),
        dim // 2, device=device).exp()
    angular_speeds = frequencies.reshape(-1, 1).mul(torch.pi*2.)
    embeddings = torch.cat([
        angular_speeds.mul(x).sin(),
        angular_speeds.mul(x).cos(),
    ], axis=0)
    return embeddings.T

def mlp(ins, hidden, outs):
    return nn.Sequential(
        nn.Linear(ins, hidden),
        nn.SiLU(),
        nn.Linear(hidden, outs),
    )

class ConditionalResidualBlock(nn.Module):
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
        x = self.mlp((1+scale1) * x + shift) * (1+scale2)
        return x/self.num_blocks + residual


class ConditionalFlow(nn.Module):
    def __init__(self, 
                 noise_dimension,
                 condition_dimension, 
                 num_blocks, 
                 latent_dimension,
                 num_classes,
                ):
        super().__init__()
        self.noise_dimension = noise_dimension
        self.blocks = nn.ModuleList([
            ConditionalResidualBlock(
                noise_dimension=noise_dimension, 
                condition_dimension=condition_dimension,
                latent_dimension=latent_dimension,
                num_blocks=num_blocks,
            ) 
            for _ in range(num_blocks)
        ])
        self.cls_emb = nn.Sequential(
            nn.Embedding(num_classes, latent_dimension),
            nn.SiLU(),
            nn.Linear(latent_dimension, condition_dimension),
        )
        
    def forward(self, x, time, cls_idx):
        cls_emb = self.cls_emb(cls_idx)
        t_embedding = sinusoidal_embedding(time.squeeze(1), cls_emb.size(-1))
        for block in self.blocks:
            x = block(x, cls_emb + t_embedding)
        return x

    def loss(self, x, cls_idx):
        '''see https://arxiv.org/pdf/2210.02747 equation (23)'''
        noise = torch.randn(x.shape, device=device)    
        time = torch.randn(size=(len(x),1), device=device).sigmoid()
        noised = (1-time) * x + (0.001 + 0.999*time) * noise
        pred = self.forward(noised, time, cls_idx)
        return F.mse_loss(pred, noise.mul(0.999).sub(x))

    def sample(self, cls_idx, n_steps=100):
        # cls_idx should be a vector of shape [num_samples]
        x = torch.randn(len(cls_idx), self.noise_dimension, device=cls_idx.device)
        dt = 1.0 / n_steps
        with torch.no_grad(): # runge-kutta-4 solve diffeq
            for t in tqdm(torch.linspace(1, 0, n_steps, device=x.device)):
                t = t.expand(len(x), 1)
                k1 = self.forward(x, t, cls_idx)
                k2 = self.forward(x - (dt*k1)/2, t - dt/2, cls_idx)
                k3 = self.forward(x - (dt*k2)/2, t - dt/2, cls_idx)
                k4 = self.forward(x - dt*k3, t - dt, cls_idx)
                x = x - (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return x


def ema(mu, dx):
    '''exponential moving average'''
    return mu*0.99 + dx*0.01 if mu else dx

batch_size = 512
n_steps = 8_000
n_warmup = 100 # helps adam to get 2nd order terms right
n_cooldown = 0 # actually hurts performance
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
model = ConditionalFlow(
    noise_dimension=28*28,
    condition_dimension=512,
    latent_dimension=1024,
    num_blocks=10,
    num_classes=10,
).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
schedule = torch.cat([
    torch.linspace(0, 1, n_warmup),
    torch.logspace(0, -1, n_steps-n_warmup+1),
    torch.linspace(1, 0, n_cooldown),
])
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=schedule.__getitem__
)

loss_avg = None
for idx, (img, tar) in enumerate(dl):
    img, tar = img.to(device), tar.to(device)
    loss = model.loss(img, tar)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    loss_avg = ema(loss_avg, loss.item())
    if idx % 50 == 0:
        print(f"{idx=:04d} {loss.item()=:.9f} {loss_avg=:.9f}")


smps = model.sample(tar, 20)
fig, axs = plt.subplots(4, 4, figsize=(6, 6))
for ax, xt, idx in zip(axs.flatten(), smps[:16], tar[:16]):
    ax.imshow(xt.reshape(28, 28).detach().cpu(), vmin=-1, vmax=1)
    ax.axis("off"); ax.set_title(idx.item())
fig.tight_layout()
