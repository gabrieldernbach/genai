from dotmap import DotMap
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from einops.layers.torch import Rearrange
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'mps'   # or 'cuda'

def sinusoidal_embedding(x, dim):
    freqs = torch.logspace(0, torch.log10(torch.tensor(1_000.)), dim//2).to(x.device)
    angles = 2*torch.pi*freqs[:, None] * x[None, :]
    emb = torch.cat((angles.sin(), angles.cos()), 0)   # [dim, B]
    return emb.T                                       # [B, dim]

def ema(mu, dx): 
    return mu*0.999 + dx*0.001 if mu is not None else dx

def mlp(ins, hidden, outs):
    return nn.Sequential(nn.Linear(ins, hidden), nn.SiLU(), nn.Linear(hidden, outs))

class ConditionalResidualBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cond = mlp(cfg.cond_dim, cfg.latent_dim, 3*cfg.noise_dim)
        self.mlp = mlp(cfg.noise_dim, cfg.latent_dim, cfg.noise_dim)
        self.n_blocks = cfg.n_blocks
    
    def forward(self, x, cond):
        res = x
        x = F.layer_norm(x, [x.shape[-1]])
        scale1, shift, scale2 = self.cond(cond).chunk(3, dim=-1)
        x = self.mlp((1+scale1)*x + shift) * (1+scale2)
        return res + x/self.n_blocks


class ConditionalFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.noise_dim = cfg.noise_dim
        self.blocks = nn.ModuleList([ConditionalResidualBlock(cfg) for _ in range(cfg.n_blocks)])
        self.cls_emb = nn.Sequential(
            nn.Embedding(cfg.n_classes, cfg.latent_dim), nn.SiLU(),
            nn.Linear(cfg.latent_dim, cfg.cond_dim)
        )

    def forward(self, x, t, r, cls_idx):
        cls = self.cls_emb(cls_idx)
        t_emb = sinusoidal_embedding(t.squeeze(1), cls.size(-1))
        r_emb = sinusoidal_embedding(r.squeeze(1), cls.size(-1))
        cond  = cls + t_emb + r_emb
        for blk in self.blocks:
            x = blk(x, cond)
        return x

    def meanflow_loss(self, x0, cls_idx, flow_ratio=0.5, gamma=0.5, c=1e-3):
        B, D = x0.shape
        # sample (t, r) with r ≤ t and overwrite flow_ratio fraction with r=t
        t = torch.rand(B, device=x0.device)
        r = torch.rand(B, device=x0.device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        same_mask = torch.rand(B, device=x0.device) < flow_ratio
        r = torch.where(same_mask, t, r)

        e  = torch.randn_like(x0)                      # Gaussian end-point
        z  = (1-t)[:, None]*x0 + t[:, None]*e          # mixture point
        v  = e - x0

        def f(z_, t_): 
            return self.forward(z_, t_.unsqueeze(1), r.unsqueeze(1), cls_idx)
            
        u, dudt = torch.autograd.functional.jvp(
            f, (z, t), (v, torch.ones_like(t)), create_graph=True)

        u_tgt = v - (t-r)[:, None] * dudt
        err   = u - u_tgt.detach()

        delta_sq = err.pow(2).mean(1)
        w = 1 / (delta_sq + c).pow(1-gamma)
        loss = (w.detach() * delta_sq).mean()

        return loss, err.pow(2).mean()

    @torch.no_grad()
    def sample(self, cls_idx, n_steps=5):
        B = len(cls_idx)
        x = torch.randn(B, self.noise_dim, device=cls_idx.device)
        t_vals = torch.linspace(1., 0., n_steps+1, device=x.device)
        for i in range(n_steps):
            t = t_vals[i].expand(B, 1)
            r = t_vals[i+1].expand(B, 1)
            v = self.forward(x, t, r, cls_idx)
            x = x - (t-r) * v
        return x

batch_size     = 512
steps          = 8_000
warmup         = 100

dataset = torchvision.datasets.MNIST('.', download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5),
            Rearrange('1 h w -> (h w)')]))
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=batch_size,
    sampler=torch.utils.data.RandomSampler(
        dataset, 
        replacement=False,
        num_samples=batch_size*steps)
)

cfg = DotMap(
    noise_dim=28*28,
    cond_dim=512,
    latent_dim=1024,
    n_blocks=10,
    n_classes=10
)
model = ConditionalFlow(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

def ema(mu, dx): return mu*0.999 + dx*0.001 if mu is not None else dx
loss_ema = loss_ema if loss_ema else None
i = i if i else None
for i, (img, lbl) in enumerate(dataloader, start=i):
    img, lbl = img.to(device), lbl.to(device)
    loss, mse = model.meanflow_loss(img, lbl)

    loss.backward()
    opt.step()
    opt.zero_grad()

    loss_ema = ema(loss_ema, loss.item())
    if i % 50 == 0:
        print(f'{i=:05d}  loss={loss.item():.6f}  loss_ema={loss_ema:.6f}  mse={mse:.6f}')

smps = model.sample(lbl, n_steps=2)
fig, axs = plt.subplots(4,4, figsize=(6,6))
for ax, xhat, idx in zip(axs.flatten(), smps[:16], lbl[:16]):
    ax.imshow(xhat.view(28,28).cpu(), vmin=-1, vmax=1, cmap='gray')
    ax.set_title(idx.item()); ax.axis('off')
plt.tight_layout(); plt.show()
