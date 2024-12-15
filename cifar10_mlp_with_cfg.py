from functools import partial
from time import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops.layers.torch import Rearrange
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm


torch.backends.cudnn.benchmark = True

def ema(mu, dx):
    '''exponential moving average'''
    return mu*0.99 + dx*0.01 if mu else dx

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

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(approximate="tanh"),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, n_seq, n_dim, depth, expansion_factor = 2, expansion_factor_token = 2, dropout = 0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
    return nn.ModuleList(
        [nn.Sequential(
            PreNormResidual(n_dim, FeedForward(n_seq, expansion_factor, dropout, chan_first)),
            PreNormResidual(n_dim, FeedForward(n_dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
    )

class DiffusionMLpMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            Rearrange("b c (h1 h2) (w1 w2) -> b (h1 w1) (h2 w2 c)", h1=32, w1=32),
            nn.Linear(1*1*3, 512),
        )
        self.unstem = nn.Sequential(
            nn.Linear(512, 1*1*3),
            Rearrange("b (h1 w1) (h2 w2 c) -> b c (h1 h2) (w1 w2)", h1=32, h2=1, w1=32, c=3)
        )

        self.backbone = MLPMixer(n_seq=1024+2, n_dim=512, depth=12, expansion_factor=2, expansion_factor_token=2)
        self.cls_emb = nn.Embedding(num_embeddings=200, embedding_dim=512)

    def forward(self, x, time, cls_idx):
        cls_emb = self.cls_emb(cls_idx.add(1)) ## allows for -1 as special cls_idx
        t_embedding = sinusoidal_embedding(time.squeeze(1), cls_emb.size(-1))
        x = self.stem(x)

        b, n, d = x.shape
        for module in self.backbone:
            x = torch.cat([cls_emb.unsqueeze(1), t_embedding.unsqueeze(1), x], dim=1)
            x = module(x)[:, 2:]
        x = self.unstem(x)
        return x

    def loss(self, x, cls_idx):
        '''see https://arxiv.org/pdf/2210.02747 equation (23)'''
        cls_idx = torch.where( # remove class condition to allow CFG
            torch.rand(len(cls_idx), device=device).ge(0.8), # on 20% of idxs
            torch.zeros(len(cls_idx), device=device).sub(1).long(), # place -1
            cls_idx # otherwise keep original
        )
        noise = torch.rand(x.shape, device=device).mul(2).sub(1)
        time = torch.randn(size=(len(x),1), device=device).sigmoid()
        noised = (1-time)[..., None, None] * x + (0.001 + 0.999*time)[..., None, None] * noise
        pred = self.forward(noised, time, cls_idx)
        return F.mse_loss(pred, noise.mul(0.999).sub(x))

    def sample(self, cls_idx, guidance=1, n_steps=100):
        x = torch.rand(len(cls_idx), 3, 32, 32, device=cls_idx.device).mul(2).sub(1)
        cls_unconditional = torch.zeros(len(cls_idx), device=cls_idx.device).sub(1).long()
        dt = 1.0 / n_steps
        with torch.no_grad():
            for t in tqdm(torch.linspace(1, 0, n_steps, device=x.device)):
                t = t.expand(len(x), 1)
                k1 = self.forward(x, t, cls_idx)
                k2 = self.forward(x, t, cls_unconditional)
                x = x - dt * (k1*guidance + (1-guidance)*k2)
                x = x.clip(-1, 1)
        return x

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

batch_size = 64 ## fits 16G RAM
tfm = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(0.5, 0.5),
])
ds = CIFAR10(root=".", transform=tfm, download=True)
dl = DataLoader(ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

n_epoch = 1_000
n_steps = n_epoch * len(dl)
n_warmup = 100 # helps adam to get 2nd order terms right
n_cooldown = 0 # actually hurts performance
device = 'cuda' 
model = DiffusionMLpMixer().to(device)
model_ema = EMA(model, decay=0.999)

# Initialize optimizer and learning rate schedule
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
schedule = torch.cat([
    torch.linspace(0, 1, n_warmup),
    torch.logspace(0, -2, n_steps - n_warmup + 1),
    torch.linspace(1, 0, n_cooldown),
])
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=schedule.__getitem__
)
scaler = GradScaler()
loss_avg = None

for epoch in range(1, 4000):
    for idx, (img, tar) in enumerate(dl):
        start = time()
        img, tar = img.to(device), tar.to(device)
        with autocast(device_type=device, dtype=torch.bfloat16):
            loss = model.loss(img, tar)
        scaler.scale(loss).backward()
        scaler.step(optimizer)       
        scaler.update()              
        optimizer.zero_grad()        
        scheduler.step()             
        model_ema.update()           
        loss_avg = ema(loss_avg, loss.item())
        
        stop = time()
        dur = stop - start
        if idx % 50 == 0:
            print(f"{epoch=:04d} {idx=:04d} {loss.item()=:.9f} {loss_avg=:.9f} {dur=:}")


model_ema.apply_shadow()


def sample(cidx):
    req_tar = torch.ones((16,), device=device).mul(cidx).long()
    smps = model.sample(req_tar, guidance=4, n_steps=100)
    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    for ax, xt, idx in zip(axs.flatten(), smps[:16], req_tar[:16]):
        ax.imshow(xt.permute(1,2,0).add(1).div(2).detach().cpu(), interpolation="nearest")
        ax.axis("off"); ax.set_title(idx.item())
    fig.tight_layout()

[sample(i) for i in range(10)]

# model_ema.restore()
