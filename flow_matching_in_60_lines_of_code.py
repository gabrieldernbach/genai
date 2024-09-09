import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from tqdm import tqdm
import matplotlib.pyplot as plt

class Flow(nn.Module):
    def __init__(self, n_dim=2, n_pos_dim=2, n_hidden=64):
        super().__init__()
        self.n_dim = n_dim
        self.n_pos_dim = n_pos_dim
        self.net = nn.Sequential(
            nn.Linear(n_dim + n_pos_dim, n_hidden), nn.ELU(),
            nn.Linear(n_hidden, n_hidden), nn.ELU(),
            nn.Linear(n_hidden, n_hidden), nn.ELU(),
            nn.Linear(n_hidden, n_dim))
        self.temb = nn.Linear(1, n_pos_dim//2)

    def forward(self, t, x):
        t = self.temb(t).mul(torch.pi)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        return self.net(torch.cat((t, x), dim=-1))

    def loss(self, x):
        time = torch.rand(len(x), 1)
        noise = torch.randn_like(x)
        noisedx = (1 - time) * x + (0.001 + 0.999 * time) * noise
        target = noise.mul(0.999).sub(x)
        prediction = self.forward(time, noisedx)
        return (prediction - target).square().mean()

    def sample(self, n_samples, n_steps=100):
        x = torch.randn((n_samples, self.n_dim))
        dt = 1.0 / n_steps
        with torch.no_grad(): # runge-kutta-4 diffeq solver
            for t in tqdm(torch.linspace(1, 0, n_steps)):
                t = t.expand(len(x), 1)
                k1 = self.forward(t, x)
                k2 = self.forward(t - dt/2, x - (dt*k1)/2)
                k3 = self.forward(t - dt/2, x - (dt*k2)/2)
                k4 = self.forward(t - dt, x - dt*k3)
                x = x - (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return x

data, _ = make_moons(16384, noise=0.05)
data = torch.from_numpy(data).float()
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

for epoch in tqdm(range(16384)):
    subset = torch.randint(0, len(data), (256,))
    x = data[subset]
    flow.loss(x).backward()
    optimizer.step()
    optimizer.zero_grad()
    
xhat = flow.sample(16384)
plt.figure(figsize=(4.8, 4.8), dpi=150)
plt.hist2d(*xhat.T, bins=128)
