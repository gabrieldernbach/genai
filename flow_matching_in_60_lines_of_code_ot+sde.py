import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import einops
import matplotlib.pyplot as plt


class Flow(nn.Module):
    def __init__(self, n_dim=2, n_pos_dim=2, n_hidden=64):
        super().__init__()
        self.n_dim = n_dim
        self.n_pos_dim = n_pos_dim
        self.net = nn.Sequential(
            nn.Linear(n_dim + n_pos_dim, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_dim),
        )
        self.temb = nn.Linear(1, n_pos_dim // 2)

    def forward(self, t, x):
        t = self.temb(t).mul(torch.pi)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        return self.net(torch.cat((t, x), dim=-1))

    def loss(self, x):
        """
        x_t = (1 - t) * x + t * noise
            = α * x + σ * noise
        score(velocity):
            s(x,t) = sigma_t^-1 ( α v(x,t) - dα/dt x ) / (dα/dt * σ_t - α_t * dσ_t/dt )^2
        """
        time = torch.rand(len(x), 1)
        noise = torch.randn_like(x)
        noisedx = (1 - time) * x + (0.001 + 0.999 * time) * noise
        target = noise.mul(0.999).sub(x)  # from data to noise
        # print(target.shape, noisedx.shape)
        prediction = self.forward(time, noisedx)
        return (prediction - target).square().mean()

    def sample(self, n_samples, n_steps=100):
        x = torch.randn((n_samples, self.n_dim))
        dt = 1.0 / n_steps
        with torch.no_grad():  # runge-kutta-4 diffeq solver
            for t in tqdm(torch.linspace(1, 0, n_steps)):
                t = t.expand(len(x), 1)
                k1 = self.forward(t, x)
                k2 = self.forward(t - dt / 2, x - (dt * k1) / 2)
                k3 = self.forward(t - dt / 2, x - (dt * k2) / 2)
                k4 = self.forward(t - dt, x - dt * k3)
                x = x - (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def sample_sde(self, n_samples, n_steps=100):
        """
        x_t = (1 - t) * x + t * noise
            = α * x + σ * noise
        score(velocity):
            s(x,t) = σ_t^-1 ( α v(x,t) - dα/dt x ) / (dα/dt * σ_t - α_t * dσ_t/dt )^2
        """
        x = torch.randn((n_samples, self.n_dim))
        dt = 1.0 / n_steps

        alpha_t = lambda t: 1 - t
        dalpha_t = -1
        sigma_t = lambda t: 0.001 + 0.999 * t
        dsigma_t = 0.999

        with torch.no_grad():  # runge-kutta-4 diffeq solver
            for t in tqdm(torch.linspace(1, 0, n_steps)):
                t = t.expand(len(x), 1)
                v = self.forward(t, x)
                score = (
                    1
                    / (sigma_t(t) + 1e-6)
                    * (alpha_t(t) * v - dalpha_t * x)
                    / (dalpha_t * sigma_t(t) - alpha_t(t) * dsigma_t + 1e-6)
                )
                dx = self.forward(t, x) * dt
                dx += -1 / 2 * sigma_t(t) * score * dt
                dx += (sigma_t(t) * dt).pow(0.5) * torch.randn_like(x)
                x = x - dx
        return x


# data, _ = make_moons(16384, noise=0.05)


"""Generate Samples from a rudimentary GMM"""
means = torch.Tensor([[-1, 1], [-0.5, 1.5], [1.5, 0.25]])
stds = torch.Tensor([[0.25], [0.25], [0.15]]).repeat((1, 2))
means.shape, stds.shape
num_samples = 5_000
gmm = torch.distributions.Normal(loc=means, scale=stds)
data = einops.rearrange(gmm.sample((num_samples,)), "n d c -> (n c) d")


"""Visualize GMM Samples"""
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
for dim in range(data.shape[1]):
    axs[dim].hist(data[:, dim], density=True, bins=100)
    axs[dim].set_xlim(-2, 2)
plt.show()

# data = torch.from_numpy(data).float()
flow = Flow(n_dim=3)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

samples = flow.sample(5_000)
plt.figure(figsize=(4.8, 4.8), dpi=150)

fig, axs = plt.subplots(3, 2)
for dim in range(samples.shape[1]):
    axs[dim, 0].hist(data[:, dim], density=True, bins=100)
    axs[dim, 1].hist(samples[:, dim], density=True, bins=100)
    axs[dim, 0].set_xlim(-2, 2)
    axs[dim, 0].set_ylim(0, 1)
    axs[dim, 1].set_xlim(-2, 2)
    axs[dim, 1].set_ylim(0, 1)

axs[0, 0].set_title("Ground Truth Dist")
axs[0, 1].set_title("Generated Samples")
plt.show()

pbar = tqdm(range(1_000))
for epoch in pbar:
    optimizer.zero_grad()
    subset = torch.randint(0, len(data), (256,))
    x = data[subset]
    loss = flow.loss(x)
    loss.backward()
    optimizer.step()
    pbar.set_description(f"Loss: {loss.item():.4f}")


samples = flow.sample(5_000)
plt.figure(figsize=(4.8, 4.8), dpi=150)

fig, axs = plt.subplots(3, 2)
for dim in range(samples.shape[1]):
    axs[dim, 0].hist(data[:, dim], density=True, bins=100)
    axs[dim, 1].hist(samples[:, dim], density=True, bins=100)
    axs[dim, 0].set_xlim(-2, 2)
    axs[dim, 0].set_ylim(0, 1)
    axs[dim, 1].set_xlim(-2, 2)
    axs[dim, 1].set_ylim(0, 1)

axs[0, 0].set_title("Ground Truth Dist")
axs[0, 1].set_title("Generated Samples")
plt.show()

samples = flow.sample_sde(5_000)
plt.figure(figsize=(4.8, 4.8), dpi=150)

fig, axs = plt.subplots(3, 2)
for dim in range(samples.shape[1]):
    axs[dim, 0].hist(data[:, dim], density=True, bins=100)
    axs[dim, 1].hist(samples[:, dim], density=True, bins=100)
    axs[dim, 0].set_xlim(-2, 2)
    axs[dim, 0].set_ylim(0, 1)
    axs[dim, 1].set_xlim(-2, 2)
    axs[dim, 1].set_ylim(0, 1)

axs[0, 0].set_title("Ground Truth Dist")
axs[0, 1].set_title("Generated Samples")
plt.show()
