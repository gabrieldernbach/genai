# %%
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import einops
import matplotlib.pyplot as plt

import sys


def divergence(f, t, x):
    """
    Compute the analytical divergence of v(t, x) using PyTorch autograd.

    Args:
        v_func: Function v(t, x) that computes the velocity field.
        x: Input tensor of shape (batch_size, dim).
        t: Scalar tensor representing time.

    Returns:
        Analytical divergence of v(t, x) for each batch element.
    """
    batch_size, dim = x.shape
    divergence = torch.zeros(batch_size, device=x.device)

    with torch.enable_grad():
        for i in range(dim):
            x.requires_grad_(True)
            v_t = f(t, x)  # Compute velocity field v(t, x)
            grad_outputs = torch.ones_like(
                v_t[:, i]
            )  # Compute gradient wrt each output dimension
            div_v_i = torch.autograd.grad(
                v_t[:, i],
                x,
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
            )[0][:, i]
            # div_v_i = torch.autograd.grad(v_t[:, i], x, grad_outputs=grad_outputs)[0][:, i]
            divergence += div_v_i  # Sum over all diagonal elements
            x.detach()

    x.detach()

    return divergence


def stochastic_divergence(f, t, x, num_samples=10):
    """
    Compute the divergence of v(t, x) using Hutchinson's trace estimator.

    Args:
        v_func: Function v(t, x) that computes the velocity field.
        x: Input tensor of shape (batch_size, dim).
        t: Scalar tensor representing time.
        num_samples: Number of Hutchinson samples for estimation.

    Returns:
        Estimated divergence of v(t, x).
    """
    batch_size, dim = x.shape
    divergence_estimate = torch.zeros(batch_size, device=x.device)

    with torch.enable_grad():
        for _ in range(num_samples):
            v = torch.randn_like(x)  # Sample Gaussian noise v ~ N(0, I)
            x.requires_grad_(True)  # Enable gradient tracking

            v_t = f(t, x)  # Compute velocity field v(t, x), shape (batch_size, dim)
            div_v = torch.autograd.grad(v_t, x, grad_outputs=v, create_graph=True)[
                0
            ]  # Compute Jv

            divergence_estimate += torch.sum(div_v * v, dim=1)  # Estimate trace

    return divergence_estimate / num_samples  # Average over samples


class Flow(nn.Module):
    def __init__(self, n_dim=2, n_pos_dim=2, n_hidden=256):
        super().__init__()
        self.n_dim = n_dim
        self.n_pos_dim = n_pos_dim
        self.net = nn.Sequential(
            nn.Linear(n_dim + n_pos_dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_dim, bias=False),
        )
        self.temb = nn.Linear(1, n_pos_dim // 2)

    def forward(self, t, x):
        t = self.temb(t).mul(torch.pi)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        return self.net(torch.cat((t, x), dim=-1))

    def loss(self, x):
        """
        Vector field points from data to noise, data -> noise
        x_t = (1 - t) * x + t * noise
            = α * x + σ * noise
        v_t = -x + noise
        score(velocity):
            s(x,t) = sigma_t^-1 ( α v(x,t) - dα/dt x ) / (dα/dt * σ_t - α_t * dσ_t/dt )^2
        """
        time = torch.rand(len(x), 1)
        noise = torch.randn_like(x)
        x_t = (1 - time) * x + (0.001 + 0.999 * time) * noise
        target = noise.mul(0.999).sub(x)  # from data to noise
        prediction = self.forward(time, x_t)
        return (prediction - target).square().mean()

    def sample(
        self, n_samples, n_steps=200, traj=True, likelihood=True, base_noise=None
    ):
        if base_noise is None:
            x = torch.randn((n_samples, self.n_dim))
        else:
            x = base_noise
        dt = 1.0 / n_steps

        def rk4_fn(t, x):
            k1 = self.forward(t, x)
            k2 = self.forward(t - dt / 2, x - (dt * k1) / 2)
            k3 = self.forward(t - dt / 2, x - (dt * k2) / 2)
            k4 = self.forward(t - dt, x - dt * k3)
            dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            return dx

        def euler_fn(t, x):
            return self.forward(t, x)

        if traj:
            traj = [x]
        if likelihood:
            dim = x.shape[-1]
            likelihood = [
                torch.distributions.MultivariateNormal(
                    torch.zeros((1, dim)), torch.diag(torch.ones(dim))
                ).log_prob(x)
            ]
        with torch.no_grad():
            for step, t in tqdm(enumerate(torch.linspace(1, 0, n_steps))):
                t = t.expand(len(x), 1)
                dx = euler_fn(t, x)
                if likelihood:
                    likelihood += [likelihood[-1] + divergence(euler_fn, t, x) * dt]
                x = x - dx * dt  # integrating vector field backwards
                if traj and (step % (n_steps // 10) == 0):
                    traj.append(x)  # x - t * dx for x1 prediction

        return {
            "samples": x.detach(),
            "traj": torch.stack(traj).detach(),
            "likelihood": torch.stack(likelihood).detach().unsqueeze(-1),
        }

    def sample_diffusion(
        self, n_samples, n_steps=200, traj=True, likelihood=True, base_noise=None
    ):
        """
        x_0: Data
        x_t:
            x_t =  (1 - t) * x + t * noise
                = α * x + σ * noise
        x_1: Noise
        score(vectorfield v(x, t)):
            s(x,t) = σ_t^-1 ( α v(x,t) - dα/dt x ) / (dα/dt * σ_t - α_t * dσ_t/dt )^2
        """
        if base_noise is None:
            x = torch.randn((n_samples, self.n_dim))
        else:
            x = base_noise
        dt = 1.0 / n_steps

        """
        Flow Matching to Diffusion SDE Conversion
        FM:     x_t = (1 - t) * x + t * noise
        SDE:    dX_t = -f(t, X_t) dt + g(t, X_t) dW_t
        f = - ∂_t log(α_t) = = - dlog(α_t) = - 1 / (1 - t + 1e-6)
        g_t^2 = 2 (σ_t ∂_t[σ_t] - σ_t^2 dlog[α_t])
        """

        alpha_t = lambda t: 1 - t
        dalpha_t = lambda t: -1
        dlog_alpha_t = lambda t: -1 / (1 - t + 1e-6)
        sigma_t = lambda t: 0.001 + 0.999 * t
        dsigma_t = lambda t: 0.9999
        diff_t = lambda t: 1.0
        f_t = lambda t: dlog_alpha_t(t)  # f(t) already incorporates - sign
        g_t = lambda t: (
            2
            * (
                sigma_t(t) * dsigma_t(t)
                - sigma_t(t) ** 2 * dalpha_t(t) / (alpha_t(t) + 1e-6)
            )
            + 1e-6
        ).pow(0.5)

        if traj:
            traj = [x]
        if likelihood:
            dim = x.shape[-1]
            loglikelihood = [
                torch.distributions.MultivariateNormal(
                    torch.zeros((1, dim)), torch.diag(torch.ones(dim))
                ).log_prob(x)
            ]

        """Notation from https://arxiv.org/pdf/2411.01293"""
        with torch.no_grad():
            for step, t in tqdm(enumerate(torch.linspace(0.99, 0.001, n_steps))):
                t = t.expand(len(x), 1)
                v = self.forward(t, x)
                score = (
                    1
                    / (sigma_t(t) + 1e-6)
                    * (alpha_t(t) * v - dalpha_t(t) * x)
                    / (dalpha_t(t) * sigma_t(t) - alpha_t(t) * dsigma_t(t) + 1e-6)
                )
                score = score.clamp(-100, 100)
                # score = (
                #     -1 / sigma_t(t) * (x + (1 - t) * v)
                # )  # - 1/sigma E[\epsilon | x_t]
                drift = f_t(t) * x - 1 / 2 * g_t(t) ** 2 * (1 + diff_t(t) ** 2) * score
                brownian_motion = torch.randn_like(x)
                diffusion = diff_t(t) * g_t(t) * brownian_motion
                x = x + drift * (-dt) + diffusion * dt**0.5
                traj.append(x)  # x - t * dx for x1 prediction
                if likelihood:
                    # Cartoonist notation: arXiv:2411.01293v2
                    det_update = -f_t(t) - 1 / 2 * g_t(t) ** 2 * score**2
                    stoch_update = g_t(t) * score * brownian_motion
                    update = det_update * (-dt) + stoch_update * dt**0.5
                    loglikelihood += [loglikelihood[-1] + update.squeeze()]

        if likelihood:
            loglikelihood = torch.stack(loglikelihood).detach().unsqueeze(-1)
        return {
            "samples": x.detach(),
            "traj": torch.stack(traj).detach(),
            "likelihood": loglikelihood,
        }


class GaussianMixtureModel:
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds
        self.components = [
            torch.distributions.Normal(loc=mean, scale=std)
            for mean, std in zip(means, stds)
        ]
        self.component_probs = torch.ones(means.shape) / means.shape[1]

    def sample(self, num_samples):
        samples = []
        components = torch.multinomial(
            self.component_probs, num_samples, replacement=True
        ).T  # [num_samples, num_components]
        for component in components:
            single_sample = []
            for dim in range(self.means.shape[0]):
                sample_dim = torch.distributions.Normal(
                    loc=self.means[dim, component[dim]],
                    scale=self.stds[dim, component[dim]],
                ).sample()
                single_sample.append(sample_dim)
            sample = torch.stack(single_sample)
            samples.append(sample)

        return torch.stack(samples)

    def prob(self, x):
        probs = []
        for sample in x:
            prob = 0
            for dim in range(self.means.shape[0]):
                for component in range(2):
                    dist = torch.distributions.Normal(
                        loc=self.means[dim, component], scale=self.stds[dim, component]
                    )
                    prob += dist.log_prob(sample[dim]).exp()
            probs.append(prob / 2)
        return torch.stack(probs)


def plot_samples(data, samples, suptitle=""):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for dim in range(samples.shape[1]):
        axs[0].hist(data[:, dim], density=True, bins=samples.numel() // 10)
        axs[1].hist(samples[:, dim], density=True, bins=samples.numel() // 10)
        axs[0].set_xlim(-2, 2)
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(0, 1)

    axs[0].set_title("Ground Truth Dist")
    axs[1].set_title("Generated Samples")
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def plot_likelihood(data, samples, likelihood_, title=""):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    assert samples.ndim == 3, "Samples should be of shape [n_steps, n_samples, n_dim]"
    assert likelihood_.ndim == 3, (
        "Likelihood should be of shape [n_steps, n_samples, 1]"
    )

    for dim in range(samples_ot.shape[1]):
        axs[0].hist(
            data[:, dim], density=True, bins=data.shape[0] // 10
        )  # bins: roughly ten samples in every bin
        axs[1].scatter(
            samples[0, :, dim].numpy(),
            likelihood_[0, :, 0].exp().numpy(),
            s=5,
            c=torch.arange(likelihood_.shape[1]),
            cmap="viridis",
        )
        axs[2].scatter(
            samples[-1, :, dim].numpy(),
            likelihood_[-1, :, 0].exp().numpy(),
            s=5,
            c=torch.arange(likelihood_.shape[1]),
            cmap="viridis",
        )
        axs[0].set_xlim(-2, 2)
        axs[0].set_ylim(0, 1)
        axs[1].set_xlim(-2, 2)
        axs[1].set_ylim(0, 1)
        axs[2].set_xlim(-2, 2)
        axs[2].set_ylim(0, 2)

    axs[0].set_title("Ground Truth Dist")
    axs[1].set_title("Base Samples Likelihood")
    axs[2].set_title("Sample Likelihood")
    fig.suptitle(title)
    plt.show()


# %%
"""Generate Samples from a rudimentary GMM (1D for simple visualization of likelihoods)"""
means = torch.Tensor([[-1, 1]])
stds = torch.Tensor([[0.25, 0.25]])
num_samples = 1_000

gmm = GaussianMixtureModel(means, stds)
data = gmm.sample(num_samples)

"""Visualize GMM Samples"""
fig, axs = plt.subplots(1, 1, figsize=(9, 3))
for dim in range(data.shape[1]):
    axs.hist(data[:, dim], density=True, bins=data.numel() // 10)
    axs.set_xlim(-2, 2)

# %%
"""Train Optimal Transport Flow"""
flow = Flow(n_dim=data.shape[-1])
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)
pbar = tqdm(range(5000))
for epoch in pbar:
    optimizer.zero_grad()
    subset = torch.randint(0, len(data), (256,))
    x = data[subset]
    loss = flow.loss(x)
    loss.backward()
    optimizer.step()
    pbar.set_description(f"Loss: {loss.item():.4f}")

# %%
"""Sample OT samples and visualize likelihoods"""
sample_dict_ot = flow.sample(n_samples=1_000, n_steps=200, likelihood=True)
plot_samples(data, samples=sample_dict_ot["samples"], suptitle="Samples from OT Flow")

"""Evenly discretize base space and see where samples are transported in target space"""
base_noise = einops.repeat(torch.linspace(-3, 3, 100), "n -> n d", d=data.shape[-1])
sample_dict_ot = flow.sample(
    n_samples=1_000, n_steps=500, likelihood=True, base_noise=base_noise
)
samples_ot, trajs_ot, likelihood_ot = (
    sample_dict_ot["samples"],
    sample_dict_ot["traj"],
    sample_dict_ot["likelihood"],
)
plot_likelihood(
    data, trajs_ot, likelihood_ot, title="Likelihood of Probability Flow Samples"
)

# %%
"""Compare OT likelihood with Ground Truth GMM likelihood"""
fig = plt.figure()
plt.plot(samples_ot, likelihood_ot[-1, :, 0].exp(), label="Likelihood OT")
plt.plot(samples_ot, gmm.prob(samples_ot), label="Likelihood GMM")
plt.plot(
    torch.linspace(-2, 2, 100),
    gmm.prob(torch.linspace(-2, 2, 100).unsqueeze(-1)),
    label="Ground Truth",
)
plt.legend()
plt.show()

# %%
""" Sample Diffusion SDE converted from OT Flow """
samples_sde_dict = flow.sample_diffusion(n_samples=2_000, n_steps=1000, likelihood=True)
plot_samples(data, samples=samples_sde_dict["samples"], suptitle="Samples from SDE")

""" Discretize base space and see where samples are transported in target space or see where single base space point is diffused to"""
# evenly discretized base space
base_noise = einops.repeat(torch.linspace(-2, 2, 200), "n -> n d", d=data.shape[-1])
# single base space point diffused to different parts of target space
# base_noise = einops.repeat(torch.ones(200) * 0.0, "n -> n d", d=data.shape[-1])
samples_sde_dict = flow.sample_diffusion(
    n_samples=1_000, n_steps=10_000, likelihood=True, base_noise=base_noise
)
plot_samples(
    data,
    samples=samples_sde_dict["samples"],
    suptitle="Samples from SDE from single init value",
)
samples_sde, trajs_sde, likelihood_sde = (
    samples_sde_dict["samples"],
    samples_sde_dict["traj"],
    samples_sde_dict["likelihood"],
)

ground_truth = gmm.prob(samples_sde)
rel_error = (
    (ground_truth - likelihood_sde[-1].exp()).abs() / ground_truth.exp()
).mean()

"""Plot likelihoods of samples from SDE obtained with Ito density estimator"""
plot_likelihood(
    data,
    trajs_sde,
    likelihood_sde,
    title=f"Likelihood of Ito Density Samples (RelError: {rel_error:.3f})",
)
