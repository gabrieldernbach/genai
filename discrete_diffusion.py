"""
Proof of concept of a simple diffusion process that is discrete in space and time.
For applications think of amino-acid-sequences, or even simpler – natural language.

Discrete optimization is notoriously difficult, as naively theses spaces don't have a natural gradient to follow.
Often, the introduction of a rigorous framework can limit its adoption, and motivated this approach.

The following is inteded as a minimal proof of concept that shows how to translate flow-matching from 
continous time and space to discrete time and space. As in the other notebooks, we train a model
to predict the flow – the direction to follow at each point in time, to translate from base-space to sample space.

This approach has not been investigated formally and failure cases will need to be outlined.
Empirically we find that
* the number of time steps should be a multiple of the space that a sample is expected to travel when translating from base space to sample space.
* the model capacity (number of parameters) must be relatively large compared to the continous space. 
Intuitively, the continous formulation has a natural notion of distance (l2) that can be leveraged if both distributions have relatively low mainfold complexity.
* the relative entropy of the base distribution, compared to the entropy of the sample distribution can have an influence on the number
of time steps required for unfolding one into the other. We recommend using a high-entropy base-space such as the uniform distribution.
"""

import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from tqdm import tqdm
import matplotlib.pyplot as plt

class Flow(nn.Module):
    def __init__(self, n_dim=2, n_pos_dim=1024, n_hidden=1024):
        super().__init__()
        self.n_dim = n_dim
        self.n_pos_dim = n_pos_dim
        self.net = nn.Sequential(
            nn.Linear(n_dim + n_pos_dim, n_hidden), nn.GELU(approximate="tanh"),
            nn.Linear(n_hidden, n_hidden), nn.GELU(approximate="tanh"),
            nn.Linear(n_hidden, n_hidden), nn.GELU(approximate="tanh"),
            nn.Linear(n_hidden, n_hidden), nn.GELU(approximate="tanh"),
            nn.Linear(n_hidden, n_dim*3))
        self.temb = nn.Linear(1, n_pos_dim//2)

    def forward(self, t, x):
        batch, dim = x.shape
        t = self.temb(t).mul(torch.pi)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        x = self.net(torch.cat((t, x), dim=-1))
        return x.reshape(batch, dim, 3) # logits for actions {-1, 0, 1}

    def interp(self, a, b, factor):
        return (1-factor) * a + factor * b

    def loss(self, x):
        # sample from a high entropy distribution, low entropy base distribution
        # tend to  require many more steps to unfold into complex sample distributions of interest.
        noise = torch.distributions.Uniform(0,1).sample(x.shape).mul(256).long().to(x.device)
        # discretize time to n_steps = 1024, this number must be much larger than the 
        # space that must be traversed, as we typically get much less than n_step transitions in one direction
        time = torch.rand(len(x), 1, device=x.device).mul(1024).round().div(1024)
        # interpolate between sample-space (x) and base-space (noise), continuously then round
        noisedx = self.interp(x, noise, factor=time).round()
        # get the interpolation of one step ahead, indicating how to move into the future state (more noise)
        noisedx1p = self.interp(x, noise, factor=time.add(1/1024)).round()
        # predict the logits for probabilistic transition into future state
        logits = flow.forward(time, noisedx)
        # construct target (direction to move) as the diff between current and future state
        target = noisedx.sub(noisedx1p).add(1) # {-1, 0, 1} to [0, 1, 2] for CE loss
        ce = nn.CrossEntropyLoss()
        return ce(logits.reshape(-1, 3), target.flatten().long())


flow = Flow().to('mps')
optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-4, weight_decay=0.00001)

# sample from two-moons, load into torch
data, _ = make_moons(16384, noise=0.05)
data = torch.from_numpy(data).float()
# rescale dataset to [0, 1]
data = data.sub(data.min(0, keepdim=True).values)
data = data.div(data.max(0, keepdim=True).values)
# map continuous [0, 1] float to discrete [0, 255] = uint8
data = data.mul(255).round().long().to('mps')
base = torch.distributions.Uniform(0,1).sample(data.shape).mul(256).long()

plt.plot(*data.cpu().T, '.', markersize=0.3, label="sample distribution")
plt.plot(*base.T, ".", markersize=0.3, label="base distribution")
plt.legend()
plt.show()


ema = lambda mu, dx: mu*0.999 + dx*0.001 if mu else dx

loss_avg = None
for idx in range(16384*8): # 8 epochs
    subset = torch.randint(0, len(data), (256,))
    x = data[subset]
    loss = flow.loss(x)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_avg = ema(loss_avg, loss.item())
    if idx % 200 == 0:
        print(f"{idx=:05d}, {loss.item()=:.5f}, {loss_avg=:.7f}")


with torch.no_grad():
    n_steps, n_samples = 1024, 16384
    x = base.to('mps')
    dt = 1.0 / n_steps
    with torch.no_grad(): # runge-kutta-4 diffeq solver
        for t in tqdm(torch.linspace(1, 0, n_steps).to('mps')):
            t = t.expand(len(x), 1)
            logits = flow.forward(t, x)
            updates = torch.distributions.Categorical(logits=logits).sample().sub(1)
            x = (x + updates).clip(0, 256)

plt.plot(*x.cpu().T, '.', markersize=0.3, label="target dist")
plt.xlim(0, 256)
plt.ylim(0, 256)
plt.show()
