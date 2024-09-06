from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from einops import rearrange
from tqdm import tqdm

def tokenized_mnist():
  '''simple patchify mnist, discretize patches into tokens using kmeans'''
  ds = torchvision.datasets.MNIST(root=".", download=True)
  arr = ds.data.div(255).sub(.5).mul(2)
  n_samp, height, width = arr.shape
  patch_size = 4
  n_token = (height // patch_size)**2
  
  fmt = "b (h1 h2) (w1 w2) -> (b h1 w1) (h2 w2)"
  patch = rearrange(arr, fmt, h2=patch_size, w2=patch_size)
  tokenizer = MiniBatchKMeans(n_clusters=512).fit(patch)
  alltokens = tokenizer.predict(patch).reshape(n_samp, n_token).reshape(n_samp, 49)
  return tokens, tokenizer

class MaskGit(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, n_token, 512))
        self.tok_emb = nn.Embedding(
            num_embeddings=512+1, # +1 for mask_token 
            embedding_dim=512,
        )
        self.mask_idx = 512
        self.cls_emb = nn.Embedding(
            num_embeddings=10,
            embedding_dim=512,
        )
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512, 
                nhead=8, 
                dim_feedforward=512,
                batch_first=True, 
                norm_first=True, 
                bias=False, 
            ), 
            num_layers=4,
            enable_nested_tensor=False, 
            mask_check=False,
        )
        self.predictor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def token_mask(self, token):
        device = token.device
        per_row_threshold = torch.rand(len(token), 1, device=device)
        mask = torch.rand(token.shape, device=device).ge(per_row_threshold)
        
        source = torch.where(
            # where mask is true, take mask_idx (token), else take token
            condition=mask, 
            input=torch.ones_like(token, device=device).mul(self.mask_idx),
            other=token,
        )

        destination = torch.where(
            # where mask is true, take token, else take -1 to skip loss computation
            condition=mask,
            input=token,
            other=torch.ones_like(token).mul(-1),
        )
        return source, destination

    def forward(self, token, cls_ids):
        source, destination = self.token_mask(token)
        inputs = torch.cat([
            self.cls_emb(cls_ids.unsqueeze(-1)),
            self.tok_emb(source).add(self.pos_emb)
        ], dim=1)

        logits = self.predictor(self.backbone(inputs))[:, 1:, :] # drop cls token
        loss = nn.CrossEntropyLoss(ignore_index=-1)(
            logits.reshape(-1, 512),
            destination.flatten(),
        )
        return loss

alltokens, tokenizer = tokenized_mnist()
# example reversing the tokenization
tmp = tokenizer.cluster_centers_[alltokens[2]]
plt.imshow(rearrange(tmp, "(h1 w1) (h2 w2) -> (h1 h2) (w1 w2)", h1=7, h2=4, w2=4))

dl = DataLoader(
    TensorDataset(torch.tensor(alltokens).long(), ds.targets.long()),
    batch_size=128,
    shuffle=True,
)
device = "mps"
model = MaskGit().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)

ema = lambda mu, dx: mu*0.99 + dx*0.01 if mu else dx
loss_avg = None

for epoch in range(100):
    for idx, (tokens, targets) in enumerate(dl):    
        loss = model(
            tokens.to(device), 
            targets.to(device),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_avg = ema(loss_avg, loss.item())
        if idx % 50 == 0:
            print(f"{idx=:04d} {loss.item()=:.09f} {loss_avg=:.09f}")


model.eval()
with torch.no_grad():
    num_samples = 16
    n_token = 49
    
    cls_ids = torch.randint(0, 10, (num_samples, 1))
    state = torch.ones(num_samples, n_token, dtype=torch.long).mul(model.mask_idx)
    
    shuff = torch.randn(num_samples, n_token).argsort(-1)
    blocks = torch.eye(7).kron(torch.ones(num_samples, 7)).chunk(7, dim=0)
    policy = torch.gather(
        torch.stack(blocks),
        dim=-1,
        index=shuff.repeat(7, 1, 1)
    ).bool()
    
    states = []
    for pol in tqdm(policy):
        inputs = torch.cat([
            model.cls_emb(cls_ids.to('mps')),
            model.tok_emb(state.to('mps')) + model.pos_emb,
        ], axis=1)
        logits = model.predictor(model.backbone(inputs.to('mps')))[:, 1:] # remove cls label

        logits = logits / 1 # temperature
        cutoff = torch.topk(logits, 16, dim=-1).values[..., -1]
        logits[logits < cutoff[..., None]] = -float('Inf')
        
        smp_onehot = torch.distributions.Multinomial(logits=logits).sample()
        proposal = smp_onehot.argmax(-1)
        state = torch.where(
            condition=pol,
            input=proposal.cpu(), 
            other=state,
        )
        states.append(state)

res = states[-1].numpy()
idx = 7
plt.imshow(rearrange(
    tokenizer.cluster_centers_[res[idx]],
    pattern="(h1 w1) (h2 w2) -> (h1 h2) (w1 w2)",
    h1=7, h2=4, w2=4,
))
plt.title(cls_ids[idx])
