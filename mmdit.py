"""MMDit: multi‑modal diffusion transformer"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__(); self.scale = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        return x * (self.scale / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

def sinusoid(t: torch.Tensor, dim: int):
    half = dim // 2
    f = torch.exp(torch.linspace(0., math.log(1000), half, device=t.device))
    ang = 2 * math.pi * t[:, None] * f[None]
    return torch.cat([ang.sin(), ang.cos()], 1)

def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(approximate="tanh"), nn.Linear(h, o))

class MMDitBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        e, h = cfg.embed_dim, cfg.n_head
        self.hd, self.drop = e // h, cfg.dropout
        self.ln_qkv_n, self.ln_qkv_c = nn.LayerNorm(e), nn.LayerNorm(e)
        self.qkv_n, self.qkv_c = nn.Linear(e, e * 3, False), nn.Linear(e, e * 3, False)
        self.o_n, self.o_c = nn.Linear(e, e), nn.Linear(e, e)
        self.film_an, self.film_ac = nn.Linear(e, e * 2), nn.Linear(e, e * 2)
        self.film_fn, self.film_fc = nn.Linear(e, e * 2), nn.Linear(e, e * 2)
        Norm = RMSNorm if getattr(cfg, 'use_rms', False) else nn.Identity
        self.rms_q, self.rms_k = Norm(self.hd), Norm(self.hd)
        self.ln_ff_n, self.ln_ff_c = nn.LayerNorm(e), nn.LayerNorm(e)
        self.ff_n, self.ff_c = mlp(e, e * 4, e), mlp(e, e * 4, e)
        self.n_head, self.e = h, e

    def _heads(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.n_head, self.hd).transpose(1, 2)

    def forward(self, noise, cond, t_emb):
        b, sn, _ = noise.shape; sc = cond.shape[1]
        tn = repeat(t_emb, 'b c -> b s c', s=sn); tc = repeat(t_emb, 'b c -> b s c', s=sc)
        gn, bn = self.film_an(tn).chunk(2, -1); gc, bc = self.film_ac(tc).chunk(2, -1)
        n = self.ln_qkv_n(noise) * (1 + gn) + bn; c = self.ln_qkv_c(cond) * (1 + gc) + bc
        qn, kn, vn = self.qkv_n(n).chunk(3, -1); qc, kc, vc = self.qkv_c(c).chunk(3, -1)
        q = torch.cat([self._heads(qn), self._heads(qc)], 2)
        k = torch.cat([self._heads(kn), self._heads(kc)], 2)
        v = torch.cat([self._heads(vn), self._heads(vc)], 2)
        a = F.scaled_dot_product_attention(self.rms_q(q), self.rms_k(k), v, dropout_p=self.drop if self.training else 0)
        a = a.transpose(1, 2).reshape(b, sn + sc, self.e)
        an, ac = a.split([sn, sc], 1)
        noise = noise + self.o_n(an); cond = cond + self.o_c(ac)
        gn2, bn2 = self.film_fn(tn).chunk(2, -1); gc2, bc2 = self.film_fc(tc).chunk(2, -1)
        nf = self.ln_ff_n(noise) * (1 + gn2) + bn2; cf = self.ln_ff_c(cond) * (1 + gc2) + bc2
        noise = noise + self.ff_n(nf); cond = cond + self.ff_c(cf)
        return noise, cond

class MMDit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        e = cfg.embed_dim
        self.cfg, self.n_reg = cfg, getattr(cfg, 'n_reg', 1)
        self.max_noise_len = getattr(cfg, 'max_noise_len', 1024)
        
        self.noise_pos = nn.Parameter(
            torch.randn(1, self.n_reg + self.max_noise_len, e) / math.sqrt(e)
        )
        self.noise_in, self.noise_out = nn.Linear(cfg.noise_dim, e), nn.Linear(e, cfg.noise_dim)
        self.cond_in = nn.Linear(cfg.cond_dim, e)
        self.reg_tok = nn.Parameter(torch.randn(1, self.n_reg, cfg.noise_dim))
        self.time_mlp = mlp(e, e * 4, e)
        self.blocks = nn.ModuleList(MMDitBlock(cfg) for _ in range(cfg.n_block))

    def forward(self, noise, cond, t):
        assert noise.dim() == 3 and cond.dim() == 3, 'expect (B, seq, dim) tensors'
        b, sn = noise.size(0), noise.size(1)
        reg = repeat(self.reg_tok, '1 r d -> b r d', b=b)
        noise = torch.cat([reg, noise], 1)
        if sn > self.max_noise_len:
            raise ValueError(f'noise sequence ({sn}) exceeds max_noise_len={self.max_noise_len}')
        pos = self.noise_pos[:, : self.n_reg + sn]                 # match current length
        noise = self.noise_in(noise) + pos                         # project + position
        cond = self.cond_in(cond)
        t_emb = self.time_mlp(sinusoid(t, self.cfg.embed_dim))
        for blk in self.blocks: noise, cond = blk(noise, cond, t_emb)
        return self.noise_out(noise[:, self.n_reg:])


class ConditionalFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.noise_dimension = cfg.noise_dim
        self.backbone = MMDit(cfg)

    def loss(self, x, condition):
        '''see https://arxiv.org/pdf/2210.02747 equation (23)'''
        overwrite = torch.rand(len(condition), device=x.device).ge(0.8)
        condition = torch.where(
            repeat(overwrite, "b -> b s d", s=condition.shape[1], d=condition.shape[2]), # on 20% samples
            torch.zeros_like(condition), # set condition to 0
            condition # else keep the acutal observed
        )
        noise = torch.randn(x.shape, device=x.device)
        time = torch.randn(size=(len(x), 1, 1), device=x.device).sigmoid()
        noised = (1-time) * x + (0.001 + 0.999*time) * noise
        pred = self.backbone(noised, condition, time.squeeze())

        target = noise.mul(0.999).sub(x)
        mse_loss = F.mse_loss(pred, target)

        pred_flat = pred.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean(0)
        
        return mse_loss, cos_sim

    def sample(self, condition, guidance=1, n_steps=100, sigma_min=0.001):
        x = torch.randn(
            len(condition), 
            self.noise_dimension, 
            self.noise_dimension, 
            device=condition.device
        )
        condition0 = torch.zeros_like(condition)
        dt = 1.0 / n_steps
        with torch.no_grad():
            for t in tqdm(torch.linspace(1, 0, n_steps, device=x.device)):
                k1 = self.backbone(x, condition, t.expand(len(x)))
                k2 = self.backbone(x, condition0, t.expand(len(x)))
                drift = (k1*guidance + (1-guidance)*k2)
                
                if sigma_min == 0:
                    x = x - drift * dt
                else: # brownian kick
                    beta_t = sigma_min + (1.0 - sigma_min) * t
                    g = torch.sqrt(2.0 * (1.0 - sigma_min) * beta_t)
                    noise = torch.randn_like(x)
                    x = x - drift * dt + g * torch.sqrt(torch.tensor(dt)) * noise
        
        return x

    @torch.no_grad()
    def sampleRK4(self, condition, guidance: float = 1.0, n_steps: int = 100, sigma_min = 0.001):
        device, B = condition.device, len(condition)
        x  = torch.randn(B, self.noise_dimension, self.noise_dimension, device=device)
        condition0 = torch.zeros_like(condition)
        dt = 1.0 / n_steps

        def f(t, x_):
            t_ = t.expand(B)
            k_c = self.backbone(x_, condition, t_)
            k_u = self.backbone(x_, condition0, t_)
            return -(guidance * k_c + (1-guidance) * k_u)

        for t in tqdm(torch.linspace(1.0, 0.0, n_steps, device=device)):
            k1 = f(t, x)
            k2 = f(t - dt/2, x - dt*k1/2)
            k3 = f(t - dt/2, x - dt*k2/2)
            k4 = f(t - dt, x - dt*k3)
            drift = dt * (k1 + 2*k2 + 2*k3 + k4) / 6

            beta_t = sigma_min + (1.0 - sigma_min) * t
            g = torch.sqrt(2.0 * (1.0 - sigma_min) * beta_t)
            noise = torch.randn_like(x)
            brownian = g * torch.sqrt(torch.tensor(dt)) * noise
            
            x += drift + brownian

        return x

@dataclasses.dataclass
class FlowConfig:
    noise_dim=20
    cond_dim=20
    embed_dim=512
    n_head=8
    n_block=6
    dropout=0.1
    n_reg=8
