# -*- coding: utf-8 -*-
"""
Minimal GDDS-MoE (for reading the idea, not a full bench).

Hybrid: CNN → Spatial MoE (per patch) → GAP → Global MoE → classifier

GDDS: experts that co-occur in top-k should have *different gradient directions*
      on the task:  L = Σ w_ij cos²(ĝ_i, ĝ_j) / Σ w_ij
      ĝ = g / ||g|| ,  w_ij = ||g_i|| ||g_j||  (magnitude-aware)

  L_total = CE + λ_lb * load_balance + λ_g * L_GDDS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# =============================================================================
# Config (edit here)
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
DATA_DIR = "./data"

NUM_CLASSES = 10
IN_CHANNELS = 1
IMAGE_SIZE = 32
DIM = 256
FEATURE_CH = 256
NUM_EXPERTS = 4
TOP_K = 2
EXPERT_HIDDEN = 384
EXPERT_DROPOUT = 0.1
ROUTER_NOISE_STD = 0.01
N_SPATIAL = 1          # MoE layers before GAP
N_GLOBAL = 1           # MoE layers after GAP
USE_POS_EMBED = True

USE_LOAD_BALANCE = True
LOAD_BALANCE_COEF = 1e-4

USE_GRAD_DIV = True
GRAD_DIV_COEF = 0.01
GRAD_DIV_SPATIAL_ONLY = True   # False → GDDS on all MoE layers
GRAD_DIV_MAX_TOKENS = 128
GRAD_DIV_WEIGHT_BY_MAG = True

EPOCHS = 80
BATCH_SIZE = 128
LR = 1.2e-3
WEIGHT_DECAY = 0.05


# =============================================================================
# MoE
# =============================================================================

class ExpertMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, EXPERT_HIDDEN),
            nn.GELU(),
            nn.Dropout(EXPERT_DROPOUT),
            nn.Linear(EXPERT_HIDDEN, DIM),
        )

    def forward(self, x):
        return self.net(x)


class SparseMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(DIM)
        self.router = nn.Linear(DIM, NUM_EXPERTS)
        self.experts = nn.ModuleList([ExpertMLP() for _ in range(NUM_EXPERTS)])

    def forward(self, x):
        h = self.norm(x)
        logits = self.router(h)
        if self.training and ROUTER_NOISE_STD > 0:
            logits = logits + torch.randn_like(logits) * ROUTER_NOISE_STD
        topk_val, topk_idx = logits.topk(TOP_K, dim=-1)
        combine = F.softmax(topk_val, dim=-1)

        # one forward per expert on its assigned tokens
        moe_out = torch.zeros_like(h)
        for e, expert in enumerate(self.experts):
            hit = topk_idx == e                          # [B, K]
            if not hit.any():
                continue
            w = (combine * hit.float()).sum(-1)          # [B]
            sel = w > 0
            moe_out[sel] = moe_out[sel] + w[sel, None] * expert(h[sel])

        y = x + moe_out
        return y, logits, topk_idx


def load_balance_loss(logits, topk_idx):
    p = F.softmax(logits, dim=-1).mean(0)
    load = torch.zeros(NUM_EXPERTS, device=logits.device)
    for k in range(TOP_K):
        load = load + F.one_hot(topk_idx[:, k], NUM_EXPERTS).float().sum(0)
    load = load / logits.shape[0]
    return (p * load).sum() * NUM_EXPERTS


# =============================================================================
# GDDS — the core idea
# =============================================================================

def _expert_grad(expert, z, upstream):
    """g = ∇_θ ⟨expert(z), u⟩  → unit direction ĝ and ||g||."""
    out = expert(z)
    scalar = (out * upstream).sum()
    params = [p for p in expert.parameters() if p.requires_grad]
    grads = torch.autograd.grad(scalar, params, create_graph=True, retain_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    mag = flat.norm().clamp_min(1e-8)
    return flat / mag, mag


def gdds_loss(layer, h, upstream, topk_idx):
    """Masked, magnitude-aware: only tokens where both experts are in top-k."""
    if h.shape[0] > GRAD_DIV_MAX_TOKENS:
        idx = torch.randperm(h.shape[0], device=h.device)[:GRAD_DIV_MAX_TOKENS]
        h, upstream, topk_idx = h[idx], upstream[idx], topk_idx[idx]

    z = layer.norm(h)
    upstream = upstream.detach()  # do not train router / backbone via GDDS

    num, den = z.new_tensor(0.0), z.new_tensor(0.0)
    for i in range(NUM_EXPERTS):
        active_i = (topk_idx == i).any(1)
        for j in range(i + 1, NUM_EXPERTS):
            both = active_i & (topk_idx == j).any(1)
            if not both.any():
                continue
            gi, mi = _expert_grad(layer.experts[i], z[both], upstream[both])
            gj, mj = _expert_grad(layer.experts[j], z[both], upstream[both])
            cos = (gi * gj).sum()
            w = mi * mj if GRAD_DIV_WEIGHT_BY_MAG else 1.0
            num = num + (cos * cos) * w
            den = den + w
    if den.item() == 0:
        return z.new_tensor(0.0)
    return num / den.detach().clamp_min(1e-12)


# =============================================================================
# Backbone: Spatial MoE → GAP → Global MoE
# =============================================================================

class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.GELU()
        self.skip = nn.Identity()
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.skip(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            Block(64, 64),
            Block(64, 64),
            Block(64, 128, 2),
            Block(128, 128),
            Block(128, 256, 2),
            Block(256, 256),
        )
        # 32→8 after two stride-2 blocks → 64 spatial tokens
        self.embed = nn.Linear(FEATURE_CH, DIM)
        self.pos = nn.Parameter(torch.zeros(1, 64, DIM)) if USE_POS_EMBED else None
        self.spatial = nn.ModuleList([SparseMoE() for _ in range(N_SPATIAL)])
        self.global_ = nn.ModuleList([SparseMoE() for _ in range(N_GLOBAL)])
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, NUM_CLASSES)

    def forward(self, x, trace=False):
        feat = self.cnn(x)                                 # [B, 256, 8, 8]
        b, c, hh, ww = feat.shape
        tok = feat.permute(0, 2, 3, 1).reshape(b, hh * ww, c)
        tok = self.embed(tok)
        if self.pos is not None:
            tok = tok + self.pos
        h = tok.reshape(b * hh * ww, DIM)

        inputs, outputs, topks = [], [], []
        aux = h.new_tensor(0.0)

        def run_layers(layers, h):
            nonlocal aux
            for layer in layers:
                if trace:
                    inputs.append(h)
                h, logits, tidx = layer(h)
                if USE_LOAD_BALANCE:
                    aux = aux + LOAD_BALANCE_COEF * load_balance_loss(logits, tidx)
                if trace:
                    outputs.append(h)
                    topks.append(tidx)
            return h

        h = run_layers(self.spatial, h)
        h = h.reshape(b, hh * ww, DIM).mean(1)             # GAP
        h = run_layers(self.global_, h)
        logits = self.head(self.norm(h))
        if trace:
            return logits, aux, inputs, outputs, topks
        return logits, aux


def gdds_on_model(model, inputs, outputs, topks, task_loss):
    """GDDS on spatial layers only if GRAD_DIV_SPATIAL_ONLY else all layers."""
    layers = list(model.spatial) + list(model.global_)
    n_sp = len(model.spatial)
    total = task_loss.new_tensor(0.0)
    for li, layer in enumerate(layers):
        if GRAD_DIV_SPATIAL_ONLY and li >= n_sp:
            continue
        u = torch.autograd.grad(
            task_loss, outputs[li], retain_graph=True, create_graph=True
        )[0]
        total = total + gdds_loss(layer, inputs[li], u, topks[li])
    return total


# =============================================================================
# Train
# =============================================================================

def main():
    torch.manual_seed(SEED)
    mean, std = (0.1307,), (0.3081,)
    train_tf = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.RandomCrop(IMAGE_SIZE, padding=2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.Resize(IMAGE_SIZE), T.ToTensor(), T.Normalize(mean, std)])
    train_ld = DataLoader(
        torchvision.datasets.MNIST(DATA_DIR, True, download=True, transform=train_tf),
        BATCH_SIZE, shuffle=True,
    )
    test_ld = DataLoader(
        torchvision.datasets.MNIST(DATA_DIR, False, download=True, transform=test_tf),
        BATCH_SIZE, shuffle=False,
    )

    model = Model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for ep in range(1, EPOCHS + 1):
        model.train()
        n, ce_sum, acc_sum = 0, 0.0, 0.0
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, aux, inp, out, topks = model(x, trace=True)
            ce = F.cross_entropy(logits, y)
            task = ce + aux
            gdiv = gdds_on_model(model, inp, out, topks, task) if USE_GRAD_DIV else 0.0
            (task + GRAD_DIV_COEF * gdiv).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = y.numel()
            n += bs
            ce_sum += ce.item() * bs
            acc_sum += (logits.argmax(1) == y).float().sum().item()

        model.eval()
        tn, tacc = 0, 0.0
        with torch.no_grad():
            for x, y in test_ld:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x)
                tn += y.numel()
                tacc += (logits.argmax(1) == y).float().sum().item()
        print(
            f"ep {ep}/{EPOCHS}  train_ce={ce_sum/n:.4f}  "
            f"tr@1={acc_sum/n:.4f}  te@1={tacc/tn:.4f}"
        )


if __name__ == "__main__":
    main()
