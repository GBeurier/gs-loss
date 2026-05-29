"""Neural genomic predictors (PyTorch), parameterized by training loss.

The same architecture is trained under different losses so that *only the loss
changes* in the headline comparison. Targets are standardized with training
statistics inside :class:`NeuralRegressor`; predictions are returned in original
units. Early stopping monitors the validation value of the *same* loss being
trained (no peeking at the evaluation metric).

``batch_size=None`` trains full-batch, i.e. the loss sees all training
predictions at once -- the *global* Pearson objective. Smaller batches realize
the mini-batch (noisy) Pearson of Proposition 3.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from ..losses import get_loss
from ..utils import get_device


# --- architectures -----------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, p: int, hidden_dims=(256, 64), dropout=0.2):
        super().__init__()
        layers, d = [], p
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, p: int, channels=(16, 32), kernel_size=11, first_stride=3,
                 fc_dim=64, dropout=0.2):
        super().__init__()
        blocks, c_in = [], 1
        for i, c_out in enumerate(channels):
            stride = first_stride if i == 0 else 1
            blocks += [
                nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=kernel_size // 2),
                nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(c_in * 8, fc_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(fc_dim, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)            # (n, 1, p)
        return self.head(self.pool(self.conv(x))).squeeze(-1)


class TransformerLite(nn.Module):
    def __init__(self, p: int, n_tokens=64, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_tokens = n_tokens
        self.patch = int(np.ceil(p / n_tokens))
        self.pad = self.patch * n_tokens - p
        self.embed = nn.Linear(self.patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        if self.pad:
            x = torch.nn.functional.pad(x, (0, self.pad))
        x = x.view(x.shape[0], self.n_tokens, self.patch)
        h = self.encoder(self.embed(x) + self.pos)
        return self.head(h.mean(dim=1)).squeeze(-1)


_ARCHS = {"mlp": MLP, "cnn": CNN1D, "transformer": TransformerLite}


# --- sklearn-style wrapper ---------------------------------------------------

class NeuralRegressor:
    def __init__(self, arch="mlp", loss="mse", loss_kwargs=None, arch_kwargs=None,
                 lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=200,
                 patience=20, device=None, seed=0, use_amp=True, val_frac=0.2,
                 verbose=False):
        self.arch = arch
        self.loss = loss
        self.loss_kwargs = loss_kwargs or {}
        self.arch_kwargs = arch_kwargs or {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device or get_device()
        self.seed = seed
        self.use_amp = use_amp
        self.val_frac = val_frac
        self.verbose = verbose
        self.best_epoch_ = None
        self.train_time_ = None

    def _build(self, p):
        return _ARCHS[self.arch](p, **self.arch_kwargs).to(self.device)

    def _loss_on(self, model, X, y, loss_fn):
        model.eval()
        with torch.no_grad():
            return float(loss_fn(model(X), y).item())

    def fit(self, X, y, X_val=None, y_val=None):
        import time

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32).ravel()
        if X_val is None:
            idx = rng.permutation(len(X))
            n_val = max(2, int(self.val_frac * len(X)))
            vi, ti = idx[:n_val], idx[n_val:]
            X_val, y_val, X, y = X[vi], y[vi], X[ti], y[ti]

        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() + 1e-8)
        yz = (y - self.y_mean_) / self.y_std_
        yv = (np.asarray(y_val, np.float32).ravel() - self.y_mean_) / self.y_std_

        dev = self.device
        Xt = torch.as_tensor(X, device=dev)
        yt = torch.as_tensor(yz, device=dev)
        Xv = torch.as_tensor(np.asarray(X_val, np.float32), device=dev)
        yv = torch.as_tensor(yv, device=dev)

        model = self._build(X.shape[1])
        loss_fn = get_loss(self.loss, **self.loss_kwargs)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        amp = self.use_amp and dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        bs = self.batch_size or len(Xt)
        bs = min(bs, len(Xt))
        best_loss, best_state, best_ep, since = np.inf, None, -1, 0
        t0 = time.perf_counter()
        for ep in range(self.max_epochs):
            model.train()
            perm = torch.randperm(len(Xt), device=dev)
            for s in range(0, len(Xt), bs):
                b = perm[s:s + bs]
                if len(b) < 4:           # correlation needs a few samples
                    continue
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=amp):
                    loss = loss_fn(model(Xt[b]), yt[b])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            vloss = self._loss_on(model, Xv, yv, loss_fn)
            if vloss < best_loss - 1e-5:
                best_loss, best_state, best_ep, since = vloss, copy.deepcopy(model.state_dict()), ep, 0
            else:
                since += 1
                if since >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_, self.best_epoch_ = model, best_ep
        self.train_time_ = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def predict(self, X):
        self.model_.eval()
        X = torch.as_tensor(np.asarray(X, np.float32), device=self.device)
        out = []
        for s in range(0, len(X), 4096):
            out.append(self.model_(X[s:s + 4096]).float().cpu().numpy())
        return np.concatenate(out) * self.y_std_ + self.y_mean_


def make_neural(arch="mlp", loss="mse", **kwargs):
    return NeuralRegressor(arch=arch, loss=loss, **kwargs)


NEURAL_ARCHS = list(_ARCHS)
