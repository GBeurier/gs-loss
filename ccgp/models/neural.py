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


class DeepGS(nn.Module):
    """DeepGS-inspired network (Ma et al. 2018, *Planta*).

    It retains the published wide 1D convolution, max pool and sigmoid dense
    head (defaults: 8 filters, kernel 18, pool 4, dense 32), with BatchNorm added
    for stable optimization across heterogeneous panels. It is an adaptation,
    not a bitwise reproduction of the original MXNet implementation.
    """

    def __init__(self, p: int, n_filters=8, kernel_size=18, pool=4, fc_dim=32, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(1, n_filters, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(n_filters)              # BN-DeepGS variant: stabilizes the
        self.pool = nn.MaxPool1d(pool, stride=pool)      # sigmoid head so MSE training does not
        self.head = nn.Sequential(                       # collapse to a constant.
            nn.Flatten(), nn.Dropout(dropout), nn.LazyLinear(fc_dim), nn.BatchNorm1d(fc_dim),
            nn.Sigmoid(), nn.Dropout(dropout), nn.Linear(fc_dim, 1),
        )

    def forward(self, x):
        x = self.pool(torch.relu(self.bn(self.conv(x.unsqueeze(1)))))
        return self.head(x).squeeze(-1)


class DNNGP(nn.Module):
    """DNNGP-inspired network (Wang et al. 2023, *Molecular Plant*): a 1D-CNN over PCA-reduced
    features -> 3x Conv1d(kernel 4) -> BatchNorm -> Dropout -> Dense. The PCA front
    end is fit on training data only by :class:`NeuralRegressor` (``pca`` arg); this
    module sees the reduced input as a 1-channel signal. The common PyTorch
    wrapper and search space make this a protocol-matched adaptation.
    """

    def __init__(self, p: int, channels=(16, 16, 16), kernel_size=4, fc_dim=64, dropout=0.2):
        super().__init__()
        blocks, c_in = [], 1
        for c_out in channels:
            blocks += [nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2), nn.ReLU()]
            c_in = c_out
        self.conv = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm1d(c_in)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.LazyLinear(fc_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(fc_dim, 1),
        )

    def forward(self, x):
        return self.head(self.bn(self.conv(x.unsqueeze(1)))).squeeze(-1)


class _ParallelRes(nn.Module):
    """One PNNGS parallel-residual block: ``n_paths`` conv paths with kernels
    1,3,5,...,2*n_paths-1, concatenated on the channel axis, with a residual."""

    def __init__(self, c_in: int, c_out: int, n_paths: int = 4):
        super().__init__()
        self.paths = nn.ModuleList(
            [nn.Conv1d(c_in, c_out, 2 * n + 1, padding=n) for n in range(n_paths)])
        out = c_out * n_paths
        self.proj = nn.Conv1d(c_in, out, 1) if c_in != out else nn.Identity()

    def forward(self, x):
        cat = torch.cat([p(x) for p in self.paths], dim=1)
        return torch.relu(cat + self.proj(x))


class PNNGS(nn.Module):
    """PNNGS-inspired network (Xie et al. 2024, *Front. Plant Sci.*): a parallel
    residual 1D network (kernels 2n-1). We add a strided stem so 20k-marker panels
    stay tractable and adaptive pooling before the head; the defining parallel
    residual multi-kernel motif is preserved. It is therefore an adaptation. The
    paper claims a Pearson loss 'cannot be set as the loss function' -- here the
    same architecture is trained under MSE and the Pearson loss alike."""

    def __init__(self, p: int, n_paths=4, channels=8, stem_stride=8, dropout=0.5):
        super().__init__()
        self.stem = nn.Conv1d(1, 1, 7, stride=stem_stride, padding=3)
        self.b1 = _ParallelRes(1, channels, n_paths)
        self.bn = nn.BatchNorm1d(channels * n_paths)
        self.b2 = _ParallelRes(channels * n_paths, channels, n_paths)
        self.gap = nn.AdaptiveAvgPool1d(8)
        self.do = nn.Dropout(dropout)
        self.head = nn.LazyLinear(1)

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        x = self.bn(self.do(self.b1(x)))
        x = self.gap(self.do(self.b2(x)))
        return self.head(x.flatten(1)).squeeze(-1)


class _CoordAtt(nn.Module):
    """Coordinate attention (Hou et al. 2021), the attention module used by SoyDNGP."""

    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        mid = max(8, ch // reduction)
        self.conv1 = nn.Conv2d(ch, mid, 1)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid, ch, 1)
        self.conv_w = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        xh = x.mean(dim=3, keepdim=True)                 # (n,c,h,1)
        xw = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (n,c,w,1)
        y = self.act(self.bn(self.conv1(torch.cat([xh, xw], dim=2))))
        xh, xw = torch.split(y, [h, w], dim=2)
        ah = torch.sigmoid(self.conv_h(xh))
        aw = torch.sigmoid(self.conv_w(xw.permute(0, 1, 3, 2)))
        return x * ah * aw


class SoyDNGP(nn.Module):
    """SoyDNGP-inspired network (Gao et al. 2023, *Brief. Bioinform.*): markers reshaped to a square
    image, a deep 2D-CNN with coordinate attention at the stem and head. To keep
    the shared leakage-safe protocol ('only the architecture changes') the input is
    the same standardized dosage vector as every other model, reshaped (tiled/padded)
    to one ``side``x``side`` channel rather than SoyDNGP's discrete 3-channel
    206x206 one-hot tensor. This is explicitly an architecture-motif adaptation.
    """

    def __init__(self, p: int, side=96, width=32, dropout=0.3):
        super().__init__()
        self.side = side
        c = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(c),
            nn.Dropout(dropout), nn.ReLU())
        self.ca1 = _CoordAtt(c)
        chans = [c, c * 2, c * 4, c * 8, c * 8]          # 32->256
        blocks, c_in = [], c
        for c_out in chans:
            blocks += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.BatchNorm2d(c_out),
                       nn.Dropout(dropout), nn.ReLU()]
            c_in = c_out
        self.body = nn.Sequential(*blocks)
        self.ca2 = _CoordAtt(c_in)
        self.gap = nn.AdaptiveAvgPool2d(2)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.LazyLinear(1))

    def forward(self, x):
        n, p = x.shape
        side = self.side
        need = side * side
        if p < need:                                     # tile then crop to side*side
            reps = (need + p - 1) // p
            x = x.repeat(1, reps)[:, :need]
        else:
            x = x[:, :need]
        x = x.view(n, 1, side, side)
        x = self.ca1(self.stem(x))
        x = self.ca2(self.body(x))
        return self.head(self.gap(x)).squeeze(-1)


_ARCHS = {"mlp": MLP, "cnn": CNN1D, "transformer": TransformerLite,
          "deepgs": DeepGS, "dnngp": DNNGP, "pnngs": PNNGS, "soydngp": SoyDNGP}


# --- sklearn-style wrapper ---------------------------------------------------

class NeuralRegressor:
    def __init__(self, arch="mlp", loss="mse", loss_kwargs=None, arch_kwargs=None,
                 lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=200,
                 patience=20, device=None, seed=0, use_amp=True, val_frac=0.2,
                 verbose=False, pca=None):
        self.arch = arch
        self.loss = loss
        self.loss_kwargs = loss_kwargs or {}
        self.arch_kwargs = arch_kwargs or {}
        self.pca = pca                       # None, or PCA variance ratio / n_components (DNNGP)
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

    def _loss_on(self, model, X, y, loss_fn, weight=None):
        model.eval()
        with torch.no_grad():
            return float(loss_fn(model(X), y, weight=weight).item())

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, val_weight=None):
        """Fit a network, optionally using precision weights in train and validation.

        Weights are consumed by every registered loss. In particular, Pearson
        training then optimizes the same weighted correlation used by SelGenPalm.
        When an internal validation split is requested, weights are split with
        their corresponding observations.
        """
        import time

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32).ravel()
        sample_weight = (None if sample_weight is None else
                         np.asarray(sample_weight, np.float32).ravel())
        val_weight = None if val_weight is None else np.asarray(val_weight, np.float32).ravel()
        if sample_weight is not None and len(sample_weight) != len(y):
            raise ValueError("sample_weight must have one value per training observation")
        if self.pca is not None:             # DNNGP: PCA fit on training data only (leakage-safe)
            from sklearn.decomposition import PCA
            # The DNNGP paper reduces to a few hundred PCs (e.g. 251 for wheat599);
            # a randomized solver with a bounded component count keeps the per-fold
            # refit tractable on the wide 20k-marker panels.
            n_comp = self.pca if isinstance(self.pca, int) else min(256, len(X) - 1, X.shape[1])
            self.pca_ = PCA(n_components=n_comp, svd_solver="randomized",
                            random_state=self.seed).fit(X)
            X = self.pca_.transform(X).astype(np.float32)
            if X_val is not None:
                X_val = self.pca_.transform(np.asarray(X_val, np.float32)).astype(np.float32)
        if X_val is None:
            if val_weight is not None:
                raise ValueError("val_weight requires explicit validation data")
            idx = rng.permutation(len(X))
            n_val = max(2, int(self.val_frac * len(X)))
            vi, ti = idx[:n_val], idx[n_val:]
            X_val, y_val, X, y = X[vi], y[vi], X[ti], y[ti]
            if sample_weight is not None:
                val_weight, sample_weight = sample_weight[vi], sample_weight[ti]
        elif val_weight is not None and len(val_weight) != len(np.asarray(y_val).ravel()):
            raise ValueError("val_weight must have one value per validation observation")

        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() + 1e-8)
        yz = (y - self.y_mean_) / self.y_std_
        yv = (np.asarray(y_val, np.float32).ravel() - self.y_mean_) / self.y_std_

        dev = self.device
        Xt = torch.as_tensor(X, device=dev)
        yt = torch.as_tensor(yz, device=dev)
        Xv = torch.as_tensor(np.asarray(X_val, np.float32), device=dev)
        yv = torch.as_tensor(yv, device=dev)
        wt = None if sample_weight is None else torch.as_tensor(sample_weight, device=dev)
        wv = None if val_weight is None else torch.as_tensor(val_weight, device=dev)

        model = self._build(X.shape[1])
        # Materialize any LazyLinear params (DeepGS/DNNGP/PNNGS/SoyDNGP heads) with a
        # dummy forward *before* building the optimizer, so Adam sees every parameter.
        if any(isinstance(m, nn.modules.lazy.LazyModuleMixin) for m in model.modules()):
            model.train()
            with torch.no_grad():
                model(Xt[:min(8, len(Xt))])
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
                    loss = loss_fn(model(Xt[b]), yt[b], weight=None if wt is None else wt[b])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            vloss = self._loss_on(model, Xv, yv, loss_fn, wv)
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
        X = np.asarray(X, np.float32)
        if self.pca is not None:
            X = self.pca_.transform(X).astype(np.float32)
        X = torch.as_tensor(X, device=self.device)
        out = []
        for s in range(0, len(X), 4096):
            out.append(self.model_(X[s:s + 4096]).float().cpu().numpy())
        return np.concatenate(out) * self.y_std_ + self.y_mean_


def make_neural(arch="mlp", loss="mse", **kwargs):
    return NeuralRegressor(arch=arch, loss=loss, **kwargs)


NEURAL_ARCHS = list(_ARCHS)
