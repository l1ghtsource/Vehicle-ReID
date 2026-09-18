import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _rsqrt_uniform_(tensor, dim):
    bound = float(dim) ** -0.5
    nn.init.uniform_(tensor, -bound, bound)
    return tensor


def _random_signs_(tensor):
    with torch.no_grad():
        tensor.bernoulli_(0.5).mul_(2.0).add_(-1.0)
    return tensor


class EnsembleLinear(nn.Module):
    def __init__(self, in_features, out_features, k, scaling_init="ones"):
        super().__init__()
        if k < 1 or in_features < 1 or out_features < 1:
            raise ValueError("Invalid ensemble linear dimensions")
        if scaling_init not in {"ones", "random-signs"}:
            raise ValueError("Unknown scaling init")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.k = int(k)
        self.scaling_init = scaling_init
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.r = nn.Parameter(torch.empty(self.k, self.in_features))
        self.s = nn.Parameter(torch.empty(self.k, self.out_features))
        self.bias = nn.Parameter(torch.empty(self.k, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        _rsqrt_uniform_(self.weight, self.in_features)
        init_fn = nn.init.ones_ if self.scaling_init == "ones" else _random_signs_
        init_fn(self.r)
        init_fn(self.s)
        shared = torch.empty(self.out_features, dtype=self.weight.dtype)
        _rsqrt_uniform_(shared, self.in_features)
        with torch.no_grad():
            self.bias.copy_(shared)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.k or x.shape[2] != self.in_features:
            raise ValueError("Expected ensemble input (B, k, in_features)")
        return (x * self.r) @ self.weight.T * self.s + self.bias


class RefusalTabM(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), k=8, dropout=0.1):
        super().__init__()
        if in_dim < 1 or len(hidden) < 1 or k < 1 or dropout < 0 or dropout >= 1:
            raise ValueError("Invalid TabM dimensions")
        widths = [int(in_dim), *[int(width) for width in hidden]]
        blocks = []
        for i in range(len(widths) - 1):
            src, dst = widths[i], widths[i + 1]
            init = "random-signs" if i == 0 else "ones"
            layer = EnsembleLinear(src, dst, k=int(k), scaling_init=init)
            if i == 0:
                nn.init.ones_(layer.s)
            blocks.extend([layer, nn.ReLU(), nn.Dropout(float(dropout))])
        self.blocks = nn.Sequential(*blocks)
        self.head = EnsembleLinear(widths[-1], 1, k=int(k), scaling_init="ones")
        self.k = int(k)
        self.in_dim = int(in_dim)
        self.feat_mean = np.zeros(int(in_dim), dtype=np.float32)
        self.feat_scale = np.ones(int(in_dim), dtype=np.float32)

    def set_norm(self, mean, scale):
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        if mean.shape != self.feat_mean.shape or scale.shape != self.feat_scale.shape:
            raise ValueError("Normalization stats must match the input dimension")
        self.feat_mean = mean.copy()
        self.feat_scale = np.maximum(scale, 1e-6).copy()

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError("Expected a 2D feature matrix")
        mean = torch.as_tensor(self.feat_mean, dtype=x.dtype, device=x.device)
        scale = torch.as_tensor(self.feat_scale, dtype=x.dtype, device=x.device)
        x = (x - mean) / scale
        x = x.unsqueeze(1).expand(-1, self.k, -1)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


def fit_tabm(
    X,
    y,
    epochs=40,
    lr=1e-3,
    seed=0,
    val_frac=0.2,
    patience=8,
    batch_size=64,
    weight_decay=1e-4,
    hidden=(64, 32),
    dropout=0.1,
    k=8,
):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or len(X) < 4 or len(np.unique(y)) < 2:
        raise ValueError("TabM needs a 2D feature matrix with both classes")
    if epochs < 1 or batch_size < 1 or lr <= 0 or k < 1:
        raise ValueError("Invalid TabM training hyperparameters")
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))
    order = rng.permutation(len(X))
    n_val = int(len(X) * float(val_frac)) if 0 < val_frac < 1 else 0
    val_idx = order[:n_val]
    train_idx = order[n_val:] if n_val else order
    if len(train_idx) < 2:
        train_idx = order
        val_idx = np.array([], dtype=int)
    model = RefusalTabM(X.shape[1], hidden=hidden, k=k, dropout=dropout)
    mean = X[train_idx].mean(0)
    scale = np.maximum(X[train_idx].std(0), 1e-6)
    model.set_norm(mean, scale)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    xt = torch.as_tensor(X)
    yt = torch.as_tensor(y)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    stall = 0
    for _ in range(int(epochs)):
        model.train()
        perm = rng.permutation(train_idx)
        for start in range(0, len(perm), int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            opt.zero_grad(set_to_none=True)
            logits = model(xt[idx])
            target = yt[idx][:, None].expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            probe = val_idx if len(val_idx) else train_idx
            current = float(
                F.binary_cross_entropy_with_logits(model(xt[probe]), yt[probe][:, None].expand(-1, model.k))
            )
        if current + 1e-6 < best_loss:
            best_loss = current
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= int(patience):
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_tabm(model, X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")
    model.eval()
    with torch.no_grad():
        param = next(model.parameters())
        logits = model(torch.as_tensor(X, device=param.device, dtype=param.dtype))
        return torch.sigmoid(logits).mean(1).cpu().numpy().astype(np.float64)
