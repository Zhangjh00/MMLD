"""English documentation."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def remap_legacy_glv_state_dict(state_dict):
    """English documentation."""
    remapped = dict(state_dict)
    key_pairs = {
        "r": "intrinsic_growth_rate",
        "_alpha_raw": "self_interaction_raw",
    }
    for old_key, new_key in key_pairs.items():
        if old_key in remapped and new_key not in remapped:
            remapped[new_key] = remapped.pop(old_key)
    return remapped


def load_glv_state_dict(model, state_dict, strict=True):
    """English documentation."""
    return model.load_state_dict(remap_legacy_glv_state_dict(state_dict), strict=strict)


def build_glv_model(model_type: str, N: int, lag_min: int, lag_max: int, edge_info: list):
    """English documentation."""
    kind = str(model_type or "euler").strip().lower()
    if kind in ("euler", "eulerian"):
        return gLVModel(N, lag_min, lag_max, edge_info)
    if kind in ("rk", "rk4", "runge-kutta", "runge_kutta"):
        return gLVModel_RK(N, lag_min, lag_max, edge_info)
    raise ValueError("Unknown gLV solver: %s (supported: euler or rk4)" % model_type)


def load_glv_checkpoint(checkpoint, device="cpu"):
    """English documentation."""
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("The model checkpoint contains neither state_dict nor model_state_dict")
    model_type = checkpoint.get("model_type") or checkpoint.get("solver") or "euler"
    model = build_glv_model(model_type, checkpoint["N"], checkpoint["lag_min"], checkpoint["lag_max"], checkpoint["edge_info"])
    load_glv_state_dict(model, state_dict)
    return model.to(device)


class gLVModel(nn.Module):
    """English documentation."""

    def __init__(self, N: int, lag_min: int, lag_max: int, edge_info: list):
        super().__init__()
        self.N = N
        self.lag_min = lag_min
        self.lag_max = lag_max
        self.num_lags = lag_max - lag_min + 1

        self.intrinsic_growth_rate = nn.Parameter(torch.zeros(N))
        self.self_interaction_raw = nn.Parameter(torch.ones(N) * 1.0)

        sources, targets, lag_idx, signs = [], [], [], []
        for edge in edge_info:
            sources.append(edge["source_idx"])
            targets.append(edge["target_idx"])
            lag_idx.append(edge.get("lag_idx", edge["lag"] - lag_min))
            signs.append(edge["sign"])

        self.register_buffer("edge_sources", torch.tensor(sources, dtype=torch.long))
        self.register_buffer("edge_targets", torch.tensor(targets, dtype=torch.long))
        self.register_buffer("edge_lag_idx", torch.tensor(lag_idx, dtype=torch.long))
        self.register_buffer("edge_signs", torch.tensor(signs, dtype=torch.float32))
        self._A_raw = nn.Parameter(torch.full((len(edge_info),), 0.5413248546))

        print(f"[gLVModel] N={N}, edges={len(edge_info)}, lag={lag_min}-{lag_max}")

    def get_self_interaction(self):
        return -F.softplus(self.self_interaction_raw)

    def get_A_edge(self):
        """English documentation."""
        return F.softplus(self._A_raw) * self.edge_signs

    def compute_growth_rate(self, h, history):
        self_interaction = self.get_self_interaction()
        A_edge = self.get_A_edge()
        h_lag_i = history[self.edge_lag_idx, self.edge_sources]
        interactions = A_edge * h_lag_i
        interaction = torch.zeros_like(h)
        interaction.index_add_(0, self.edge_targets, interactions)
        return self.intrinsic_growth_rate + self_interaction * h + interaction

    def forward(self, h, history, dt):
        h = torch.clamp(h, min=1e-5)
        growth_rate = self.compute_growth_rate(h, history)
        log_growth = torch.clamp(growth_rate * dt, min=-10.0, max=10.0)
        h_next = h * torch.exp(log_growth)
        h_next = torch.clamp(h_next, min=1e-5)
        return h_next

    def integrate(self, h0, history_init, times):
        """English documentation."""
        if history_init.shape[0] != self.lag_max:
            if self.lag_min == 1 and history_init.shape[0] == self.num_lags:
                pass
            else:
                raise ValueError(
                    "history_init must contain states for lags 1..lag_max"
                )
        recent_states = [history_init[i].clone() for i in range(history_init.shape[0])]
        trajectory = [h0.clone()]
        h = h0.clone()
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            hist_list = []
            for lag in range(self.lag_min, self.lag_max + 1):
                hist_list.append(recent_states[lag - 1])
            history_tensor = torch.stack(hist_list)
            h_next = self.forward(h, history_tensor, dt)
            recent_states = [h.clone()] + recent_states[: self.lag_max - 1]
            trajectory.append(h_next)
            h = h_next
        return torch.stack(trajectory)


class gLVModel_RK(nn.Module):
    """English documentation."""

    def __init__(self, N, lag_min, lag_max, edge_info):
        super().__init__()
        self.N = N
        self.lag_min = lag_min
        self.lag_max = lag_max
        self.num_lags = lag_max - lag_min + 1
        self.intrinsic_growth_rate = nn.Parameter(torch.zeros(N))
        self.self_interaction_raw = nn.Parameter(torch.ones(N) * 1.0)
        sources, targets, lag_idx, signs = [], [], [], []
        for edge in edge_info:
            sources.append(edge["source_idx"])
            targets.append(edge["target_idx"])
            lag_idx.append(edge.get("lag_idx", edge["lag"] - lag_min))
            signs.append(edge["sign"])
        self.register_buffer("edge_sources", torch.tensor(sources, dtype=torch.long))
        self.register_buffer("edge_targets", torch.tensor(targets, dtype=torch.long))
        self.register_buffer("edge_lag_idx", torch.tensor(lag_idx, dtype=torch.long))
        self.register_buffer("edge_signs", torch.tensor(signs, dtype=torch.float32))
        self._A_raw = nn.Parameter(torch.full((len(edge_info),), 0.5413248546))
        print(f"[gLVModel_RK] N={N}, edges={len(edge_info)}, lag={lag_min}-{lag_max}")

    def get_self_interaction(self):
        return -F.softplus(self.self_interaction_raw)

    def get_A_edge(self):
        return F.softplus(self._A_raw) * self.edge_signs

    def compute_growth_rate(self, h, history):
        self_interaction = self.get_self_interaction()
        A_edge = self.get_A_edge()
        h_lag_i = history[self.edge_lag_idx, self.edge_sources]
        interactions = A_edge * h_lag_i
        interaction = torch.zeros_like(h)
        interaction.index_add_(0, self.edge_targets, interactions)
        return self.intrinsic_growth_rate + self_interaction * h + interaction

    def forward(self, h, history, dt):
        gr1 = self.compute_growth_rate(h, history)
        k1 = h * gr1
        h1 = h + 0.5 * dt * k1
        gr2 = self.compute_growth_rate(h1, history)
        k2 = h1 * gr2
        h2 = h + 0.5 * dt * k2
        gr3 = self.compute_growth_rate(h2, history)
        k3 = h2 * gr3
        h3 = h + dt * k3
        gr4 = self.compute_growth_rate(h3, history)
        k4 = h3 * gr4
        h_next = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return torch.clamp(h_next, min=1e-5)

    def integrate(self, h0, history_init, times):
        if history_init.shape[0] != self.lag_max:
            if self.lag_min == 1 and history_init.shape[0] == self.num_lags:
                pass
            else:
                raise ValueError(
                    "history_init must contain states for lags 1..lag_max"
                )
        recent_states = [history_init[i].clone() for i in range(history_init.shape[0])]
        trajectory = [h0.clone()]
        h = h0.clone()
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            hist_list = []
            for lag in range(self.lag_min, self.lag_max + 1):
                hist_list.append(recent_states[lag - 1])
            history_tensor = torch.stack(hist_list)
            h_next = self.forward(h, history_tensor, dt)
            recent_states = [h.clone()] + recent_states[: self.lag_max - 1]
            trajectory.append(h_next)
            h = h_next
        return torch.stack(trajectory)



def build_history_from_abundances(abundances: np.ndarray, t: int,
                                  lag_min: int, lag_max: int):
    """English documentation."""
    num_lags = lag_max - lag_min + 1
    N = abundances.shape[1]
    history = np.zeros((num_lags, N), dtype=np.float32)
    for k, lag in enumerate(range(lag_min, lag_max + 1)):
        idx = t - lag
        if idx >= 0:
            history[k, :] = abundances[idx, :]
        else:
            history[k, :] = abundances[0, :]
    return torch.tensor(history, dtype=torch.float32)


def train_glv_model(model, train_pairs, epochs: int = 10, lr: float = 0.01,
                    device="cpu", verbose: bool = True):
    """English documentation."""
    import random
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_pairs)
        total_mse = 0.0
        n = 0
        for pair in train_pairs:
            current = pair["current"].to(device)
            hist = pair["history"].to(device)
            target = pair["target"].to(device)
            dt = pair["dt"].to(device) if hasattr(pair["dt"], "to") else torch.tensor(pair["dt"], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pred = model(current, hist, dt)
            mse = torch.mean((pred - target) ** 2)
            self_interaction_reg = 0.001 * torch.mean(F.softplus(model.self_interaction_raw) ** 2)
            A_edge = model.get_A_edge()
            if A_edge.numel() > 0:
                edge_reg = 0.001 * torch.mean(A_edge ** 2)
            else:
                edge_reg = torch.zeros((), device=device)
            loss = mse + self_interaction_reg + edge_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_mse += float(mse.item())
            n += 1
        avg = total_mse / max(n, 1)
        history_log.append(avg)
    return history_log
