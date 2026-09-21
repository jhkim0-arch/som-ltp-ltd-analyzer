"""PyTorch MNIST simulation using ideal or measured SOM conductance updates.

The measured-device path stores each synapse as a differential pair of
conductance states. LTP and LTD curves are endpoint-matched before use. C2C is
an explicit user input and perturbs each programmed conductance increment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _resample(values, pulses):
    values = np.asarray(values, dtype=np.float64)
    return np.interp(np.linspace(0, 1, pulses + 1), np.linspace(0, 1, len(values)), values)


def endpoint_match(ltp_raw, ltd_raw, pulses=None):
    """Return endpoint-matched low-to-high LTP and LTD ladders."""
    if len(ltp_raw) < 2 or len(ltd_raw) < 2:
        raise ValueError("LTP and LTD each need at least two conductance values.")
    raw_pulses = max(len(ltp_raw), len(ltd_raw)) - 1
    pulses = raw_pulses if pulses is None else int(pulses)
    ltp = _resample(ltp_raw, pulses)
    ltd_forward = _resample(np.asarray(ltd_raw, dtype=float)[::-1], pulses)
    if np.any(np.diff(ltp) < 0):
        raise ValueError("LTP must be entered from low conductance to high conductance.")
    if np.any(np.diff(ltd_forward) < 0):
        raise ValueError("LTD must be entered from high conductance to low conductance.")
    gmin = 0.5 * (ltp[0] + ltd_forward[0])
    gmax = 0.5 * (ltp[-1] + ltd_forward[-1])
    if not (gmax > gmin > 0):
        raise ValueError("Conductance endpoints must satisfy 0 < Gmin < Gmax.")

    def remap(curve):
        return gmin + (curve - curve[0]) * (gmax - gmin) / (curve[-1] - curve[0])

    return remap(ltp), remap(ltd_forward)


def ideal_ladders(pulses=64, ratio=50.0):
    gmin = 1.0
    return np.geomspace(gmin, gmin * ratio, int(pulses) + 1)


@dataclass
class DeviceMetrics:
    mode: str
    pulses: int
    gmin: float
    gmax: float
    g_ratio: float
    c2c_percent: float


class PulsePairLinear(nn.Module):
    """Differential conductance pair with a residual pulse accumulator."""

    def __init__(self, in_features, out_features, ladder_ltp, ladder_ltd, c2c_fraction):
        super().__init__()
        self.proxy = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.proxy, a=np.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("ltp", ladder_ltp)
        self.register_buffer("ltd", ladder_ltd)
        self.register_buffer("state_plus", torch.zeros_like(self.proxy, dtype=torch.long))
        self.register_buffer("state_minus", torch.zeros_like(self.proxy, dtype=torch.long))
        self.register_buffer("g_plus", torch.zeros_like(self.proxy))
        self.register_buffer("g_minus", torch.zeros_like(self.proxy))
        self.register_buffer("residual", torch.zeros_like(self.proxy))
        self.c2c_fraction = float(c2c_fraction)
        self.reset_devices()

    @property
    def pulses(self):
        return self.ltp.numel() - 1

    @property
    def gmin(self):
        return torch.minimum(self.ltp.min(), self.ltd.min())

    @property
    def gmax(self):
        return torch.maximum(self.ltp.max(), self.ltd.max())

    def physical_weight(self):
        return (self.g_plus - self.g_minus) / (self.gmax - self.gmin)

    @torch.no_grad()
    def reset_devices(self):
        midpoint = self.pulses // 2
        delta = torch.round(self.proxy.detach().clamp(-0.5, 0.5) * self.pulses / 2).to(torch.long)
        self.state_plus.copy_((midpoint + delta).clamp(0, self.pulses))
        self.state_minus.copy_((midpoint - delta).clamp(0, self.pulses))
        average_ladder = 0.5 * (self.ltp + self.ltd)
        self.g_plus.copy_(average_ladder[self.state_plus])
        self.g_minus.copy_(average_ladder[self.state_minus])
        self.proxy.zero_()
        self.residual.zero_()

    def forward(self, x):
        physical = self.physical_weight()
        # Straight-through estimator: forward uses conductance, backward reaches proxy.
        if self.training:
            physical = physical + self.proxy - self.proxy.detach()
        return F.linear(x, physical, self.bias)

    @torch.no_grad()
    def _write(self, state, conductance, direction):
        next_state = (state + direction).clamp(0, self.pulses)
        changed = next_state != state
        if not changed.any():
            return
        state.copy_(next_state)
        moving_up = direction[changed] > 0
        target = torch.where(moving_up, self.ltp[next_state[changed]], self.ltd[next_state[changed]])
        delta_g = target - conductance[changed]
        if self.c2c_fraction:
            delta_g = delta_g * (1 + torch.randn_like(delta_g) * self.c2c_fraction)
        conductance[changed] = (conductance[changed] + delta_g).clamp(self.gmin, self.gmax)

    @torch.no_grad()
    def program_from_gradient(self, learning_rate):
        if self.proxy.grad is None:
            return
        requested = self.residual + (-learning_rate * self.proxy.grad).clamp(-1.0, 1.0)
        # Multiple pulses can be programmed after an accumulated SGD update.
        for _ in range(self.pulses):
            direction = torch.sign(requested).to(torch.long)
            plus_next = (self.state_plus + direction).clamp(0, self.pulses)
            minus_next = (self.state_minus - direction).clamp(0, self.pulses)
            plus_target = torch.where(direction > 0, self.ltp[plus_next], self.ltd[plus_next])
            minus_target = torch.where((-direction) > 0, self.ltp[minus_next], self.ltd[minus_next])
            nominal = ((plus_target - self.g_plus) - (minus_target - self.g_minus)) / (self.gmax - self.gmin)
            active = (direction != 0) & (nominal.abs() > 1e-12) & (requested.abs() >= 0.5 * nominal.abs())
            if not active.any():
                break
            write_direction = direction * active.to(torch.long)
            self._write(self.state_plus, self.g_plus, write_direction)
            self._write(self.state_minus, self.g_minus, -write_direction)
            requested = requested - nominal * active.to(torch.float32)
        self.residual.copy_(requested)
        self.proxy.zero_()


class SOMMLP(nn.Module):
    def __init__(self, ltp, ltd, c2c):
        super().__init__()
        self.fc1 = PulsePairLinear(400, 100, ltp, ltd, c2c)
        self.fc2 = PulsePairLinear(100, 10, ltp, ltd, c2c)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

    @torch.no_grad()
    def program_from_gradient(self, learning_rate):
        self.fc1.program_from_gradient(learning_rate)
        self.fc2.program_from_gradient(learning_rate)


def _loaders(data_dir, batch_size, train_limit, test_limit):
    transform = transforms.Compose([
        transforms.Resize((20, 20)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    if train_limit and train_limit < len(train):
        train = Subset(train, range(int(train_limit)))
    if test_limit and test_limit < len(test):
        test = Subset(test, range(int(test_limit)))
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0), DataLoader(test, batch_size=1000, shuffle=False, num_workers=0)


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    matrix = torch.zeros((10, 10), dtype=torch.int64)
    for image, label in loader:
        label = label.to(device)
        prediction = model(image.to(device)).argmax(1)
        correct += (prediction == label).sum().item()
        total += label.numel()
        matrix += torch.bincount(label.cpu() * 10 + prediction.cpu(), minlength=100).reshape(10, 10)
    return 100.0 * correct / total, matrix.numpy()


def run_mnist_simulation(ltp_raw, ltd_raw, c2c_percent=0.0, mode="Measured SOM", epochs=5,
                         batch_size=128, learning_rate=0.01, train_limit=10000, test_limit=2000,
                         ideal_pulses=64, seed=7, progress=None):
    """Run MNIST and return history, final confusion matrix and device metrics."""
    if c2c_percent < 0:
        raise ValueError("C2C must be zero or greater.")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "Ideal":
        ladder = ideal_ladders(ideal_pulses)
        ltp_np = ltd_np = ladder
        metric = DeviceMetrics("Ideal", ideal_pulses, float(ladder[0]), float(ladder[-1]), float(ladder[-1] / ladder[0]), float(c2c_percent))
    else:
        ltp_np, ltd_np = endpoint_match(ltp_raw, ltd_raw)
        metric = DeviceMetrics("Measured SOM", len(ltp_np) - 1, float(ltp_np[0]), float(ltp_np[-1]), float(ltp_np[-1] / ltp_np[0]), float(c2c_percent))
    ltp = torch.tensor(ltp_np, dtype=torch.float32, device=device)
    ltd = torch.tensor(ltd_np, dtype=torch.float32, device=device)
    model = SOMMLP(ltp, ltd, metric.c2c_percent / 100.0).to(device)
    optimizer = torch.optim.SGD([model.fc1.bias, model.fc2.bias], lr=learning_rate)
    train, test = _loaders(Path("./mnist_data"), int(batch_size), train_limit, test_limit)
    initial, _ = _evaluate(model, test, device)
    history = [initial]
    if progress:
        progress(0, initial)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        for image, label in train:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(image.to(device)), label.to(device))
            loss.backward(); optimizer.step(); model.program_from_gradient(learning_rate)
        accuracy, _ = _evaluate(model, test, device)
        history.append(accuracy)
        if progress:
            progress(epoch, accuracy)
    _, matrix = _evaluate(model, test, device)
    return history, matrix, metric
