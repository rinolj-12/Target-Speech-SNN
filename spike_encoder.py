"""
Spike Encoding Interface.

Converts a continuous latent feature representation W [Batch, N, Latent_Length]
into a high-precision temporal spike/current sequence [Timesteps S, Batch, N_out, Latent_Length]
for the SNN separator.

Supported Schemes:
1. 'bit_plane' / 'fs_neuron': Positional base-2 multi-threshold bit-plane decomposition (2^S levels in S steps).
2. 'population': Gaussian receptive-field population coding (maps N -> K * N channels with spatial tuning).
3. 'learnable_plif' / 'learnable': 1D Conv projection + LayerNorm + Parametric LIF with learnable thresholds.
4. 'direct_current': Replicates input features W across all S simulation timesteps as constant input current.
5. 'rate': Rate-based stochastic Bernoulli spike encoding.
6. 'threshold': Deterministic step thresholding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from snn import get_surrogate_fn, PLIFNeuron, FSNeuron


class BitPlaneSpikeEncoder(nn.Module):
    """
    Bit-Plane / FS-Neuron Spike Encoder.
    Decomposes continuous latent tensor W in [0, max_val] into S binary spike bit planes
    representing binary radix weights [2^-1, 2^-2, ..., 2^-S].

    Forward pass:
        R[0] = W_norm
        For t = 0 to S-1:
            weight_t = 2^(-(t+1))
            S[t] = SurrogateStep(R[t] - weight_t)
            R[t+1] = R[t] - weight_t * S[t]

    Enables exact 2^S discrete representation with full differentiability.
    """

    def __init__(
        self,
        timesteps: int = 4,
        max_val: float = 2.0,
        surrogate: str = "fast_sigmoid",
    ):
        super().__init__()
        self.timesteps = timesteps
        self.max_val = max_val
        self.spike_fn = get_surrogate_fn(surrogate)
        
        # Base-2 weights: [0.5, 0.25, 0.125, 0.0625, ...]
        weights = [2.0 ** (-(t + 1)) for t in range(timesteps)]
        self.register_buffer("bit_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # w: [Batch, N, Latent_Length]
        # Normalize w to [0, 1) range with FP16-safe epsilon floor
        eps = 1e-4 if w.dtype in (torch.float16, torch.bfloat16) else 1e-7
        w_norm = torch.clamp(w / (self.max_val + eps), min=0.0, max=0.9999)
        
        remainder = w_norm
        spike_list = []
        
        for t in range(self.timesteps):
            wt = self.bit_weights[t].to(dtype=w.dtype, device=w.device)
            # Fire spike if remainder >= wt
            spike_t = self.spike_fn(remainder - wt, 0.0)
            # Subtract fired bit
            remainder = remainder - wt * spike_t
            spike_list.append(spike_t)

        # Shape: [Timesteps S, Batch, N, Latent_Length]
        spike_seq = torch.stack(spike_list, dim=0)
        return spike_seq


class PopulationSpikeEncoder(nn.Module):
    """
    Gaussian Population Spike Encoder.
    Expands each continuous channel N into K sub-neurons with Gaussian receptive fields:
        f_k(x) = exp(- (x - mu_k)^2 / (2 * sigma_k^2))
    Preserves continuous amplitude with high dynamic range at S=1 to 4 timesteps.
    """

    def __init__(
        self,
        in_channels: int = 64,
        population_size: int = 4,
        timesteps: int = 4,
        min_val: float = 0.0,
        max_val: float = 2.0,
        surrogate: str = "fast_sigmoid",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.population_size = population_size
        self.timesteps = timesteps
        self.spike_fn = get_surrogate_fn(surrogate)

        # Initialize evenly spaced Gaussian centers and widths across [min_val, max_val]
        mu = torch.linspace(min_val, max_val, population_size).view(1, 1, population_size, 1)
        sigma = torch.full_like(mu, (max_val - min_val) / (population_size * 1.5))
        self.mu = nn.Parameter(mu)
        self.sigma = nn.Parameter(sigma)

        # 1x1 Conv to project expanded (N * K) channels back to N channels for separator compatibility
        self.proj = nn.Conv1d(in_channels * population_size, in_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, in_channels, eps=1e-8)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # w: [Batch, N, Latent_Length]
        batch, n_chan, l_len = w.shape
        w_expanded = w.unsqueeze(2)  # [B, N, 1, L]

        # Gaussian activation: [B, N, K, L] with FP16-safe epsilon floor
        eps = 1e-4 if w.dtype in (torch.float16, torch.bfloat16) else 1e-7
        act = torch.exp(-((w_expanded - self.mu) ** 2) / (2 * (self.sigma ** 2) + eps))
        
        # Flatten spatial population into channel dimension: [B, N * K, L]
        flat_act = act.view(batch, n_chan * self.population_size, l_len).to(dtype=self.proj.weight.dtype)
        projected = self.norm(self.proj(flat_act)).to(dtype=w.dtype)  # [B, N, L]

        # Expand (zero-copy view) across S timesteps — avoids allocating a new tensor
        # [S, B, N, L]
        spike_seq = projected.unsqueeze(0).expand(self.timesteps, -1, -1, -1).contiguous()
        return spike_seq


class LearnableSpikeEncoder(nn.Module):
    """
    Learnable Conv-PLIF Spike Encoder.
    Uses a 1D Convolution + GroupNorm + PLIF layer with learnable thresholds and time constants.
    """

    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 4,
        threshold: float = 0.8,
        beta: float = 0.9,
        surrogate: str = "fast_sigmoid",
    ):
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        self.conv1d = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, channels, eps=1e-8)
        self.plif = PLIFNeuron(channels=channels, init_beta=beta, init_threshold=threshold, surrogate=surrogate)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # w: [Batch, N, Latent_Length]
        batch, n_chan, l_len = w.shape
        w_in = w.to(dtype=self.conv1d.weight.dtype)
        h = self.norm(self.conv1d(w_in))  # [B, N, L]

        u = torch.zeros(batch, n_chan, l_len, dtype=h.dtype, device=w.device)
        s = torch.zeros(batch, n_chan, l_len, dtype=h.dtype, device=w.device)
        spike_list = []

        for t in range(self.timesteps):
            s, u = self.plif.step(h, u, s, step_idx=t)
            spike_list.append(s)

        spike_seq = torch.stack(spike_list, dim=0)  # [S, B, N, L]
        return spike_seq


class SpikeEncoder(nn.Module):
    """
    Unified Spike Encoding Interface.

    Converts continuous features W of shape [Batch, N, Latent_Length] into a temporal
    sequence of shape [Timesteps S, Batch, N, Latent_Length].

    Encoding Schemes:
        1. 'bit_plane' / 'fs_neuron' (Recommended): Exact radix bit-plane decomposition (16 levels at S=4).
        2. 'population': Gaussian receptive field population encoding.
        3. 'learnable_plif' / 'learnable': 1D Conv + LayerNorm + Parametric LIF.
        4. 'direct_current': Replicates features across all S simulation timesteps.
        5. 'rate': Stochastic Bernoulli spike generator.
        6. 'threshold': Step thresholding.

    Args:
        timesteps (int): Number of SNN simulation timesteps S. Default: 4.
        encoding_type (str): Encoding mode name. Default: 'bit_plane'.
        channels (int): Number of latent channels N. Default: 64.
        threshold (float): Baseline threshold for rate/threshold/LIF. Default: 0.8.
        population_factor (int): Population size K for population encoder. Default: 4.
        surrogate (str): Surrogate gradient function. Default: 'fast_sigmoid'.
    """

    def __init__(
        self,
        timesteps: int = 4,
        encoding_type: str = "bit_plane",
        channels: int = 64,
        threshold: float = 0.8,
        population_factor: int = 4,
        surrogate: str = "fast_sigmoid",
    ):
        super().__init__()
        self.timesteps = timesteps
        self.encoding_type = encoding_type.lower()
        self.channels = channels
        self.threshold = threshold
        self.surrogate = surrogate

        if self.encoding_type in ["bit_plane", "fs_neuron", "bitplane", "multi_threshold"]:
            self.encoder = BitPlaneSpikeEncoder(
                timesteps=timesteps,
                max_val=max(threshold * 2.0, 1.0),
                surrogate=surrogate,
            )
        elif self.encoding_type in ["population", "pop"]:
            self.encoder = PopulationSpikeEncoder(
                in_channels=channels,
                population_size=population_factor,
                timesteps=timesteps,
                max_val=max(threshold * 2.0, 1.0),
                surrogate=surrogate,
            )
        elif self.encoding_type in ["learnable_plif", "learnable", "plif_encoder"]:
            self.encoder = LearnableSpikeEncoder(
                channels=channels,
                timesteps=timesteps,
                threshold=threshold,
                surrogate=surrogate,
            )
        else:
            self.encoder = None

    def forward(self, w: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Convert continuous latent tensor W to temporal spike/current sequence.

        Args:
            w (torch.Tensor): Continuous latent representation from encoder [Batch, N, Latent_Length].
            verbose (bool): If True, print shapes.

        Returns:
            torch.Tensor: Temporal spike sequence [Timesteps S, Batch, N, Latent_Length].
        """
        if self.encoder is not None:
            spike_seq = self.encoder(w)
        elif self.encoding_type == "direct_current":
            spike_seq = w.unsqueeze(0).expand(self.timesteps, -1, -1, -1).contiguous()
        elif self.encoding_type == "rate":
            eps = 1e-4 if w.dtype in (torch.float16, torch.bfloat16) else 1e-7
            w_norm = torch.clamp(w / (self.threshold + eps), min=0.0, max=1.0)
            probs = w_norm.unsqueeze(0).repeat(self.timesteps, 1, 1, 1)
            spike_seq = torch.bernoulli(probs)
        elif self.encoding_type == "threshold":
            spike_seq = (w.unsqueeze(0).repeat(self.timesteps, 1, 1, 1) >= self.threshold).to(dtype=w.dtype)
        else:
            raise ValueError(f"Unknown spike encoding type: {self.encoding_type}")

        if verbose:
            print(f"[SpikeEncoder ({self.encoding_type})] Latent W: {w.shape} -> Spike sequence: {spike_seq.shape} (S={self.timesteps})")

        return spike_seq
