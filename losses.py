"""
Target Speech Extraction (TSE) Losses and Metrics.

Implements:
1. Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
2. Direct Negative SI-SDR Loss (TargetSISDRLoss)
3. Multi-Resolution STFT Loss (MultiResolutionSTFTLoss)
4. Composite Target Speech Extraction Loss (TSECombinedLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Sequence


# ==============================================================================
# 1. SI-SDR Metric and Loss
# ==============================================================================

def calculate_sisdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> torch.Tensor:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels (dB).

    Formula:
        s_target = (<estimate, target> / ||target||^2) * target
        e_noise  = estimate - s_target
        SI-SDR   = 10 * log10(||s_target||^2 / ||e_noise||^2)
    """
    # Harmonize shape if tensor has singleton channel dimension [Batch, 1, Time] -> [Batch, Time]
    if estimate.dim() == 3 and estimate.size(1) == 1:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        if target.size(1) == 1:
            target = target.squeeze(1)
        elif estimate.dim() == 2 and target.size(1) > 1:
            # If estimate is a single target stream [Batch, Time] and target has multiple sources [Batch, K, Time],
            # evaluate against the primary target speaker (Source 0)
            target = target[:, 0]

    estimate = estimate.float()
    target = target.float()

    if zero_mean:
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)

    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps

    alpha = dot / target_energy
    s_target = alpha * target
    e_noise = estimate - s_target

    s_target_energy = torch.sum(s_target ** 2, dim=-1) + eps
    e_noise_energy = torch.sum(e_noise ** 2, dim=-1) + eps

    ratio = torch.clamp(s_target_energy / torch.clamp(e_noise_energy, min=eps), min=eps)
    sisdr = 10.0 * torch.log10(ratio)
    return sisdr


class NegSISDRLoss(nn.Module):
    """Negative SI-SDR Loss function for minimization."""

    def __init__(self, eps: float = 1e-8, zero_mean: bool = True):
        super().__init__()
        self.eps = eps
        self.zero_mean = zero_mean

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -calculate_sisdr(estimate, target, eps=self.eps, zero_mean=self.zero_mean)


# ==============================================================================
# 2. Multi-Resolution STFT Loss
# ==============================================================================

class SingleResolutionSTFTLoss(nn.Module):
    """Computes spectral convergence and log STFT magnitude loss for a single FFT resolution."""

    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: [..., Time]
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1]).float()
        y_flat = y.view(-1, orig_shape[-1]).float()

        window = self.window.to(dtype=torch.float32, device=x.device)

        X = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        Y = torch.stft(
            y_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        mag_x = torch.abs(X) + 1e-7
        mag_y = torch.abs(Y) + 1e-7

        # Spectral convergence loss
        sc_loss = torch.norm(mag_y - mag_x, p="fro", dim=(-2, -1)) / (torch.norm(mag_y, p="fro", dim=(-2, -1)) + 1e-7)
        # Log magnitude L1 loss
        log_loss = F.l1_loss(torch.log(mag_x), torch.log(mag_y), reduction="none").mean(dim=(-2, -1))

        total_single_loss = sc_loss + log_loss
        return total_single_loss.view(orig_shape[:-1])


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Auxiliary Loss across 3 complementary time-frequency scales:
    (512, 128, 512), (256, 64, 256), and (128, 32, 128).
    Provides smooth, dense gradient flow to stabilize early SNN separation training.
    """

    def __init__(
        self,
        resolutions: Sequence[Tuple[int, int, int]] = (
            (512, 128, 512),
            (256, 64, 256),
            (128, 32, 128),
        ),
    ):
        super().__init__()
        self.loss_layers = nn.ModuleList([
            SingleResolutionSTFTLoss(n_fft, hop, win) for (n_fft, hop, win) in resolutions
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for layer in self.loss_layers:
            loss = loss + layer(x, y)
        return loss / len(self.loss_layers)


# ==============================================================================
# 3. Target Speech Extraction (TSE) Losses (Deterministic, No Permutations)
# ==============================================================================

class TargetSISDRLoss(nn.Module):
    """
    Direct Negative SI-SDR loss for Target Speech Extraction.
    Evaluates only the extracted target stream against target clean ground truth.
    Requires no permutation calculations (O(1) complexity).
    """

    def __init__(self, eps: float = 1e-8, zero_mean: bool = True):
        super().__init__()
        self.eps = eps
        self.zero_mean = zero_mean

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            estimate: [Batch, 1, Time] or [Batch, Time]
            target:   [Batch, 1, Time] or [Batch, Time]

        Returns:
            Tuple of (scalar_loss, per_sample_sisdr_scores)
        """
        if estimate.dim() == 3 and estimate.size(1) == 1:
            estimate = estimate.squeeze(1)
        if target.dim() == 3:
            if target.size(1) == 1:
                target = target.squeeze(1)
            elif estimate.dim() == 2 and target.size(1) > 1:
                target = target[:, 0]

        sisdr = calculate_sisdr(estimate, target, eps=self.eps, zero_mean=self.zero_mean)
        loss = -torch.mean(sisdr)
        return loss, sisdr


class TSECombinedLoss(nn.Module):
    """
    Composite Loss for Target Speech Extraction:
        L_total = -SI_SDR(est, target) + mr_stft_weight * MR_STFT(est, target) + spike_reg_weight * SpikeReg(spike_seq)
    """

    def __init__(
        self,
        mr_stft_weight: float = 0.0,
        spike_reg_weight: float = 0.0,
        target_rate_min: float = 0.15,
        target_rate_max: float = 0.25,
    ):
        super().__init__()
        self.mr_stft_weight = mr_stft_weight
        self.spike_reg_weight = spike_reg_weight
        self.target_rate_min = target_rate_min
        self.target_rate_max = target_rate_max
        self.sisdr_loss = TargetSISDRLoss()
        self.mr_stft_loss = MultiResolutionSTFTLoss() if mr_stft_weight > 0 else None

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        spike_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            estimate: [Batch, 1, Time] or [Batch, Time]
            target:   [Batch, 1, Time] or [Batch, Time]
            spike_seq: Optional SNN spike sequence [S, B, C, L] or [S, ...] for firing rate regularization.

        Returns:
            Tuple of (total_loss, estimate, sisdr_scores)
        """
        loss_sisdr, sisdr_scores = self.sisdr_loss(estimate, target)

        if self.mr_stft_loss is not None and self.mr_stft_weight > 0:
            est_flat = estimate.squeeze(1) if estimate.dim() == 3 and estimate.size(1) == 1 else estimate
            if target.dim() == 3:
                tgt_flat = target[:, 0] if target.size(1) > 1 else target.squeeze(1)
            else:
                tgt_flat = target
            stft_loss = self.mr_stft_loss(est_flat, tgt_flat).mean()
            total_loss = loss_sisdr + self.mr_stft_weight * stft_loss
        else:
            total_loss = loss_sisdr

        if spike_seq is not None and self.spike_reg_weight > 0:
            mean_rate = torch.mean(spike_seq)
            # Penalize deviation from target healthy firing rate band [target_rate_min, target_rate_max]
            reg_loss = F.relu(self.target_rate_min - mean_rate) ** 2 + F.relu(mean_rate - self.target_rate_max) ** 2
            total_loss = total_loss + self.spike_reg_weight * reg_loss

        return total_loss, estimate, sisdr_scores

