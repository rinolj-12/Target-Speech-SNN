"""
STFT Spectrogram Analysis Encoder for Speech Separation.

Transforms raw 1-D time-domain audio mixture waveform into a time-frequency
magnitude spectrogram representation W and phase tensor, with optional
learnable 1x1 1-D convolution projection to match the desired latent channel width N.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SpectrogramEncoder(nn.Module):
    """
    Short-Time Fourier Transform (STFT) Spectrogram Encoder.

    Transforms raw 1-D time-domain audio mixture waveform into a time-frequency
    magnitude spectrogram representation W and phase tensor, with optional
    learnable 1x1 1-D convolution projection to match the desired latent channel width N.

    Args:
        n_fft (int): Size of Fourier Transform (default: 256, e.g. 32ms @ 8kHz).
        hop_length (int): Hop length / frame shift (default: 64, e.g. 8ms @ 8kHz).
        win_length (Optional[int]): Window size (default: matches n_fft).
        window (str): Window function ('hann', 'hamming'). Default: 'hann'.
        encoder_channels (int): Target latent dimension N for separator interface (default: 64).
        use_projection (bool): Whether to project STFT frequency bins (n_fft//2 + 1)
                               to encoder_channels via 1x1 Conv + GroupNorm + ReLU. Default: True.
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: Optional[int] = None,
        window: str = "hann",
        encoder_channels: int = 64,
        use_projection: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.n_freq = self.n_fft // 2 + 1
        self.encoder_channels = encoder_channels
        self.use_projection = use_projection

        # Create and register STFT analysis window
        if window.lower() == "hamming":
            win_tensor = torch.hamming_window(self.win_length)
        else:
            win_tensor = torch.hann_window(self.win_length)
        self.register_buffer("window", win_tensor)

        # Optional 1x1 Conv projection between STFT frequency bins and latent channels
        if self.use_projection or (self.n_freq != self.encoder_channels):
            self.proj = nn.Conv1d(self.n_freq, self.encoder_channels, kernel_size=1, bias=False)
            self.norm = nn.GroupNorm(1, self.encoder_channels, eps=1e-8)
            self.relu = nn.ReLU()
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
        else:
            self.proj = None
            self.norm = None
            self.relu = None

    def forward(self, x: torch.Tensor, verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing STFT spectrogram.

        Args:
            x (torch.Tensor): Time-domain audio tensor [Batch, 1, Time] or [Batch, Time].
            verbose (bool): If True, print tensor shapes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - w (torch.Tensor): Magnitude feature representation [Batch, N, Latent_Length].
                - phase (torch.Tensor): STFT phase angles [Batch, F, Latent_Length] for reconstruction.
        """
        if x.dim() == 3:
            x_flat = x.squeeze(1)  # [B, 1, T] -> [B, T]
        else:
            x_flat = x

        window = self.window.to(dtype=x.dtype, device=x.device)
        
        # Compute Complex STFT: [Batch, F, Frames]
        X = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        mag = torch.abs(X)     # [B, F, L]
        phase = torch.angle(X) # [B, F, L]

        # Project frequency bins F -> N if configured
        if self.proj is not None:
            w = self.relu(self.norm(self.proj(mag)))
        else:
            w = mag

        if verbose:
            print(f"[SpectrogramEncoder] Waveform {x.shape} -> STFT Mag {mag.shape} -> Latent W {w.shape}")

        return w, phase


# Canonical alias for SpectrogramEncoder
STFTEncoder = SpectrogramEncoder
