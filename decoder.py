"""
STFT Spectrogram Synthesis Decoder (iSTFT) for Speech Separation.

Transforms separated latent feature representations back into the time domain
using Inverse Short-Time Fourier Transform (iSTFT) with mixture phase.
"""

import torch
import torch.nn as nn
from typing import Optional


class SpectrogramDecoder(nn.Module):
    """
    Inverse Short-Time Fourier Transform (iSTFT) Spectrogram Decoder.

    Synthesizes time-domain separated waveforms from masked latent features and mixture phase.
    Unprojects latent channels N back to STFT frequency bins F (if projection was used),
    combines estimated magnitudes with mixture phase angles, and applies inverse STFT.

    Args:
        n_fft (int): Size of Fourier Transform (default: 256).
        hop_length (int): Hop length / frame shift (default: 64).
        win_length (Optional[int]): Window size (default: matches n_fft).
        window (str): Window function ('hann', 'hamming'). Default: 'hann'.
        encoder_channels (int): Latent dimension N from separator (default: 64).
        use_projection (bool): Whether 1x1 Conv projection is needed from N -> F. Default: True.
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: Optional[int] = None,
        window: str = "hann",
        encoder_channels: int = 64,
        use_projection: bool = True,
        mask_type: str = "complex",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.n_freq = self.n_fft // 2 + 1
        self.encoder_channels = encoder_channels
        self.use_projection = use_projection
        self.mask_type = mask_type.lower()

        # Create and register STFT synthesis window
        if window.lower() == "hamming":
            win_tensor = torch.hamming_window(self.win_length)
        else:
            win_tensor = torch.hann_window(self.win_length)
        self.register_buffer("window", win_tensor)

        # Unprojection: [encoder_channels N -> n_freq F]
        # Not needed for complex ratio masking since complex_spec is synthesized directly from STFT bins
        if self.mask_type != "complex" and (self.use_projection or (self.n_freq != self.encoder_channels)):
            self.unproj = nn.Conv1d(self.encoder_channels, self.n_freq, kernel_size=1, bias=False)
            self.relu = nn.ReLU()
            nn.init.kaiming_normal_(self.unproj.weight, mode="fan_in")
        else:
            self.unproj = None
            self.relu = None

    def forward(
        self,
        w_masked: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        verbose: bool = False,
        complex_spec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Spectrogram Decoder using iSTFT.

        Args:
            w_masked (torch.Tensor, optional): Masked latent features [Batch, Num_Sources, N, Latent_Length]
                                               or [Batch, N, Latent_Length] (for magnitude mode).
            phase (torch.Tensor, optional): Mixture STFT phase [Batch, F, Latent_Length] (for magnitude mode).
            target_length (int, optional): Original waveform length in samples.
            verbose (bool): If True, print tensor shapes.
            complex_spec (torch.Tensor, optional): Estimated target complex STFT spectrum
                                                   [Batch, Num_Sources, F, Latent_Length] or [Batch, F, Latent_Length]
                                                   (for complex ratio masking mode).

        Returns:
            torch.Tensor: Reconstructed time-domain audio [Batch, Num_Sources, Time] or [Batch, 1, Time].
        """
        # Fast path: Direct synthesis from estimated complex STFT (Complex Ratio Masking)
        if complex_spec is not None:
            if complex_spec.dim() == 4:
                b, k, f, l = complex_spec.shape
                spec_flat = complex_spec.reshape(b * k, f, l)
            elif complex_spec.dim() == 3:
                b, f, l = complex_spec.shape
                k = 1
                spec_flat = complex_spec
            else:
                raise ValueError(f"Unexpected complex_spec dimension: {complex_spec.shape}")

            window = self.window.to(dtype=torch.float32, device=spec_flat.device)
            if spec_flat.dtype not in (torch.complex64, torch.complex128):
                spec_flat = spec_flat.cfloat()

            audio_flat = torch.istft(
                spec_flat,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                length=target_length,
            )
            time_dim = audio_flat.shape[-1]
            audio = audio_flat.reshape(b, k, time_dim)

            if target_length is not None:
                if audio.shape[-1] > target_length:
                    audio = audio[..., :target_length]
                elif audio.shape[-1] < target_length:
                    pad_len = target_length - audio.shape[-1]
                    audio = nn.functional.pad(audio, (0, pad_len))

            if verbose:
                print(f"[SpectrogramDecoder (Complex)] Input: {complex_spec.shape} -> Synthesized waveform: {audio.shape}")

            return audio

        if w_masked is None:
            raise ValueError("Either complex_spec or w_masked must be provided to SpectrogramDecoder.")

        is_multi_source = (w_masked.dim() == 4)

        if is_multi_source:
            b, k, n, l = w_masked.shape
            w_flat = w_masked.reshape(b * k, n, l)

            # Map from latent channels N -> frequency bins F
            if self.unproj is not None:
                mag_flat = self.relu(self.unproj(w_flat))
            else:
                mag_flat = w_flat

            # Expand phase across sources: [B, F, L] -> [B*K, F, L]
            if phase is not None:
                phase_flat = phase.unsqueeze(1).repeat(1, k, 1, 1).reshape(b * k, -1, l)
            else:
                phase_flat = torch.zeros_like(mag_flat)

            # Combine magnitude and phase into complex STFT representation
            complex_spec = torch.polar(mag_flat.float(), phase_flat.float())
            if complex_spec.dtype not in (torch.complex64, torch.complex128):
                complex_spec = complex_spec.cfloat()

            window = self.window.to(dtype=torch.float32, device=complex_spec.device)

            # Synthesize time-domain audio via inverse STFT
            audio_flat = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                length=target_length,
            )

            # Reshape back to [Batch, Num_Sources, Time]
            time_dim = audio_flat.shape[-1]
            audio = audio_flat.reshape(b, k, time_dim)

            # Exact length trim/pad if necessary
            if target_length is not None:
                if audio.shape[-1] > target_length:
                    audio = audio[..., :target_length]
                elif audio.shape[-1] < target_length:
                    pad_len = target_length - audio.shape[-1]
                    audio = nn.functional.pad(audio, (0, pad_len))
        else:
            b, n, l = w_masked.shape
            if self.unproj is not None:
                mag = self.relu(self.unproj(w_masked))
            else:
                mag = w_masked

            if phase is not None:
                phase_val = phase
            else:
                phase_val = torch.zeros_like(mag)

            complex_spec = torch.polar(mag.float(), phase_val.float())
            if complex_spec.dtype not in (torch.complex64, torch.complex128):
                complex_spec = complex_spec.cfloat()

            window = self.window.to(dtype=torch.float32, device=complex_spec.device)

            audio_flat = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                length=target_length,
            )

            audio = audio_flat.unsqueeze(1)  # [B, 1, Time]
            if target_length is not None:
                if audio.shape[-1] > target_length:
                    audio = audio[..., :target_length]
                elif audio.shape[-1] < target_length:
                    pad_len = target_length - audio.shape[-1]
                    audio = nn.functional.pad(audio, (0, pad_len))

        if verbose:
            print(f"[SpectrogramDecoder] Latent input: {w_masked.shape} -> Synthesized waveform: {audio.shape}")

        return audio


# Canonical alias for SpectrogramDecoder
STFTDecoder = SpectrogramDecoder


