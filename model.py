"""
Spiking Conv-TasNet Architecture (STFT Spectrogram Pipeline):
SpikingConvTasNet (STFT Spectrogram Encoder, SNN-based Separator with PLIF/ALIF/FS-Neuron,
Bit-Plane/Population/Direct Coding, Continuous Membrane Readout, Continuous Residual Bridge,
and iSTFT Spectrogram Decoder with mixture phase).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from config import ModelConfig
from encoder import SpectrogramEncoder
from decoder import SpectrogramDecoder
from spike_encoder import SpikeEncoder
from separator import SpikingTCNSeparator


# ==============================================================================
# 1. Spiking Conv-TasNet Model (STFT Frontend)
# ==============================================================================

class SpikingConvTasNet(nn.Module):
    """
    SNN-based Conv-TasNet Speech Separation Model (STFT Spectrogram Frontend).

    Modular Pipeline:
        Mixture Waveform x: [B, 1, T]
            | (STFT Spectrogram Encoder: W = ReLU(Norm(Proj(|STFT(x)|))))
        Latent Representation W: [B, N, L] + Mixture Phase: [B, F, L]
            | (Spike Encoding: Direct Current, Bit-Plane, Learnable PLIF, or Population)
        Spike Sequence: [S, B, N, L]
            | (Spiking TCN Separator with PLIF / FS-Neuron / ALIF dynamics & Membrane Readout)
        Speaker Masks M: [B, K, N, L]
            | (Masking: W_k = M_k * W)
        Masked Latent Features W_masked: [B, K, N, L]
            | (iSTFT Spectrogram Decoder: Unproj(W_masked) + Mixture Phase -> iSTFT)
        Separated Waveforms s_hat: [B, K, T]
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        self.encoder_type: str = "spectrogram"

        # 1. STFT Spectrogram Encoder
        self.encoder = SpectrogramEncoder(
            n_fft=config.stft_n_fft,
            hop_length=config.stft_hop_length,
            win_length=config.stft_win_length,
            window=config.stft_window,
            encoder_channels=config.encoder_channels,
            use_projection=config.stft_use_projection,
        )

        # 2. Spike Encoder (continuous W -> temporal spike train)
        self.spike_encoder = SpikeEncoder(
            timesteps=config.snn_timesteps,
            encoding_type=config.spike_encoding,
            channels=config.encoder_channels,
            threshold=config.snn_threshold,
            population_factor=getattr(config, "population_factor", 4),
            surrogate=config.surrogate,
        )

        # 3. Spiking TCN Separator with selectable neuron dynamics & continuous membrane readout
        spk_dim = getattr(config, "spk_embed_dim", 192) if getattr(config, "task_mode", "tse") == "tse" else None
        self.separator = SpikingTCNSeparator(
            in_channels=config.encoder_channels,
            bottleneck_channels=config.bottleneck_channels,
            hidden_channels=config.hidden_channels,
            kernel_size=config.kernel_size,
            dilations=config.dilations,
            num_repeats=config.num_repeats,
            num_sources=config.num_sources,
            mask_activation=config.mask_activation,
            neuron_type=getattr(config, "neuron_type", "plif"),
            beta=config.snn_beta,
            threshold=config.snn_threshold,
            timesteps=config.snn_timesteps,
            snn_readout=getattr(config, "snn_readout", "membrane"),
            use_residual_bridge=getattr(config, "use_residual_bridge", True),
            surrogate=config.surrogate,
            use_checkpointing=getattr(config, "use_checkpointing", True),
            spk_embed_dim=spk_dim,
        )

        # 4. iSTFT Spectrogram Decoder
        self.decoder = SpectrogramDecoder(
            n_fft=config.stft_n_fft,
            hop_length=config.stft_hop_length,
            win_length=config.stft_win_length,
            window=config.stft_window,
            encoder_channels=config.encoder_channels,
            use_projection=config.stft_use_projection,
        )

    def forward(
        self,
        x: torch.Tensor,
        spk_embed: Optional[torch.Tensor] = None,
        verbose: Optional[bool] = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through STFT-based Spiking Conv-TasNet.

        Args:
            x (torch.Tensor): Mixture audio waveform.
                              Shape: [Batch, 1, Time] or [Batch, Time].
            spk_embed (torch.Tensor, optional): Target speaker voiceprint embedding [Batch, D_spk].
            verbose (bool, optional): If True, prints tensor shapes throughout the pipeline.
            return_intermediates (bool, optional): If True, returns (separated_waveforms, masks, spike_seq).

        Returns:
            torch.Tensor or Tuple: Separated audio waveforms [Batch, Num_Sources, Time]
                                   or (separated_waveforms, masks, spike_seq).
        """
        if verbose is None:
            verbose = self.config.verbose_shapes

        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        batch, _, orig_time = x.shape

        if verbose:
            print("\n" + "=" * 50)
            print(" [SpikingConvTasNet (STFT Spectrogram)] Forward Pass Shape Trace")
            print("=" * 50)
            print(f"1. Input Mixture Waveform      : {x.shape} (B={batch}, Time={orig_time})")
            if spk_embed is not None:
                print(f"   Target Speaker Voiceprint   : {spk_embed.shape}")

        # Step 1: STFT Spectrogram Encoding (returns latent magnitude W and mixture phase)
        w, phase = self.encoder(x, verbose=verbose)

        # Step 2: Temporal Spike Encoding
        spike_seq = self.spike_encoder(w, verbose=verbose)

        if verbose:
            print(f"2. Latent Representation W     : {w.shape} (B={w.shape[0]}, N={w.shape[1]}, L={w.shape[2]})")
            print(f"3. Spiking Sequence (S={self.config.snn_timesteps})     : {spike_seq.shape}")

        # Step 3: Estimate speaker separation masks via Spiking TCN (with continuous bridge if enabled)
        bridge_w = w if getattr(self.config, "use_residual_bridge", True) else None
        masks = self.separator(spike_seq, continuous_w=bridge_w, spk_embed=spk_embed, verbose=verbose)
        if verbose:
            print(f"4. Predicted Speaker Masks M   : {masks.shape} (Sources K={masks.shape[1]})")

        # Step 4: Mask the latent representation W
        w_expanded = w.unsqueeze(1)
        w_masked = masks * w_expanded
        if verbose:
            print(f"5. Masked Latent Representations: {w_masked.shape}")

        # Step 5: iSTFT Spectrogram Synthesis with mixture phase
        separated_waveforms = self.decoder(w_masked, phase=phase, target_length=orig_time, verbose=verbose)

        if verbose:
            print(f"6. Separated Output Waveforms   : {separated_waveforms.shape}")
            print("=" * 50 + "\n")

        if return_intermediates:
            return separated_waveforms, masks, spike_seq

        return separated_waveforms


# Aliases for clarity and backward compatibility
SpikingSTFTTasNet = SpikingConvTasNet


def build_model(config: Optional[ModelConfig] = None) -> SpikingConvTasNet:
    """Factory helper to build Spiking Conv-TasNet model."""
    if config is None:
        config = ModelConfig()
    return SpikingConvTasNet(config)


