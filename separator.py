"""
Spiking Separator Modules for Conv-TasNet.

Implements:
1. SpikingTCNSeparator: Dilated Spiking Temporal Convolutional Network with selectable neuron dynamics
   (PLIF, ALIF, APLIF, FS-Neuron, LIF), continuous membrane potential readout, and continuous residual bridge.
"""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple
from snn import SpikingConvBlock1d
from speaker_encoder import SpeakerAdapter


# ==============================================================================
# 1. Spiking TCN Separator (SNN)
# ==============================================================================

class SpikingTCNSeparator(nn.Module):
    """
    Spiking Temporal Convolutional Network (Spiking TCN) Separator.

    Processes temporal spike sequences using stacked dilated 1-D spiking convolution
    blocks with selectable neuron dynamics (PLIF, ALIF, APLIF, FS-Neuron, LIF),
    continuous membrane potential readout, and continuous residual bridging.

    Args:
        in_channels (int): N, number of latent channels from encoder. Default: 64.
        bottleneck_channels (int): B_conv, channels in bottleneck and residual paths. Default: 64.
        hidden_channels (int): H, channels in depthwise convolutional blocks. Default: 128.
        kernel_size (int): P, kernel size of 1-D depthwise convolutions. Default: 3.
        dilations (List[int]): List of dilation rates. Default: [1, 2, 4, 8, 16, 32].
        num_repeats (int): R, number of times to repeat the dilation stack. Default: 2.
        num_sources (int): K, number of target speakers (2 for 2-speaker separation). Default: 2.
        mask_activation (str): Activation function for masks ('relu', 'sigmoid', 'softmax'). Default: 'sigmoid'.
        neuron_type (str): 'plif', 'alif', 'aplif', 'fs_neuron', 'lif'. Default: 'plif'.
        beta (float): SNN membrane potential decay factor. Default: 0.9.
        threshold (float): SNN firing threshold. Default: 0.8.
        timesteps (int): Number of simulation timesteps S. Default: 4.
        snn_readout (str): 'membrane', 'weighted_bit', or 'rate'. Default: 'membrane'.
        use_residual_bridge (bool): Add continuous linear bridge from continuous latent features. Default: True.
        surrogate (str): Surrogate gradient function ('fast_sigmoid', 'atan', 'piecewise'). Default: 'fast_sigmoid'.
        mask_type (str): 'complex' (cRM) or 'magnitude' (legacy). Default: 'complex'.
        n_freq (Optional[int]): Number of STFT frequency bins if complex masking is used. Default: None.
        complex_mask_bound (float): Bounded tanh scaling for complex ratio mask. Default: 2.0.
    """

    def __init__(
        self,
        in_channels: int = 128,
        bottleneck_channels: int = 128,
        hidden_channels: int = 256,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        num_repeats: int = 2,
        num_sources: int = 1,
        mask_activation: str = "sigmoid",
        neuron_type: str = "plif",
        beta: float = 0.9,
        threshold: float = 0.8,
        timesteps: int = 4,
        snn_readout: str = "membrane",
        use_residual_bridge: bool = True,
        surrogate: str = "fast_sigmoid",
        use_checkpointing: bool = True,
        spk_embed_dim: Optional[int] = None,
        mask_type: str = "complex",
        n_freq: Optional[int] = None,
        complex_mask_bound: float = 2.0,
        film_mode: str = "shared_deep",
        use_speaker_adapter: bool = False,
        spk_adapter_hidden_dim: int = 256,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]

        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.hidden_channels = hidden_channels
        self.num_sources = num_sources
        self.mask_activation = mask_activation
        self.neuron_type = neuron_type.lower()
        self.timesteps = timesteps
        self.snn_readout = snn_readout.lower()
        self.use_residual_bridge = use_residual_bridge
        self.use_checkpointing = use_checkpointing
        self.spk_embed_dim = spk_embed_dim
        self.film_mode = film_mode.lower()
        self.use_speaker_adapter = use_speaker_adapter

        # Trainable Residual Speaker Voiceprint Adapter
        if self.use_speaker_adapter and spk_embed_dim is not None and spk_embed_dim > 0:
            self.speaker_adapter = SpeakerAdapter(embed_dim=spk_embed_dim, hidden_dim=spk_adapter_hidden_dim)
        else:
            self.speaker_adapter = None

        # Neuromodulatory FiLM Conditioning Projections for Target Speech Extraction
        self.film_proj = None
        self.shared_film_proj = None
        self.block_film_projs = None

        if spk_embed_dim is not None and spk_embed_dim > 0:
            if self.film_mode == "shared_deep":
                # Single Shared Multi-Scale FiLM generator across all 12 dilation blocks (~49k params)
                self.shared_film_proj = nn.Linear(spk_embed_dim, 2 * bottleneck_channels)
                nn.init.zeros_(self.shared_film_proj.bias)
                nn.init.normal_(self.shared_film_proj.weight, std=0.01)
            elif self.film_mode == "deep":
                # Legacy unshared block FiLM (593k params) + bottleneck FiLM (49k params)
                self.film_proj = nn.Linear(spk_embed_dim, 2 * bottleneck_channels)
                nn.init.zeros_(self.film_proj.bias)
                nn.init.normal_(self.film_proj.weight, std=0.02)
            elif self.film_mode == "bottleneck":
                # Legacy single FiLM on bottleneck
                self.film_proj = nn.Linear(spk_embed_dim, 2 * bottleneck_channels)
                nn.init.zeros_(self.film_proj.bias)
                nn.init.normal_(self.film_proj.weight, std=0.02)

        # Bottleneck 1x1 conv: [N -> B_conv]
        self.bottleneck_conv = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bottleneck_norm = nn.GroupNorm(1, bottleneck_channels, eps=1e-8)

        # Continuous residual bridge projection if enabled: [N -> B_conv]
        if self.use_residual_bridge:
            self.bridge_conv = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            self.bridge_norm = nn.GroupNorm(1, bottleneck_channels, eps=1e-8)
        else:
            self.bridge_conv = None
            self.bridge_norm = None

        # Build stacked Spiking TCN blocks
        self.blocks = nn.ModuleList()
        for r in range(num_repeats):
            for d in dilations:
                block = SpikingConvBlock1d(
                    in_channels=bottleneck_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    neuron_type=neuron_type,
                    beta=beta,
                    threshold=threshold,
                    timesteps=timesteps,
                    surrogate=surrogate,
                    return_membrane=(self.snn_readout == "membrane"),
                )
                self.blocks.append(block)

        # Legacy Deep Multi-Scale FiLM Projections: one dedicated affine generator per TCN block
        if self.film_mode == "deep" and spk_embed_dim is not None and spk_embed_dim > 0:
            self.block_film_projs = nn.ModuleList([
                nn.Linear(spk_embed_dim, 2 * bottleneck_channels)
                for _ in range(len(self.blocks))
            ])
            for proj in self.block_film_projs:
                nn.init.zeros_(proj.bias)
                nn.init.normal_(proj.weight, std=0.01)

        self.mask_type = mask_type.lower()
        self.complex_mask_bound = complex_mask_bound
        self.target_freq = n_freq if n_freq is not None else in_channels

        # Output mask generator
        if self.mask_type == "complex":
            # Real and imaginary mask channels per source: [B_conv -> 2 * K * target_freq]
            out_mask_channels = 2 * num_sources * self.target_freq
            self.mask_conv = nn.Conv1d(
                bottleneck_channels,
                out_mask_channels,
                kernel_size=1,
                bias=True,
            )
            # Initialize with small normal weights for stable start
            nn.init.normal_(self.mask_conv.weight, std=0.01)
            # Initialize bias for identity pass-through at step 0 (M_r = 1.0, M_i = 0.0)
            init_r_bias = math.atanh(min(0.99, 1.0 / max(1.01, self.complex_mask_bound)))
            with torch.no_grad():
                bias_view = self.mask_conv.bias.view(num_sources, 2, self.target_freq)
                bias_view[:, 0, :].fill_(init_r_bias)
                bias_view[:, 1, :].fill_(0.0)
        else:
            # Legacy magnitude mask: [B_conv -> K * N]
            self.mask_conv = nn.Conv1d(
                bottleneck_channels,
                num_sources * in_channels,
                kernel_size=1,
                bias=True,
            )

        if mask_activation == "relu":
            self.activation = nn.ReLU()
        elif mask_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif mask_activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = nn.Identity()

        # Precompute bit weights for radix-weighted readout
        weights = torch.tensor([2.0 ** (-(t + 1)) for t in range(timesteps)], dtype=torch.float32)
        self.register_buffer("bit_weights", weights.view(timesteps, 1, 1, 1))

    def forward(
        self,
        spike_seq: torch.Tensor,
        continuous_w: Optional[torch.Tensor] = None,
        spk_embed: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for Spiking TCN Separator.

        Args:
            spike_seq: Spike / current sequence of shape [Timesteps S, Batch, N, Latent_Length].
            continuous_w: Optional continuous latent features [Batch, N, Latent_Length] for residual bridge.
            spk_embed: Optional target speaker voiceprint vector [Batch, D_spk] for FiLM conditioning.
            verbose: If True, print tensor shapes.

        Returns:
            torch.Tensor: Speaker masks M of shape [Batch, Num_Sources, N, Latent_Length].
        """
        s_steps, batch, n_chan, l_len = spike_seq.shape

        # Step 0: Ensure speaker embedding shape, device, and dtype match expectation
        if spk_embed is not None:
            if spk_embed.dim() == 3 and spk_embed.size(1) == 1:
                spk_embed = spk_embed.squeeze(1)
            target_dtype = self.bottleneck_conv.weight.dtype
            spk_embed = spk_embed.to(dtype=target_dtype, device=spike_seq.device)
            if self.speaker_adapter is not None:
                spk_embed = self.speaker_adapter(spk_embed)

        # Step 1: Apply bottleneck 1x1 conv to each simulation timestep
        flat_spikes = spike_seq.reshape(s_steps * batch, n_chan, l_len).to(dtype=self.bottleneck_conv.weight.dtype)
        flat_bottleneck = self.bottleneck_norm(self.bottleneck_conv(flat_spikes))
        x_seq = flat_bottleneck.reshape(s_steps, batch, self.bottleneck_channels, l_len)

        # Neuromodulatory FiLM Conditioning: Scale and shift SNN bottleneck based on target voiceprint
        gamma, beta = None, None
        if self.film_proj is not None and spk_embed is not None:
            film_params = self.film_proj(spk_embed)  # [B, 2 * B_conv]
            gamma, beta = film_params.chunk(2, dim=-1)  # [B, B_conv] each
            # Centered at 1.0 with sigmoid gating
            gamma = (torch.sigmoid(gamma) * 2.0).unsqueeze(-1)  # [B, B_conv, 1]
            beta = beta.unsqueeze(-1)                           # [B, B_conv, 1]
            # Broadcast across simulation timesteps S: [S, B, B_conv, L]
            x_seq = x_seq * gamma.unsqueeze(0) + beta.unsqueeze(0)

        # Shared Multi-Scale FiLM generator across all 12 dilation blocks
        shared_gamma, shared_beta = None, None
        if self.shared_film_proj is not None and spk_embed is not None:
            film_params = self.shared_film_proj(spk_embed)  # [B, 2 * B_conv]
            g, b = film_params.chunk(2, dim=-1)
            shared_gamma = (torch.sigmoid(g) * 2.0).unsqueeze(-1)  # [B, B_conv, 1]
            shared_beta = b.unsqueeze(-1)                          # [B, B_conv, 1]

        # Step 2: Pass through stacked spiking dilated conv blocks with multi-scale FiLM
        skip_total = torch.zeros_like(x_seq)
        u_total = torch.zeros_like(x_seq) if self.snn_readout == "membrane" else None
        current_res = x_seq

        for i, block in enumerate(self.blocks):
            # FiLM modulation across dilation blocks:
            if shared_gamma is not None and shared_beta is not None:
                current_res = current_res * shared_gamma.unsqueeze(0) + shared_beta.unsqueeze(0)
            elif self.block_film_projs is not None and spk_embed is not None:
                film_params_i = self.block_film_projs[i](spk_embed)
                gamma_i, beta_i = film_params_i.chunk(2, dim=-1)
                gamma_i = (torch.sigmoid(gamma_i) * 2.0).unsqueeze(-1)
                beta_i = beta_i.unsqueeze(-1)
                current_res = current_res * gamma_i.unsqueeze(0) + beta_i.unsqueeze(0)

            if self.use_checkpointing and self.training and current_res.requires_grad:
                current_res, skip, u_block = checkpoint(block, current_res, use_reentrant=False)
            else:
                current_res, skip, u_block = block(current_res)

            skip_total = skip_total + skip
            if u_total is not None and u_block is not None:
                u_total = u_total + u_block

        total_features = skip_total + current_res

        # Collect SNN block spikes across all 12 blocks for telemetry
        block_spikes = [getattr(b, "last_spikes", None) for b in self.blocks]
        if all(bs is not None for bs in block_spikes):
            self.last_spikes = torch.cat(block_spikes, dim=2)
        else:
            self.last_spikes = None

        # Step 3: Temporal Readout Strategy
        if self.snn_readout == "membrane":
            # Continuous Membrane Potential Readout (High resolution & smooth gradient flow)
            # Combines membrane trajectory with skip features
            readout = torch.mean(u_total + total_features, dim=0)

        elif self.snn_readout in ["weighted_bit", "bit_plane"]:
            # Radix Bit-Weighted Readout (Sum 2^-t * S[t])
            weights = self.bit_weights[:s_steps].to(total_features.device)
            readout = torch.sum(total_features * weights, dim=0)

        else:
            # Standard Rate Readout (Average spikes across S)
            readout = torch.mean(total_features, dim=0)

        # Step 4: Inject Continuous Residual Bridge if configured and continuous_w is supplied
        if self.bridge_conv is not None and continuous_w is not None:
            w_in = continuous_w.to(dtype=self.bridge_conv.weight.dtype, device=spike_seq.device)
            bridge_feats = self.bridge_norm(self.bridge_conv(w_in))
            eff_gamma = gamma if gamma is not None else shared_gamma
            eff_beta = beta if beta is not None else shared_beta
            if eff_gamma is not None and eff_beta is not None:
                bridge_feats = bridge_feats * eff_gamma + eff_beta
            readout = readout + bridge_feats

        # Step 5: Generate masks
        mask_raw = self.mask_conv(readout)

        if self.mask_type == "complex":
            # Reshape to [Batch, Num_Sources, 2 (real, imag), Target_Freq, Latent_Length]
            masks_reshaped = mask_raw.reshape(batch, self.num_sources, 2, self.target_freq, l_len)
            mask_r = self.complex_mask_bound * torch.tanh(masks_reshaped[:, :, 0])
            mask_i = self.complex_mask_bound * torch.tanh(masks_reshaped[:, :, 1])
            masks = torch.stack([mask_r, mask_i], dim=2)  # [B, K, 2, F, L]
        else:
            # Reshape to [B, K, N, L]
            masks = mask_raw.reshape(batch, self.num_sources, self.in_channels, l_len)
            masks = self.activation(masks)

        if verbose:
            print(f"[SpikingTCNSeparator ({self.neuron_type.upper()}/{self.snn_readout})] Spikes in: {spike_seq.shape} -> Masks out ({self.mask_type}): {masks.shape}")

        return masks

