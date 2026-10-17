"""
Spiking Separator Modules for Conv-TasNet.

Implements:
1. SpikingTCNSeparator: Dilated Spiking Temporal Convolutional Network with selectable neuron dynamics
   (PLIF, ALIF, APLIF, FS-Neuron, LIF), continuous membrane potential readout, and continuous residual bridge.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple
from snn import SpikingConvBlock1d


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
    """

    def __init__(
        self,
        in_channels: int = 64,
        bottleneck_channels: int = 64,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        num_repeats: int = 2,
        num_sources: int = 2,
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

        # Neuromodulatory FiLM Conditioning Projection for Target Speech Extraction
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.film_proj = nn.Linear(spk_embed_dim, 2 * bottleneck_channels)
            nn.init.zeros_(self.film_proj.bias)
            nn.init.normal_(self.film_proj.weight, std=0.02)
        else:
            self.film_proj = None

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

        # Output mask generator: [B_conv -> K * N]
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

        # Step 1: Apply bottleneck 1x1 conv to each simulation timestep
        flat_spikes = spike_seq.reshape(s_steps * batch, n_chan, l_len)
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

        # Step 2: Pass through stacked spiking dilated conv blocks
        skip_total = torch.zeros_like(x_seq)
        u_total = torch.zeros_like(x_seq) if self.snn_readout == "membrane" else None
        current_res = x_seq

        for block in self.blocks:
            if self.use_checkpointing and self.training and current_res.requires_grad:
                current_res, skip, u_block = checkpoint(block, current_res, use_reentrant=False)
            else:
                current_res, skip, u_block = block(current_res)

            skip_total = skip_total + skip
            if u_total is not None and u_block is not None:
                u_total = u_total + u_block

        total_features = skip_total + current_res

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
            bridge_feats = self.bridge_norm(self.bridge_conv(continuous_w))
            if gamma is not None and beta is not None:
                bridge_feats = bridge_feats * gamma + beta
            readout = readout + bridge_feats

        # Step 5: Generate masks: [B, B_conv, L] -> [B, K * N, L]
        mask_raw = self.mask_conv(readout)

        # Reshape to [B, K, N, L]
        masks = mask_raw.reshape(batch, self.num_sources, self.in_channels, l_len)

        masks = self.activation(masks)

        if verbose:
            print(f"[SpikingTCNSeparator ({self.neuron_type.upper()}/{self.snn_readout})] Spikes in: {spike_seq.shape} -> Masks out: {masks.shape}")

        return masks

