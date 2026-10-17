"""
Configuration settings for SNN Conv-TasNet Speech Separation.
Defines model hyperparameters, SNN neuron dynamics, audio settings, and training parameters.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """
    Configuration for Conv-TasNet (Encoder, Separator, Decoder) and SNN dynamics.
    
    Tensor Dimensions Overview:
        Input Waveform   : [Batch, 1, Time]
        Target Voiceprint: [Batch, 192] (from ECAPA-TDNN)
        Latent W         : [Batch, N, Latent_Length]
        Spike Train      : [Timesteps S, Batch, N, Latent_Length]
        Target Mask      : [Batch, 1, N, Latent_Length]
        Extracted Output : [Batch, 1, Time]
    """
    # Audio Parameters
    sample_rate: int = 8000
    segment_length: float = 2.0  # Duration in seconds (2.0s = 16000 samples @ 8kHz, standard LibriMix)
    
    # STFT Frontend & Audio Architecture:
    encoder_type: str = "spectrogram" # Dedicated STFT Spectrogram Frontend
    encoder_channels: int = 64   # N: Projected latent channel dimension for separator interface
    
    # STFT / Spectrogram Frontend Parameters
    stft_n_fft: int = 256        # FFT size (e.g. 256 = 32ms @ 8kHz -> 129 frequency bins)
    stft_hop_length: int = 64    # STFT hop size (e.g. 64 = 8ms @ 8kHz)
    stft_win_length: int = 256   # Window length (default: matches n_fft)
    stft_window: str = "hann"    # Window type ('hann', 'hamming')
    stft_use_projection: bool = True # 1x1 Conv projection between STFT bins and encoder_channels
    
    # SNN Neuron & Temporal Dynamics
    neuron_type: str = "plif"    # 'plif', 'alif', 'aplif', 'fs_neuron', 'lif'
    snn_timesteps: int = 6       # S: Number of simulation timesteps (6 for high temporal fidelity)
    snn_threshold: float = 0.8   # V_th: Membrane firing threshold (learnable in PLIF/APLIF)
    snn_beta: float = 0.9        # Membrane potential decay factor (learnable in PLIF/APLIF)
    snn_reset_mechanism: str = "subtract"  # 'subtract' (soft) or 'zero' (hard)
    surrogate: str = "fast_sigmoid"        # Surrogate gradient: 'fast_sigmoid', 'atan', 'piecewise'
    
    # Spike Encoding Schemes
    # Options: 'direct_current' (preserves full continuous dynamic range into SNN),
    #          'bit_plane' (high-precision radix-2, 64 levels at S=6),
    #          'learnable_plif', 'population', 'rate', 'threshold'
    spike_encoding: str = "direct_current"
    population_factor: int = 4   # Expansion factor K when using 'population' encoding
    
    # Task Mode & Speaker Conditioning (Target Speech Extraction)
    task_mode: str = "tse"           # Target Speech Extraction ('tse')
    spk_embed_dim: int = 192         # Speaker voiceprint vector dimension from ECAPA-TDNN
    spk_cache_path: str = "./data/minilibrimix_spk_embeddings.pt" # Path to precomputed speaker cache

    # TCN Separator Architecture (Depthwise Dilated Convolutions)
    bottleneck_channels: int = 128   # B: Channels in bottleneck and residual paths (default matches train.py)
    hidden_channels: int = 256       # H: Channels in depthwise convolutional blocks (default matches train.py)
    kernel_size: int = 3             # P: Kernel size of 1-D depthwise convolutions
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])  # Receptive field > 1.2s
    num_repeats: int = 2             # R: Number of times to repeat the dilation stack
    num_sources: int = 1             # K: 1 output target speaker stream for TSE
    mask_activation: str = "sigmoid" # 'sigmoid', 'softmax', 'relu'
    use_checkpointing: bool = True   # Enable gradient checkpointing to reduce activation VRAM by ~80%
    
    # SNN Readout & Residual Bridge
    snn_readout: str = "membrane"    # 'membrane' (smooth continuous), 'weighted_bit', 'rate'
    use_residual_bridge: bool = True # Continuous residual skip connection to mask head
    
    # Loss Regularization Hyperparameters
    mr_stft_weight: float = 0.0      # Multi-resolution STFT auxiliary loss weight (optional spectral loss)
    spike_reg_weight: float = 1e-4   # Spike rate regularization penalty weight
    
    # Debugging
    verbose_shapes: bool = False     # If True, prints tensor shapes during forward pass


@dataclass
class TrainConfig:
    """Training, optimization, and dataset hyperparameters matching train.py defaults."""
    batch_size: int = 4              # Mini-batch size (matches train.py default: 4)
    grad_accum_steps: int = 4        # Gradient accumulation steps (effective batch size: 4 * 4 = 16)
    train_samples_per_epoch: Optional[int] = None # None or 0 to train on all dataset samples (e.g. 13.9k) per epoch
    learning_rate: float = 1.0e-3    # Adam learning rate (1e-3 provides stable SNN surrogate convergence)
    lr_scheduler: str = "cosine"     # 'cosine' or 'plateau'
    patience: int = 10               # Patience for plateau scheduler
    min_lr: float = 1e-5             # Minimum learning rate
    weight_decay: float = 1e-5       # L2 regularization
    epochs: int = 70                 # Number of training epochs
    clip_grad: float = 5.0           # Maximum gradient norm for clipping
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_threads: int = 6             # Multi-threading for CPU fallback execution
    
    # CUDA & GPU Acceleration Settings
    use_amp: bool = True             # PyTorch Automatic Mixed Precision (torch.amp FP16)
    pin_memory: Optional[bool] = None # Pin memory for faster CPU-to-GPU memory copies (auto-detected)
    non_blocking: bool = True        # Asynchronous tensor transfer to GPU
    cudnn_benchmark: bool = True     # Fast cuDNN kernel selection for fixed-size convolutions
    num_workers: int = 0             # DataLoader workers (0 for stability/container safety)
    
    # Dataset & Task settings
    task_mode: str = "tse"               # Target Speech Extraction
    spk_cache_path: str = "./data/minilibrimix_spk_embeddings.pt" # Path to precomputed speaker cache
    dataset_type: str = "librimix"       # 'librimix', 'mini_librimix', or 'wav_folder'
    data_dir: str = "./data/Libri2Mix"   # Path to Libri2Mix (or ./data/MiniLibriMix)
    mixture_type: str = "mix_clean"      # 'mix_clean' (clean speech) or 'mix_both' (noisy speech)
    
    # Paths & Logging
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    seed: int = 42

