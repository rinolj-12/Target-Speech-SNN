"""
Training Script for SNN Conv-TasNet on MiniLibriMix.

Features:
1. Full NVIDIA CUDA & multi-GPU device support with automatic fallback.
2. PyTorch Automatic Mixed Precision (torch.amp FP16) & GradScaler for fast GPU training.
3. Supports PLIF, ALIF, APLIF, FS-Neuron, and LIF neuron dynamics.
4. Supports Bit-Plane, Population, Learnable PLIF, and Direct spike encodings.
5. Continuous Membrane Readout & Continuous Residual Bridge.
6. Composite Target Speech Extraction Loss (Direct Target SI-SDR + Multi-Resolution STFT Auxiliary Loss).
7. Learning rate scheduling, gradient clipping, VRAM telemetry, and validation tracking.
8. Saves best model checkpoints with full metadata and configuration.
"""

from typing import Optional, Tuple
import os
import time
import argparse

# Configure PyTorch CUDA allocator to prevent memory fragmentation during SNN unrolling
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from config import ModelConfig, TrainConfig
from model import SpikingConvTasNet
from losses import calculate_sisdr, TSECombinedLoss, TargetSISDRLoss
from dataset import get_dataloaders


def resolve_device(device_arg: str) -> torch.device:
    """Resolves device string to a valid torch.device with CUDA fallback."""
    if device_arg in ["auto", "cuda"]:
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            torch.cuda.set_device(dev)
            return dev
        return torch.device("cpu")
    elif device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[Warning] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        dev = torch.device(device_arg)
        torch.cuda.set_device(dev)
        return dev
    else:
        return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = True,
    clip_grad: float = 5.0,
    epoch: int = 1,
    total_epochs: int = 70,
    max_batches: Optional[int] = None,
    grad_accum_steps: int = 1,
) -> float:
    """Trains the model for one epoch with optional AMP mixed-precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_sisdr = 0.0
    total_sisdr_i = 0.0
    total_batches = len(train_loader) if max_batches is None else min(len(train_loader), max_batches)
    amp_enabled = use_amp and (device.type == "cuda")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        if max_batches and batch_idx >= max_batches:
            break

        if len(batch) == 3:
            mix_batch, target_batch, spk_embed = batch
            spk_embed = spk_embed.to(device, non_blocking=True)
        else:
            mix_batch, target_batch = batch
            spk_embed = None

        mix_batch = mix_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)

        # Forward pass under AMP Autocast
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            if spk_embed is not None:
                separated = model(mix_batch, spk_embed=spk_embed)
            else:
                separated = model(mix_batch)

            loss, best_estimates, sisdr_score = criterion(separated, target_batch)
            # Scale loss for gradient accumulation
            loss_scaled = loss / grad_accum_steps

        # Backward pass & gradient clipping with Scaler
        if scaler is not None and amp_enabled:
            scaler.scale(loss_scaled).backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_scaled.backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()

        # SI-SDRi = SI-SDR(estimate, source) - SI-SDR(mixture, source)
        with torch.no_grad():
            mix_repeated = mix_batch.expand_as(target_batch)
            mixture_sisdr = calculate_sisdr(
                mix_repeated.reshape(-1, mix_repeated.shape[-1]).float(),
                target_batch.reshape(-1, target_batch.shape[-1]).float(),
            ).mean()
            sisdr_mean = sisdr_score.mean() if isinstance(sisdr_score, torch.Tensor) else sisdr_score
            sisdr_i = sisdr_mean - mixture_sisdr

        total_loss += loss.item()
        total_sisdr += sisdr_mean.item() if isinstance(sisdr_mean, torch.Tensor) else sisdr_mean
        total_sisdr_i += sisdr_i.item() if isinstance(sisdr_i, torch.Tensor) else sisdr_i

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * mix_batch.size(0) / max(elapsed, 1e-4)
            gpu_mem_str = ""
            if device.type == "cuda":
                max_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                res_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                gpu_mem_str = f" | Peak VRAM: {max_alloc_mb:.0f}/{res_mb:.0f} MB"

            print(
                f"  Epoch [{epoch:02d}/{total_epochs:02d}] | "
                f"Batch [{batch_idx+1:03d}/{total_batches:03d}] | "
                f"Loss: {loss.item():6.2f} | "
                f"SI-SDR: {(sisdr_mean.item() if isinstance(sisdr_mean, torch.Tensor) else sisdr_mean):6.2f} dB | "
                f"SI-SDRi: {(sisdr_i.item() if isinstance(sisdr_i, torch.Tensor) else sisdr_i):6.2f} dB | "
                f"Speed: {samples_per_sec:.1f} samples/s"
                f"{gpu_mem_str}",
                flush=True,
            )

    return total_sisdr / max(total_batches, 1), total_sisdr_i / max(total_batches, 1)


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    max_batches: Optional[int] = None,
) -> float:
    """Evaluates model on validation set and returns mean SI-SDR (dB)."""
    model.eval()
    val_sisdr_list = []
    amp_enabled = use_amp and (device.type == "cuda")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches and batch_idx >= max_batches:
                break

            if len(batch) == 3:
                mix_batch, target_batch, spk_embed = batch
                spk_embed = spk_embed.to(device, non_blocking=True)
            else:
                mix_batch, target_batch = batch
                spk_embed = None

            mix_batch = mix_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                if spk_embed is not None:
                    separated = model(mix_batch, spk_embed=spk_embed)
                else:
                    separated = model(mix_batch)
                _, _, sisdr_score = criterion(separated, target_batch)
            sisdr_val = sisdr_score.mean().item() if isinstance(sisdr_score, torch.Tensor) else float(sisdr_score)
            val_sisdr_list.append(sisdr_val)

    return float(np.mean(val_sisdr_list)) if val_sisdr_list else 0.0


def main():
    default_model_cfg = ModelConfig()
    default_train_cfg = TrainConfig()

    parser = argparse.ArgumentParser(description="Train SNN Speech Separation Model with STFT Frontend on Libri2Mix / MiniLibriMix")
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="spectrogram",
        choices=["spectrogram", "stft"],
        help="Encoder frontend: 'spectrogram' (STFT magnitude + phase preservation)",
    )
    parser.add_argument(
        "--neuron_type",
        type=str,
        default=default_model_cfg.neuron_type,
        choices=["plif", "alif", "aplif", "fs_neuron", "lif"],
        help="SNN neuron model ('plif', 'alif', 'aplif', 'fs_neuron', 'lif')",
    )
    parser.add_argument(
        "--spike_encoding",
        type=str,
        default=default_model_cfg.spike_encoding,
        choices=["direct_current", "bit_plane", "learnable_plif", "fs_neuron", "population", "rate", "threshold"],
        help="Spike encoding scheme for continuous latent W",
    )
    parser.add_argument(
        "--snn_readout",
        type=str,
        default=default_model_cfg.snn_readout,
        choices=["membrane", "weighted_bit", "rate"],
        help="SNN temporal mask readout strategy ('membrane', 'weighted_bit', 'rate')",
    )
    parser.add_argument(
        "--use_residual_bridge",
        dest="use_residual_bridge",
        action="store_true",
        default=default_model_cfg.use_residual_bridge,
        help="Enable continuous residual bridge to mask generator head",
    )
    parser.add_argument(
        "--no_residual_bridge",
        dest="use_residual_bridge",
        action="store_false",
        help="Disable continuous residual bridge (pure SNN masking)",
    )
    parser.add_argument("--mr_stft_weight", type=float, default=default_model_cfg.mr_stft_weight, help="Multi-Resolution STFT auxiliary loss weight")
    parser.add_argument("--population_factor", type=int, default=default_model_cfg.population_factor, help="Gaussian population size when using population coding")
    parser.add_argument("--stft_n_fft", type=int, default=default_model_cfg.stft_n_fft, help="STFT N_FFT size when using spectrogram encoder")
    parser.add_argument("--stft_hop_length", type=int, default=default_model_cfg.stft_hop_length, help="STFT hop length when using spectrogram encoder")
    parser.add_argument("--encoder_channels", type=int, default=default_model_cfg.encoder_channels, help="Number of latent channels N")
    parser.add_argument("--stft_use_projection", dest="stft_use_projection", action="store_true", default=default_model_cfg.stft_use_projection, help="Enable 1x1 Conv projection between STFT bins and encoder_channels")
    parser.add_argument("--no_stft_projection", dest="stft_use_projection", action="store_false", help="Disable projection: direct 1:1 STFT frequency bin masking")
    parser.add_argument(
        "--task_mode",
        type=str,
        default=default_model_cfg.task_mode,
        help="Task mode: 'tse' (Target Speech Extraction)",
    )
    parser.add_argument(
        "--spk_cache_path",
        type=str,
        default=default_model_cfg.spk_cache_path,
        help="Path to precomputed speaker embedding cache",
    )
    parser.add_argument("--device", type=str, default=default_train_cfg.device, help="Device: 'auto', 'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument("--amp", dest="use_amp", action="store_true", default=default_train_cfg.use_amp, help="Enable PyTorch FP16 Automatic Mixed Precision")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable PyTorch AMP (use standard FP32)")
    parser.add_argument("--checkpointing", dest="use_checkpointing", action="store_true", default=default_model_cfg.use_checkpointing, help="Enable gradient checkpointing (saves ~80% activation VRAM)")
    parser.add_argument("--no_checkpointing", dest="use_checkpointing", action="store_false", help="Disable gradient checkpointing")
    parser.add_argument("--multi_gpu", dest="multi_gpu", action="store_true", default=False, help="Enable DataParallel across multiple GPUs")
    parser.add_argument("--mask_activation", type=str, default=default_model_cfg.mask_activation, choices=["sigmoid", "softmax", "relu"], help="Mask activation function")
    parser.add_argument("--segment_length", type=float, default=default_model_cfg.segment_length, help="Audio segment duration in seconds")
    parser.add_argument("--epochs", type=int, default=default_train_cfg.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=default_train_cfg.batch_size, help="Training batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=default_train_cfg.grad_accum_steps, help="Gradient accumulation steps")
    parser.add_argument("--train_samples_per_epoch", type=int, default=0, help="Virtual samples per epoch (0 for full dataset)")
    parser.add_argument("--mixture_type", type=str, default=default_train_cfg.mixture_type, choices=["mix_both", "mix_clean"], help="Mixture condition: 'mix_both' or 'mix_clean'")
    parser.add_argument("--lr", type=float, default=default_train_cfg.learning_rate, help="Learning rate for Adam optimizer")
    parser.add_argument("--lr_scheduler", type=str, default=default_train_cfg.lr_scheduler, choices=["cosine", "plateau"], help="Learning rate scheduler")
    parser.add_argument("--patience", type=int, default=default_train_cfg.patience, help="Epoch patience before decaying LR when using 'plateau' scheduler")
    parser.add_argument("--min_lr", type=float, default=default_train_cfg.min_lr, help="Minimum learning rate for scheduler")
    parser.add_argument("--weight_decay", type=float, default=default_train_cfg.weight_decay, help="Weight decay regularization")
    parser.add_argument("--snn_timesteps", type=int, default=default_model_cfg.snn_timesteps, help="SNN simulation timesteps S")
    parser.add_argument("--snn_beta", type=float, default=default_model_cfg.snn_beta, help="LIF membrane potential decay factor")
    parser.add_argument("--surrogate", type=str, default=default_model_cfg.surrogate, choices=["fast_sigmoid", "atan", "piecewise"], help="Surrogate gradient function")
    parser.add_argument("--num_repeats", type=int, default=default_model_cfg.num_repeats, help="Number of dilation stack repeats R")
    parser.add_argument("--bottleneck_channels", type=int, default=default_model_cfg.bottleneck_channels, help="Number of bottleneck channels B")
    parser.add_argument("--hidden_channels", type=int, default=default_model_cfg.hidden_channels, help="Number of hidden channels H in depthwise blocks")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=default_train_cfg.dataset_type,
        choices=["librimix", "mini_librimix", "wav_folder"],
        help="Dataset type: 'librimix', 'mini_librimix', or 'wav_folder'",
    )
    parser.add_argument("--data_dir", type=str, default=default_train_cfg.data_dir, help="Path to Libri2Mix (or MiniLibriMix) dataset")
    parser.add_argument("--checkpoint_dir", type=str, default=default_train_cfg.checkpoint_dir, help="Directory to save model checkpoints")
    parser.add_argument("--num_workers", type=int, default=default_train_cfg.num_workers, help="DataLoader workers")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=default_train_cfg.pin_memory, help="Pin memory for faster host-to-device transfers")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Limit number of training batches per epoch (optional)")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Limit number of validation batches (optional)")
    parser.add_argument("--num_threads", type=int, default=default_train_cfg.num_threads, help="Number of CPU threads for PyTorch fallback")
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume training from checkpoint path or 'auto' for best checkpoint")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = resolve_device(args.device)

    # Resolve STFT encoder channels & projection
    n_freq = args.stft_n_fft // 2 + 1
    if args.stft_use_projection is False:
        enc_channels = n_freq
        use_proj = False
    elif args.encoder_channels is not None:
        enc_channels = args.encoder_channels
        use_proj = (enc_channels != n_freq) if args.stft_use_projection is None else args.stft_use_projection
    else:
        enc_channels = n_freq
        use_proj = False

    # CUDA / Hardware Optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(device)
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        cuda_cap = torch.cuda.get_device_capability(device)
    else:
        torch.set_num_threads(args.num_threads)
        gpu_name = "N/A (CPU Mode)"
        total_vram_gb = 0.0
        cuda_cap = (0, 0)

    proj_str = "with 1x1 Conv proj" if use_proj else "direct 1:1 frequency masking"
    enc_desc = f"STFT Spectrogram (n_fft={args.stft_n_fft}, hop={args.stft_hop_length}, bins={n_freq}, {proj_str})"

    print("=" * 68)
    print(f"       Training SNN Conv-TasNet on {args.dataset_type.upper()}")
    print("=" * 68)
    print(f"- Task Mode:      {args.task_mode.upper()} ({'Target Speech Extraction' if args.task_mode == 'tse' else 'Blind Source Separation'})")
    print(f"- Output Streams:  {1 if args.task_mode == 'tse' else 2} ('Single target mask' if args.task_mode == 'tse' else '2 speaker masks')")
    print(f"- Dataset:         {args.dataset_type} ({args.data_dir})")
    print(f"- Device:          {device} ({gpu_name})")
    if device.type == "cuda":
        print(f"- VRAM Capacity:   {total_vram_gb:.2f} GB | Compute Capability: {cuda_cap[0]}.{cuda_cap[1]}")
        print(f"- CUDA AMP (FP16): {'Enabled (torch.amp.autocast)' if args.use_amp else 'Disabled (FP32)'}")
        print(f"- cuDNN Benchmark: Enabled (Fast kernel selection)")
    else:
        print(f"- CPU Threads:     {torch.get_num_threads()}")
    print(f"- Model Type:      SNN")
    print(f"- Encoder Type:    {args.encoder_type.upper()} ({enc_desc})")
    print(f"- Encoder Channels:{enc_channels}")
    print(f"- Mask Activation: {args.mask_activation}")
    print(f"- Segment Length:  {args.segment_length:.1f}s ({int(args.segment_length * 8000)} samples @ 8kHz)")
    print(f"- Epochs:          {args.epochs}")
    print(f"- Batch Size:      {args.batch_size} (Grad Accum: {args.grad_accum_steps}x -> Effective Batch Size: {args.batch_size * args.grad_accum_steps})")
    sample_exposure_str = "Full dataset (all utterances / epoch)" if args.train_samples_per_epoch <= 0 else f"{args.train_samples_per_epoch} / epoch (dynamic random crops)"
    print(f"- Mixture Type:    {args.mixture_type}")
    print(f"- Train Samples:   {sample_exposure_str}")
    print(f"- Learning Rate:   {args.lr} (Scheduler: {args.lr_scheduler.upper()})")
    print(f"- Num Repeats (R): {args.num_repeats}")
    print(f"- Channels (B/H):  {args.bottleneck_channels} / {args.hidden_channels}")
    print(f"- Neuron Model:    {args.neuron_type.upper()}")
    print(f"- Spike Encoding:  {args.spike_encoding}")
    print(f"- SNN Readout:     {args.snn_readout}")
    print(f"- Residual Bridge: {args.use_residual_bridge}")
    print(f"- Checkpointing:   {'Enabled (saves ~80% activation VRAM)' if args.use_checkpointing else 'Disabled'}")
    print(f"- SNN Timesteps S: {args.snn_timesteps}")
    print(f"- SNN Decay Beta:  {args.snn_beta}")
    print(f"- Surrogate Grad:  {args.surrogate}")
    print(f"- MR-STFT Weight:  {args.mr_stft_weight}")

    # Build Model Configuration
    model_config = ModelConfig(
        task_mode=args.task_mode,
        num_sources=1 if args.task_mode == "tse" else 2,
        spk_embed_dim=192 if args.task_mode == "tse" else None,
        spk_cache_path=args.spk_cache_path,
        sample_rate=8000,
        segment_length=args.segment_length,
        encoder_type=args.encoder_type,
        encoder_channels=enc_channels,
        stft_n_fft=args.stft_n_fft,
        stft_hop_length=args.stft_hop_length,
        stft_use_projection=use_proj,
        mask_activation=args.mask_activation,
        neuron_type=args.neuron_type,
        spike_encoding=args.spike_encoding,
        population_factor=args.population_factor,
        snn_readout=args.snn_readout,
        use_residual_bridge=args.use_residual_bridge,
        mr_stft_weight=args.mr_stft_weight,
        snn_timesteps=args.snn_timesteps,
        snn_beta=args.snn_beta,
        surrogate=args.surrogate,
        num_repeats=args.num_repeats,
        bottleneck_channels=args.bottleneck_channels,
        hidden_channels=args.hidden_channels,
        use_checkpointing=args.use_checkpointing,
    )

    # --- DataLoaders FIRST (before model.to(device)) ---
    # IMPORTANT: DataLoader workers are forked from the current process. If CUDA
    # is initialized (e.g. via model.to('cuda')) before fork(), the forked worker
    # processes inherit a corrupt CUDA context and crash with SIGSEGV in
    # libtorch_python.so. Creating DataLoaders before touching CUDA avoids this.
    #
    # Additionally, when running inside Docker/Kaggle containers, /dev/shm is
    # often very small (64 MB), causing pin_memory IPC to OOM in workers.
    # We auto-detect this and force num_workers=0 + pin_memory=False.
    safe_num_workers = args.num_workers
    pin_mem = args.pin_memory if args.pin_memory is not None else (device.type == "cuda")
    if device.type == "cuda" and safe_num_workers > 0:
        try:
            import shutil
            shm_free_mb = shutil.disk_usage("/dev/shm").free / (1024 ** 2)
            if shm_free_mb < 512:
                print(f"[Warning] /dev/shm has only {shm_free_mb:.0f} MB free (< 512 MB). "
                      f"Forcing num_workers=0 and pin_memory=False to prevent DataLoader worker SIGSEGV.")
                safe_num_workers = 0
                pin_mem = False
        except OSError:
            pass  # /dev/shm doesn't exist on this platform — leave as-is

    print(f"\n[1/3] Loading {args.dataset_type.upper()} dataset from: {args.data_dir} (condition: {args.mixture_type}) ...")
    effective_train_samples = args.train_samples_per_epoch if args.train_samples_per_epoch > 0 else None
    train_loader, val_loader = get_dataloaders(
        dataset_type=args.dataset_type,
        sample_rate=model_config.sample_rate,
        segment_length=model_config.segment_length,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        mixture_type=args.mixture_type,
        num_workers=safe_num_workers,
        pin_memory=pin_mem,
        train_samples_per_epoch=effective_train_samples,
        task_mode=args.task_mode,
        spk_cache_path=args.spk_cache_path,
    )
    print(f"  - Train: {len(train_loader.dataset)} samples across {len(train_loader)} batches (pin_memory={pin_mem}, num_workers={safe_num_workers})")
    print(f"  - Val:   {len(val_loader.dataset)} samples across {len(val_loader)} batches")

    # Initialize Model (AFTER DataLoaders — keeps CUDA uninit'd during fork)
    model = SpikingConvTasNet(model_config)

    # Device & Multi-GPU Configuration (Single-GPU is recommended for stability)
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    is_multi_gpu = args.multi_gpu and (num_gpus > 1)
    if is_multi_gpu:
        print(f"- Multi-GPU:       Active ({num_gpus} GPUs detected: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]})")
        if args.use_checkpointing:
            print("[Warning] nn.DataParallel is incompatible with activation checkpointing on systems without NVLink/P2P.")
            print("          Disabling gradient checkpointing to prevent CUDA illegal memory access.")
            model_config.use_checkpointing = False
            model = SpikingConvTasNet(model_config)
        model = nn.DataParallel(model)
    elif device.type == "cuda":
        gpu_detail = f"{gpu_name} (device: {device})"
        if num_gpus > 1:
            gpu_detail += f" [Single-GPU mode active; {num_gpus} total GPUs available]"
        print(f"- Accelerator:     Single GPU ({gpu_detail})")

    model.to(device)
    raw_model = model.module if is_multi_gpu else model
    total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"- Trainable Params:{total_params:,}")

    # Loss Function Configuration (Target Speech Extraction)
    criterion = TSECombinedLoss(mr_stft_weight=args.mr_stft_weight)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.patience, min_lr=args.min_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=args.use_amp and (device.type == "cuda")) \
        if (args.use_amp and device.type == "cuda") else None

    checkpoint_name = f"best_snn_{args.task_mode}_{args.encoder_type}_{args.neuron_type}.pt"
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)

    start_epoch = 1
    best_val_sisdr = -float("inf")

    # Resume from checkpoint if requested
    if args.resume:
        resume_target = checkpoint_path if args.resume == "auto" else args.resume
        if os.path.exists(resume_target):
            print(f"[Resume] Loading checkpoint from: {resume_target}")
            ckpt = torch.load(resume_target, map_location=device, weights_only=False)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"[Resume] Warning: Could not restore optimizer state ({e}). Using fresh optimizer.")
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_sisdr = ckpt.get("val_sisdr", -float("inf"))
            print(f"[Resume] Resumed successfully. Starting at Epoch {start_epoch}, previous best Val SI-SDR: {best_val_sisdr:.2f} dB")
        else:
            print(f"[Resume] Warning: Checkpoint '{resume_target}' not found. Starting from scratch.")

    print(f"\n[2/3] Starting Training from Epoch {start_epoch} to {args.epochs} on {device}...")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_sisdr, train_sisdr_i = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=args.use_amp,
            clip_grad=5.0,
            epoch=epoch,
            total_epochs=args.epochs,
            max_batches=args.max_train_batches,
            grad_accum_steps=args.grad_accum_steps,
        )

        val_sisdr = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=args.use_amp,
            max_batches=args.max_val_batches,
        )

        epoch_time = time.time() - epoch_start
        if args.lr_scheduler == "plateau":
            scheduler.step(val_sisdr)
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} Summary [{epoch_time:.1f}s] | "
            f"Train SI-SDR: {train_sisdr:6.2f} dB | Train SI-SDRi: {train_sisdr_i:6.2f} dB | "
            f"Val SI-SDR: {val_sisdr:6.2f} dB | "
            f"LR: {current_lr:.6f}",
            flush=True,
        )

        # Save Best Model Checkpoint
        if val_sisdr > best_val_sisdr:
            best_val_sisdr = val_sisdr
            checkpoint_data = {
                "epoch": epoch,
                "model_type": "snn",
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model_config,
                "val_sisdr": val_sisdr,
                "device": str(device),
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  [*] Saved new best checkpoint to '{checkpoint_path}' (Val SI-SDR: {val_sisdr:.2f} dB)")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 68)
    print(f"[3/3] Training Completed! Best Validation SI-SDR: {best_val_sisdr:.2f} dB")
    print(f"      Best Model saved at: {checkpoint_path}")
    print("=" * 68)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
