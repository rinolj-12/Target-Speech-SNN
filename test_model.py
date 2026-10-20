"""
Model Testing, Multi-Speaker Evaluation, and Inference Suite for Spiking Target Speech Extraction.

Performs:
1. Benchmark evaluation on MiniLibriMix validation/test dataset.
2. Metrics computation: Input SI-SDR, Output SI-SDR, SI-SNRi (Improvement), and Latency.
3. Multi-Speaker Testing: Dynamically mixes 3 or 4 speakers and extracts strictly the selected target voice.
4. Custom audio evaluation: Supports passing custom mixed audio or individual speaker WAVs with target reference.
5. Generates separated WAV audio and categorized diagnostic PNG visualizations.
"""

import os
import sys
import argparse
import time
from typing import Tuple, Optional, List
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from config import ModelConfig, TrainConfig
from model import SpikingConvTasNet
from losses import TargetSISDRLoss, calculate_sisdr
from dataset import get_dataloaders
from speaker_encoder import SpeakerEncoder, get_speaker_cache_manager
from visualize import (
    save_separate_waveforms,
    save_separate_spectrograms,
    save_separate_mask_plots,
    save_separate_snn_raster_plots,
)


def get_encoder_folder_info(encoder_type: str = "spectrogram", base_output_dir: str = "./test_outputs") -> Tuple[str, str]:
    """Resolves the canonical subfolder name and path for evaluation outputs."""
    folder_name = "spectrogram"
    target_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    return folder_name, target_dir


def load_model(checkpoint_path: str, device: torch.device = torch.device("cpu")) -> Tuple[nn.Module, ModelConfig, dict]:
    """Loads model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_cfg = checkpoint.get("model_config", checkpoint.get("model_config_dict", None))
    if isinstance(raw_cfg, dict):
        config = ModelConfig(**{k: v for k, v in raw_cfg.items() if hasattr(ModelConfig, k)})
    elif isinstance(raw_cfg, ModelConfig):
        config = raw_cfg
    else:
        config = ModelConfig()

    # Auto-detect mask_type directly from the saved mask_conv weight shape
    state_dict = checkpoint.get("model_state_dict", {})
    mask_weight = state_dict.get("separator.mask_conv.weight")
    if mask_weight is not None:
        if mask_weight.shape[0] == config.num_sources * config.encoder_channels:
            config.mask_type = "magnitude"
        elif mask_weight.shape[0] == 2 * config.num_sources * (config.stft_n_fft // 2 + 1):
            config.mask_type = "complex"

    # Auto-detect Phase 2/3 modules (speaker adapter and deep/shared FiLM) if not explicitly defined
    if not hasattr(raw_cfg, "use_speaker_adapter") and (not isinstance(raw_cfg, dict) or "use_speaker_adapter" not in raw_cfg):
        config.use_speaker_adapter = any("separator.speaker_adapter" in k for k in state_dict.keys())
    if not hasattr(raw_cfg, "film_mode") and (not isinstance(raw_cfg, dict) or "film_mode" not in raw_cfg):
        has_deep_film = any("separator.block_film_projs" in k for k in state_dict.keys())
        has_shared_film = any("separator.shared_film_proj" in k for k in state_dict.keys())
        if has_shared_film:
            config.film_mode = "shared_deep"
        elif has_deep_film:
            config.film_mode = "deep"
        else:
            config.film_mode = "bottleneck"

    model = SpikingConvTasNet(config)

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as e:
        print(f"[Warning] Exact state_dict key match failed: {e}\nAttempting load with strict=False...")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model.to(device)
    model.eval()
    return model, config, checkpoint


def resolve_device(device_arg: str) -> torch.device:
    """Resolves device string to a valid torch.device with CUDA fallback."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    elif device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[Warning] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_arg)
    else:
        return torch.device("cpu")


def evaluate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    task_mode: str = "tse",
    max_batches: Optional[int] = None,
) -> dict:
    """Evaluates model over validation data loader and computes SI-SDR, SI-SNRi, and accurate latency."""
    model.eval()
    total_output_sisdr = []
    total_input_sisdr = []
    latencies = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break

            if len(batch) == 3:
                mix_batch, target_batch, spk_embed = batch
                spk_embed = spk_embed.to(device, non_blocking=True)
            else:
                mix_batch, target_batch = batch
                spk_embed = None

            mix_batch = mix_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if spk_embed is not None:
                est_sources = model(mix_batch, spk_embed=spk_embed)
            else:
                est_sources = model(mix_batch)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0 / mix_batch.size(0))

            # Target Speech Extraction metric calculation
            if target_batch.dim() == 3 and target_batch.size(1) > 1 and est_sources.size(1) == 1:
                target_eval = target_batch[:, 0:1]
            else:
                target_eval = target_batch

            out_sisdr = calculate_sisdr(est_sources, target_eval)
            in_sisdr = calculate_sisdr(mix_batch, target_eval)

            out_val = out_sisdr.cpu().tolist()
            if isinstance(out_val, list):
                total_output_sisdr.extend(out_val)
            else:
                total_output_sisdr.append(out_val)

            in_val = in_sisdr.cpu().tolist()
            if isinstance(in_val, list):
                total_input_sisdr.extend(in_val)
            else:
                total_input_sisdr.append(in_val)

    out_arr = np.array(total_output_sisdr)
    in_arr = np.array(total_input_sisdr)
    sisdri_arr = out_arr - in_arr

    results = {
        "mean_output_sisdr": float(np.mean(out_arr)),
        "std_output_sisdr": float(np.std(out_arr)),
        "median_output_sisdr": float(np.median(out_arr)),
        "min_output_sisdr": float(np.min(out_arr)),
        "max_output_sisdr": float(np.max(out_arr)),
        "mean_input_sisdr": float(np.mean(in_arr)),
        "mean_sisdri": float(np.mean(sisdri_arr)),
        "avg_latency_per_sample_ms": float(np.mean(latencies)),
    }
    return results


def run_sample_separation(
    model: nn.Module,
    config: ModelConfig,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str = "./test_outputs",
):
    """
    Runs dual target extraction on a representative mixture:
    Extracts Speaker 1 using Vector 1, and Speaker 2 using Vector 2 from the exact same mixture.
    """
    enc_type = getattr(config, "encoder_type", "spectrogram")
    _, enc_output_dir = get_encoder_folder_info(enc_type, output_dir)
    os.makedirs(enc_output_dir, exist_ok=True)

    batch = next(iter(val_loader))
    dataset = val_loader.dataset

    if hasattr(dataset, "pairs") and len(dataset.pairs) > 0:
        p_mix, p_s1, p_s2 = dataset.pairs[0]
        s1_full, _ = sf.read(p_s1)
        s2_full, _ = sf.read(p_s2)
        if s1_full.ndim > 1:
            s1_full = s1_full.mean(axis=-1)
        if s2_full.ndim > 1:
            s2_full = s2_full.mean(axis=-1)
        mix_full, sr = sf.read(p_mix) if p_mix and os.path.exists(p_mix) else (s1_full[:len(s2_full)] + s2_full[:len(s1_full)], config.sample_rate)
        if mix_full.ndim > 1:
            mix_full = mix_full.mean(axis=-1)

        common_len = min(len(s1_full), len(s2_full), len(mix_full), int(config.segment_length * config.sample_rate))
        s1_clip = s1_full[:common_len]
        s2_clip = s2_full[:common_len]
        mix_clip = mix_full[:common_len]

        max_v = max(np.max(np.abs(mix_clip)), 1e-7)
        mix_norm = mix_clip / max_v
        s1_norm = s1_clip / max_v
        s2_norm = s2_clip / max_v

        x = torch.from_numpy(mix_norm).unsqueeze(0).unsqueeze(0).float().to(device)

        # Extract embeddings for Speaker 1 and Speaker 2
        f1 = os.path.basename(p_s1).replace(".wav", "").replace(".flac", "")
        parts1 = f1.split("_")
        spk1_id = parts1[0].split("-")[0]
        spk2_id = parts1[1].split("-")[0] if len(parts1) > 1 else os.path.basename(p_s2).split("-")[0]

        cache_mgr = getattr(dataset, "cache_manager", None)
        if cache_mgr is None:
            cache_mgr = get_speaker_cache_manager(device=str(device))

        emb1 = cache_mgr.get_embedding(spk1_id, reference_audio_path=p_s1).unsqueeze(0).to(device)
        emb2 = cache_mgr.get_embedding(spk2_id, reference_audio_path=p_s2).unsqueeze(0).to(device)
    else:
        # Fallback for custom / WavFolder datasets without .pairs attribute
        sr = getattr(dataset, "sample_rate", config.sample_rate)
        if len(batch) == 3:
            mix_batch, target_batch, spk_embed = batch
            emb1 = spk_embed[0:1].to(device) if spk_embed is not None else torch.randn(1, config.spk_embed_dim, device=device)
            emb2 = -emb1
        else:
            mix_batch, target_batch = batch
            emb1 = torch.randn(1, config.spk_embed_dim, device=device)
            emb2 = -emb1

        mix_norm = mix_batch[0, 0].cpu().numpy()
        s1_norm = target_batch[0, 0].cpu().numpy()
        s2_norm = mix_norm - s1_norm
        x = mix_batch[0:1].to(device)

    audio_dir = os.path.join(enc_output_dir, "audio")
    waveform_dir = os.path.join(enc_output_dir, "waveforms")
    spec_dir = os.path.join(enc_output_dir, "spectrograms")
    mask_dir = os.path.join(enc_output_dir, "masks")
    snn_dir = os.path.join(enc_output_dir, "snn")

    for d in [audio_dir, waveform_dir, spec_dir, mask_dir, snn_dir]:
        os.makedirs(d, exist_ok=True)

    with torch.no_grad():
        # Pass 1: Extract Speaker 1 using Vector 1 + intermediates
        est1_out, mask1, spike_seq1 = model(x, spk_embed=emb1, return_intermediates=True)
        est1 = est1_out[0, 0].cpu().numpy()

        # Pass 2: Extract Speaker 2 using Vector 2 from the EXACT same mixture!
        est2_out, mask2, _ = model(x, spk_embed=emb2, return_intermediates=True)
        est2 = est2_out[0, 0].cpu().numpy()

    # Metrics
    in_sdr1 = calculate_sisdr(torch.from_numpy(mix_norm), torch.from_numpy(s1_norm)).item()
    out_sdr1 = calculate_sisdr(torch.from_numpy(est1), torch.from_numpy(s1_norm)).item()
    in_sdr2 = calculate_sisdr(torch.from_numpy(mix_norm), torch.from_numpy(s2_norm)).item()
    out_sdr2 = calculate_sisdr(torch.from_numpy(est2), torch.from_numpy(s2_norm)).item()

    sf.write(os.path.join(audio_dir, "mixture.wav"), mix_norm, sr)
    sf.write(os.path.join(audio_dir, "target_speaker1_clean.wav"), s1_norm, sr)
    sf.write(os.path.join(audio_dir, "target_speaker2_clean.wav"), s2_norm, sr)
    sf.write(os.path.join(audio_dir, "extracted_speaker1.wav"), est1, sr)
    sf.write(os.path.join(audio_dir, "extracted_speaker2.wav"), est2, sr)

    print(f"\n[Target Extraction Audio Outputs Saved in '{audio_dir}']")
    print(f"  - Input Mixture:               {os.path.join(audio_dir, 'mixture.wav')}")
    print(f"  - Extracted Spk 1 (via Vec 1): {os.path.join(audio_dir, 'extracted_speaker1.wav')} (SI-SDRi: {out_sdr1 - in_sdr1:+.2f} dB)")
    print(f"  - Extracted Spk 2 (via Vec 2): {os.path.join(audio_dir, 'extracted_speaker2.wav')} (SI-SDRi: {out_sdr2 - in_sdr2:+.2f} dB)")

    # Visualizations in dedicated subfolders
    targets_2d = np.stack([s1_norm, s2_norm], axis=0)
    estimates_2d = np.stack([est1, est2], axis=0)
    save_separate_waveforms(mix_norm, targets_2d, estimates_2d, sr, output_dir=waveform_dir)
    save_separate_spectrograms(mix_norm, targets_2d, estimates_2d, sr, output_dir=spec_dir)
    stacked_masks = torch.cat([mask1, mask2], dim=1)
    save_separate_mask_plots(stacked_masks, output_dir=mask_dir)
    if spike_seq1 is not None:
        save_separate_snn_raster_plots(spike_seq1, output_dir=snn_dir)


def run_multitalker_test(
    model: nn.Module,
    config: ModelConfig,
    device: torch.device,
    data_dir: str = "./data/MiniLibriMix",
    num_talkers: int = 3,
    target_idx: int = 0,
    output_dir: str = "./test_outputs",
):
    """
    Multi-Talker Cocktail Party Test:
    Mixes 3 or 4 overlapping speakers, then extracts strictly the selected target voice.
    """
    print(f"\n" + "=" * 68)
    print(f"      MULTI-TALKER TEST ({num_talkers} SPEAKERS OVERLAPPING)")
    print("=" * 68)
    print(f"Generating synthetic cocktail party mixture with {num_talkers} voices...")

    # Collect clean audio files from unique speakers
    s1_files = []
    for root, _, files in os.walk(data_dir):
        if os.path.basename(root) in ["s1", "s2"]:
            for f in sorted(files):
                if f.endswith((".wav", ".flac")):
                    s1_files.append(os.path.join(root, f))
                    if len(s1_files) >= num_talkers * 3:
                        break
            if len(s1_files) >= num_talkers * 3:
                break

    # Fallback to any audio files in data_dir if no structured s1/s2 directories exist
    if len(s1_files) < num_talkers:
        for root, _, files in os.walk(data_dir):
            for f in sorted(files):
                if f.endswith((".wav", ".flac", ".ogg")):
                    full_p = os.path.join(root, f)
                    if full_p not in s1_files:
                        s1_files.append(full_p)
                    if len(s1_files) >= num_talkers * 3:
                        break
            if len(s1_files) >= num_talkers * 3:
                break

    if len(s1_files) < num_talkers:
        print(f"[Warning] Not enough audio files found in '{data_dir}' to form a {num_talkers}-speaker mixture.")
        return

    # Select distinct files
    stride = len(s1_files) // num_talkers
    chosen_files = [s1_files[i * stride] for i in range(num_talkers)]
    sr = config.sample_rate
    target_samples = int(config.segment_length * sr)

    clips = []
    speaker_ids = []
    for f in chosen_files:
        wav, in_sr = sf.read(f)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if len(wav) < target_samples:
            wav = np.pad(wav, (0, target_samples - len(wav)))
        else:
            wav = wav[:target_samples]
        # Normalize
        wav = wav / max(np.max(np.abs(wav)), 1e-7)
        clips.append(wav)

        spk_id = os.path.basename(f).split("_")[0].split("-")[0]
        speaker_ids.append(spk_id)

    # Form crowded mixture
    crowded_mixture = np.sum(clips, axis=0)
    crowded_mixture = crowded_mixture / max(np.max(np.abs(crowded_mixture)), 1e-7)

    target_idx = max(0, min(target_idx, num_talkers - 1))
    target_clean = clips[target_idx]
    target_spk_id = speaker_ids[target_idx]
    target_ref_path = chosen_files[target_idx]

    # Extract target embedding
    cache_mgr = get_speaker_cache_manager(device=str(device))
    target_emb = cache_mgr.get_embedding(target_spk_id, reference_audio_path=target_ref_path).unsqueeze(0).to(device)

    # Run SNN Extraction
    x_tensor = torch.from_numpy(crowded_mixture).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        extracted_target = model(x_tensor, spk_embed=target_emb)[0, 0].cpu().numpy()

    # Calculate metrics
    in_sdr = calculate_sisdr(torch.from_numpy(crowded_mixture), torch.from_numpy(target_clean)).item()
    out_sdr = calculate_sisdr(torch.from_numpy(extracted_target), torch.from_numpy(target_clean)).item()
    sisdr_gain = out_sdr - in_sdr

    multi_dir = os.path.join(output_dir, "multitalker")
    os.makedirs(multi_dir, exist_ok=True)

    mix_path = os.path.join(multi_dir, f"crowded_mixture_{num_talkers}speakers.wav")
    clean_path = os.path.join(multi_dir, f"target_speaker_{target_idx+1}_clean.wav")
    extracted_path = os.path.join(multi_dir, f"target_speaker_{target_idx+1}_extracted.wav")

    sf.write(mix_path, crowded_mixture, sr)
    sf.write(clean_path, target_clean, sr)
    sf.write(extracted_path, extracted_target, sr)

    print(f"- Selected Target Speaker:  Speaker #{target_idx+1} (ID: {target_spk_id})")
    print(f"- Interfering Talkers:      {num_talkers - 1} overlapping voices")
    print(f"- Input Mixture SI-SDR:     {in_sdr:6.2f} dB")
    print(f"- Extracted Target SI-SDR:  {out_sdr:6.2f} dB")
    print(f"- SI-SDR Improvement (Δ):   {sisdr_gain:+6.2f} dB")
    print(f"\n[Saved Multi-Talker Audio Clips in '{multi_dir}']")
    print(f"  - Crowded Mixture:        {mix_path}")
    print(f"  - Ground Truth Target:    {clean_path}")
    print(f"  - Extracted Target Audio: {extracted_path}")
    print("=" * 68)


def main():
    default_model_cfg = ModelConfig()
    default_train_cfg = TrainConfig()

    parser = argparse.ArgumentParser(description="Test and Evaluate Spiking Target Speech Extraction Model")
    parser.add_argument("--snn_checkpoint", type=str, default=None, help="Path to trained SNN model checkpoint (.pt)")
    parser.add_argument("--task_mode", type=str, default="tse", help="Task mode: 'tse' (Target Speech Extraction)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cuda:0', 'cpu', 'auto')")
    parser.add_argument("--dataset_type", type=str, default="mini_librimix", help="Dataset type: 'librimix', 'mini_librimix'")
    parser.add_argument("--data_dir", type=str, default="./data/MiniLibriMix", help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--segment_length", type=float, default=2.0, help="Evaluation segment length (seconds)")
    parser.add_argument("--mixture_type", type=str, default="mix_clean", choices=["mix_both", "mix_clean"], help="Mixture condition")
    parser.add_argument("--subset", type=str, default="val", choices=["test", "val", "dev"], help="Evaluation subset")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of evaluation batches")
    parser.add_argument("--output_dir", type=str, default="./test_outputs", help="Directory to save test audio & plots")
    parser.add_argument("--num_talkers", type=int, default=3, choices=[2, 3, 4], help="Number of speakers for multi-talker test (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic evaluation")
    parser.add_argument("--target_talker", type=int, default=0, help="Target speaker index (0-indexed) to extract in multi-talker test")
    parser.add_argument("--test_multitalker", action="store_true", default=True, help="Run multi-talker (3-4 speakers) cocktail party test")
    parser.add_argument("--no_multitalker", dest="test_multitalker", action="store_false", help="Disable multi-talker test")
    args = parser.parse_args()

    # Deterministic execution
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    print("=" * 68)
    print("       Spiking Target Speech Extraction (SNN-TSE) Model Testing")
    print("=" * 68)
    print(f"- Task Mode:       {args.task_mode.upper()} (Target Speech Extraction)")
    print(f"- Device:          {device} ({gpu_name})")
    print(f"- Dataset:         {args.dataset_type} ({args.data_dir})")
    print(f"- Target Subset:   {args.subset}")
    print(f"- Mixture Type:    {args.mixture_type}")

    # Checkpoint Detection & Validation
    ckpt_path = args.snn_checkpoint
    if ckpt_path is not None and not os.path.exists(ckpt_path):
        print(f"\n[Error] Specified checkpoint not found at '{ckpt_path}'.")
        print("Please check the file path or train a model first using: python train.py")
        sys.exit(1)

    if ckpt_path is None:
        candidates = [
            "checkpoints/best_snn_tse_spectrogram_plif.pt",
            "checkpoints/best_snn_tse_spectrogram.pt",
            "checkpoints/best_snn_spectrogram_plif.pt",
            "checkpoints/best_snn_spectrogram.pt",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                ckpt_path = cand
                break
        if ckpt_path is None:
            print("\n[Error] No SNN model checkpoint found in 'checkpoints/' directory.")
            print("Please train a model first using: python train.py")
            sys.exit(1)

    print(f"\n[1/3] Loading Model Checkpoint: {ckpt_path} ...")
    try:
        snn_model, snn_config, snn_meta = load_model(ckpt_path, device=device)
    except Exception as e:
        print(f"\n[Error] Failed to load checkpoint from '{ckpt_path}': {e}")
        sys.exit(1)

    total_snn_params = sum(p.numel() for p in snn_model.parameters())
    enc_type = getattr(snn_config, "encoder_type", "spectrogram")
    actual_task = getattr(snn_config, "task_mode", args.task_mode)

    print(f"  - Model Type:       Spiking Conv-TasNet ({enc_type.upper()})")
    print(f"  - Task Mode:        {actual_task.upper()}")
    print(f"  - Checkpoint Epoch: {snn_meta.get('epoch', 'N/A')}")
    print(f"  - Checkpoint Val SI-SDR: {snn_meta.get('val_sisdr', 0.0):.2f} dB")
    print(f"  - Parameters:       {total_snn_params:,}")
    print(f"  - Timesteps (S):    {snn_config.snn_timesteps}")
    print(f"  - Membrane Decay:   {snn_config.snn_beta}")
    print(f"  - Surrogate Grad:   {snn_config.surrogate}")

    # Load Evaluation Dataset
    print(f"\n[2/3] Preparing Evaluation Dataset ({args.subset}) from: {args.data_dir} ...")
    try:
        _, val_loader = get_dataloaders(
            dataset_type=args.dataset_type,
            sample_rate=8000,
            segment_length=args.segment_length,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            mixture_type=args.mixture_type,
            eval_subset=args.subset,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            task_mode=actual_task,
        )
    except Exception as e:
        print(f"\n[Error] Failed to load dataset from '{args.data_dir}': {e}")
        print("Please verify the dataset directory or download MiniLibriMix using:")
        print("  python scripts/setup_data.py --storage_dir ./data")
        sys.exit(1)

    if len(val_loader.dataset) == 0:
        print(f"\n[Error] Dataset in '{args.data_dir}' contains 0 evaluation samples for subset '{args.subset}'.")
        sys.exit(1)

    print(f"  - Evaluation pairs: {len(val_loader.dataset)} samples across {len(val_loader)} batches")

    # Evaluate SNN on Benchmark
    print(f"\n[3/3] Running Benchmark Evaluation on {device}...")
    snn_results = evaluate_model(snn_model, val_loader, device=device, task_mode=actual_task, max_batches=args.max_batches)

    # Print Summary Table
    print("\n" + "=" * 68)
    print("                    EVALUATION BENCHMARK RESULTS")
    print("=" * 68)
    print(f"{'Metric':<36} | {'SNN Target Speech Extractor':<25}")
    print("-" * 68)
    print(f"{'Task Mode':<36} | {actual_task.upper():>25}")
    print(f"{'Encoder Type':<36} | {enc_type.upper():>25}")
    print(f"{'Input Mixture SI-SDR (dB)':<36} | {snn_results['mean_input_sisdr']:>22.2f} dB")
    print(f"{'Extracted Target SI-SDR (dB)':<36} | {snn_results['mean_output_sisdr']:>22.2f} dB")
    print(f"{'SI-SDR Improvement (SI-SNRi)':<36} | {snn_results['mean_sisdri']:>22.2f} dB")
    print(f"{'Std Dev SI-SDR (dB)':<36} | {snn_results['std_output_sisdr']:>22.2f} dB")
    print(f"{'Median SI-SDR (dB)':<36} | {snn_results['median_output_sisdr']:>22.2f} dB")
    print(f"{'Latency per Sample (ms)':<36} | {snn_results['avg_latency_per_sample_ms']:>22.1f} ms")
    print("=" * 68)

    # Generate Dual Target Extraction Samples and Visual Plots
    print(f"\nGenerating Sample Dual Target Audio and Diagnostic Plots in '{args.output_dir}'...")
    run_sample_separation(snn_model, snn_config, val_loader, device=device, output_dir=args.output_dir)

    # Multi-Speaker Cocktail Party Test (3 or 4 speakers)
    if args.test_multitalker:
        actual_data_dir = getattr(val_loader.dataset, "data_dir", args.data_dir)
        run_multitalker_test(
            model=snn_model,
            config=snn_config,
            device=device,
            data_dir=actual_data_dir,
            num_talkers=args.num_talkers,
            target_idx=args.target_talker,
            output_dir=args.output_dir,
        )

    print("\n[OK] Model Testing, Multi-Speaker Evaluation, and Visualizations Completed Successfully!")


if __name__ == "__main__":
    main()
