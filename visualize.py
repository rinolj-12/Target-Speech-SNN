"""
Visualization Utilities for Speech Separation and SNN Dynamics.

Provides functions to generate:
1. Waveform comparisons (Mixture, Target Sources, Separated Estimates, Overlays, and Individual plots).
2. Spectrogram comparisons (STFT time-frequency representations for all signals).
3. SNN Spike Raster activity maps and firing rate profiles across simulation timesteps and latent channels.
4. Latent Mask analysis heatmaps, contrasts, and value distribution histograms.
5. 3-Way Encoder Comparative plots (Metrics bar charts, Latency, and Side-by-side signal overlays).
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Dict, List, Any


# Standard visual palette for consistency
COLOR_MIX = "#333333"
COLOR_SPK1_TGT = "#1f77b4"
COLOR_SPK1_EST = "#ff7f0e"
COLOR_SPK2_TGT = "#2ca02c"
COLOR_SPK2_EST = "#d62728"
COLOR_ACCENT = "#6f42c1"


def plot_waveform_comparison(
    mixture: np.ndarray,
    targets: np.ndarray,
    estimates: np.ndarray,
    sample_rate: int = 8000,
    save_path: Optional[str] = None,
    title_suffix: str = "",
):
    """
    Plots multi-panel waveform comparison figure (Mixture, Spk1, Spk2).

    Args:
        mixture: 1-D array [Time]
        targets: 2-D array [2, Time]
        estimates: 2-D array [2, Time]
        sample_rate: Audio sampling rate.
        save_path: Optional file path to save figure.
        title_suffix: Optional title suffix.
    """
    time = np.linspace(0, len(mixture) / sample_rate, len(mixture), endpoint=False)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # 1. Mixture
    axes[0].plot(time, mixture, color=COLOR_MIX, lw=1.0)
    axes[0].set_title(f"Mixture Waveform x(t) {title_suffix}".strip(), fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # 2. Speaker 1
    axes[1].plot(time, targets[0], label="Ground Truth Speaker 1", color=COLOR_SPK1_TGT, alpha=0.7, lw=1.2)
    axes[1].plot(time, estimates[0], label="Estimated Speaker 1", color=COLOR_SPK1_EST, linestyle="--", alpha=0.85, lw=1.2)
    axes[1].set_title("Speaker 1: Target vs. Separated Estimate", fontsize=11)
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # 3. Speaker 2
    axes[2].plot(time, targets[1], label="Ground Truth Speaker 2", color=COLOR_SPK2_TGT, alpha=0.7, lw=1.2)
    axes[2].plot(time, estimates[1], label="Estimated Speaker 2", color=COLOR_SPK2_EST, linestyle="--", alpha=0.85, lw=1.2)
    axes[2].set_title("Speaker 2: Target vs. Separated Estimate", fontsize=11)
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] Saved waveform comparison plot to {save_path}")
    plt.close()


def save_separate_waveforms(
    mixture: np.ndarray,
    targets: np.ndarray,
    estimates: np.ndarray,
    sample_rate: int = 8000,
    output_dir: str = ".",
    title_suffix: str = "",
) -> Dict[str, str]:
    """
    Saves individual/separate PNG plots for every waveform signal and comparison overlay.

    Outputs:
        - waveform_mixture.png
        - waveform_target_speaker1.png
        - waveform_target_speaker2.png
        - waveform_separated_speaker1.png
        - waveform_separated_speaker2.png
        - waveform_speaker1_overlay.png
        - waveform_speaker2_overlay.png
        - waveform_comparison_all.png
    """
    os.makedirs(output_dir, exist_ok=True)
    time = np.linspace(0, len(mixture) / sample_rate, len(mixture), endpoint=False)
    saved_paths = {}

    # 1. Mixture Waveform
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(time, mixture, color=COLOR_MIX, lw=1.0)
    ax.set_title(f"Input Mixture Waveform x(t) {title_suffix}".strip(), fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    p = os.path.join(output_dir, "waveform_mixture.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["mixture"] = p

    # 2. Target Speaker 1
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(time, targets[0], color=COLOR_SPK1_TGT, lw=1.1)
    ax.set_title("Ground Truth Speaker 1 Waveform", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    p = os.path.join(output_dir, "waveform_target_speaker1.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["target_spk1"] = p

    # 3. Target Speaker 2
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(time, targets[1], color=COLOR_SPK2_TGT, lw=1.1)
    ax.set_title("Ground Truth Speaker 2 Waveform", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    p = os.path.join(output_dir, "waveform_target_speaker2.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["target_spk2"] = p

    # 4. Separated Speaker 1
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(time, estimates[0], color=COLOR_SPK1_EST, lw=1.1)
    ax.set_title(f"Separated Speaker 1 Estimated Waveform {title_suffix}".strip(), fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    p = os.path.join(output_dir, "waveform_separated_speaker1.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["separated_spk1"] = p

    # 5. Separated Speaker 2
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(time, estimates[1], color=COLOR_SPK2_EST, lw=1.1)
    ax.set_title(f"Separated Speaker 2 Estimated Waveform {title_suffix}".strip(), fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    p = os.path.join(output_dir, "waveform_separated_speaker2.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["separated_spk2"] = p

    # 6. Speaker 1 Overlay (Target vs Estimate)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(time, targets[0], label="Target (Ground Truth)", color=COLOR_SPK1_TGT, alpha=0.7, lw=1.2)
    ax.plot(time, estimates[0], label="Estimated Output", color=COLOR_SPK1_EST, linestyle="--", alpha=0.85, lw=1.2)
    ax.set_title(f"Speaker 1: Target vs Separated Estimate {title_suffix}".strip(), fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    p = os.path.join(output_dir, "waveform_speaker1_overlay.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["overlay_spk1"] = p

    # 7. Speaker 2 Overlay (Target vs Estimate)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(time, targets[1], label="Target (Ground Truth)", color=COLOR_SPK2_TGT, alpha=0.7, lw=1.2)
    ax.plot(time, estimates[1], label="Estimated Output", color=COLOR_SPK2_EST, linestyle="--", alpha=0.85, lw=1.2)
    ax.set_title(f"Speaker 2: Target vs Separated Estimate {title_suffix}".strip(), fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    p = os.path.join(output_dir, "waveform_speaker2_overlay.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["overlay_spk2"] = p

    # 8. All Waveforms 3-Panel Overview
    p_all = os.path.join(output_dir, "waveform_comparison_all.png")
    plot_waveform_comparison(mixture, targets, estimates, sample_rate, save_path=p_all, title_suffix=title_suffix)
    saved_paths["comparison_all"] = p_all

    return saved_paths


def plot_spectrograms(
    mixture: np.ndarray,
    targets: np.ndarray,
    estimates: np.ndarray,
    sample_rate: int = 8000,
    save_path: Optional[str] = None,
):
    """Plots STFT Spectrograms in a 3x2 grid figure."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True, sharey=True)

    # Mixture
    axes[0, 0].specgram(mixture, NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    axes[0, 0].set_title("Mixture Spectrogram", fontweight="bold")
    axes[0, 0].set_ylabel("Frequency (Hz)")
    axes[0, 1].axis("off")

    # Speaker 1
    axes[1, 0].specgram(targets[0], NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    axes[1, 0].set_title("Speaker 1 (Ground Truth)")
    axes[1, 0].set_ylabel("Frequency (Hz)")

    axes[1, 1].specgram(estimates[0], NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    axes[1, 1].set_title("Speaker 1 (Separated Estimate)")

    # Speaker 2
    axes[2, 0].specgram(targets[1], NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    axes[2, 0].set_title("Speaker 2 (Ground Truth)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Frequency (Hz)")

    axes[2, 1].specgram(estimates[1], NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    axes[2, 1].set_title("Speaker 2 (Separated Estimate)")
    axes[2, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] Saved spectrogram comparison plot to {save_path}")
    plt.close()


def save_separate_spectrograms(
    mixture: np.ndarray,
    targets: np.ndarray,
    estimates: np.ndarray,
    sample_rate: int = 8000,
    output_dir: str = ".",
) -> Dict[str, str]:
    """
    Saves individual/separate PNG spectrogram plots for each signal.

    Outputs:
        - spectrogram_mixture.png
        - spectrogram_target_speaker1.png
        - spectrogram_target_speaker2.png
        - spectrogram_separated_speaker1.png
        - spectrogram_separated_speaker2.png
        - spectrogram_speaker1_comparison.png
        - spectrogram_speaker2_comparison.png
        - spectrogram_overview.png
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    def _save_single_spec(sig: np.ndarray, title: str, filename: str):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.specgram(sig, NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frequency (Hz)")
        p = os.path.join(output_dir, filename)
        fig.tight_layout()
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return p

    saved_paths["spec_mixture"] = _save_single_spec(mixture, "Input Mixture Spectrogram (STFT)", "spectrogram_mixture.png")
    saved_paths["spec_target_spk1"] = _save_single_spec(targets[0], "Ground Truth Speaker 1 Spectrogram", "spectrogram_target_speaker1.png")
    saved_paths["spec_target_spk2"] = _save_single_spec(targets[1], "Ground Truth Speaker 2 Spectrogram", "spectrogram_target_speaker2.png")
    saved_paths["spec_sep_spk1"] = _save_single_spec(estimates[0], "Separated Speaker 1 Spectrogram", "spectrogram_separated_speaker1.png")
    saved_paths["spec_sep_spk2"] = _save_single_spec(estimates[1], "Separated Speaker 2 Spectrogram", "spectrogram_separated_speaker2.png")

    # Side-by-side comparison for Speaker 1 (Target vs Estimate)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    axes[0].specgram(targets[0], NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    axes[0].set_title("Speaker 1 Target (Ground Truth)", fontweight="bold")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")

    axes[1].specgram(estimates[0], NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    axes[1].set_title("Speaker 1 Separated Estimate", fontweight="bold")
    axes[1].set_xlabel("Time (s)")
    p1 = os.path.join(output_dir, "spectrogram_speaker1_comparison.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["spec_spk1_comparison"] = p1

    # Side-by-side comparison for Speaker 2 (Target vs Estimate)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    axes[0].specgram(targets[1], NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    axes[0].set_title("Speaker 2 Target (Ground Truth)", fontweight="bold")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")

    axes[1].specgram(estimates[1], NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    axes[1].set_title("Speaker 2 Separated Estimate", fontweight="bold")
    axes[1].set_xlabel("Time (s)")
    p2 = os.path.join(output_dir, "spectrogram_speaker2_comparison.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["spec_spk2_comparison"] = p2

    # Overview grid
    p_overview = os.path.join(output_dir, "spectrogram_overview.png")
    plot_spectrograms(mixture, targets, estimates, sample_rate, save_path=p_overview)
    saved_paths["spec_overview"] = p_overview

    return saved_paths


def plot_mask_analysis(
    masks: torch.Tensor,
    save_path: Optional[str] = None,
    title_suffix: str = "",
):
    """Plots 2x2 Mask analysis figure."""
    if masks.dim() == 4:
        masks = masks[0]

    m_np = masks.detach().cpu().numpy()

    if m_np.shape[0] == 1:
        m1 = m_np[0]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        im0 = axes[0].imshow(m1, aspect="auto", cmap="viridis", interpolation="nearest")
        axes[0].set_title(f"Target Latent Mask M {title_suffix}".strip(), fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Latent Channel (N)")
        axes[0].set_xlabel("Latent Time Frame (L)")
        plt.colorbar(im0, ax=axes[0], label="Mask Value")

        axes[1].hist(m1.flatten(), bins=40, alpha=0.7, color=COLOR_SPK1_TGT, label="Mask Values", density=True)
        axes[1].set_title("Mask Value Distribution", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Mask Value")
        axes[1].set_ylabel("Density")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)
    else:
        m1 = m_np[0]
        m2 = m_np[1]
        m_diff = m1 - m2

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        im0 = axes[0, 0].imshow(m1, aspect="auto", cmap="viridis", interpolation="nearest")
        axes[0, 0].set_title(f"Speaker 1 Mask M_1 {title_suffix}".strip(), fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Latent Channel (N)")
        axes[0, 0].set_xlabel("Latent Time Frame (L)")
        plt.colorbar(im0, ax=axes[0, 0], label="Mask Value")

        im1 = axes[0, 1].imshow(m2, aspect="auto", cmap="viridis", interpolation="nearest")
        axes[0, 1].set_title(f"Speaker 2 Mask M_2 {title_suffix}".strip(), fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Latent Channel (N)")
        axes[0, 1].set_xlabel("Latent Time Frame (L)")
        plt.colorbar(im1, ax=axes[0, 1], label="Mask Value")

        im2 = axes[1, 0].imshow(m_diff, aspect="auto", cmap="coolwarm", interpolation="nearest")
        axes[1, 0].set_title("Mask Contrast (M_1 - M_2): Blue=Spk2, Red=Spk1", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Latent Channel (N)")
        axes[1, 0].set_xlabel("Latent Time Frame (L)")
        plt.colorbar(im2, ax=axes[1, 0], label="Contrast")

        axes[1, 1].hist(m1.flatten(), bins=40, alpha=0.6, color=COLOR_SPK1_TGT, label="M_1 Values", density=True)
        axes[1, 1].hist(m2.flatten(), bins=40, alpha=0.6, color=COLOR_SPK1_EST, label="M_2 Values", density=True)
        axes[1, 1].set_title("Mask Value Distribution", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Mask Value")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].legend(loc="upper right")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] Saved mask analysis plot to {save_path}")
    plt.close()


def save_separate_mask_plots(
    masks: torch.Tensor,
    output_dir: str = ".",
    title_suffix: str = "",
) -> Dict[str, str]:
    """
    Saves individual/separate PNG plots for latent mask heatmaps and distribution.

    Outputs:
        - mask_speaker1.png
        - mask_speaker2.png
        - mask_contrast.png
        - mask_distribution.png
        - mask_analysis_summary.png
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    if masks.dim() == 4:
        masks = masks[0]

    m_np = masks.detach().cpu().numpy()

    if m_np.shape[0] == 1:
        m1 = m_np[0]
        # 1. Target Mask
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(m1, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(f"Target Latent Mask M {title_suffix}".strip(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Latent Channel (N)")
        ax.set_xlabel("Latent Time Frame (L)")
        plt.colorbar(im, ax=ax, label="Mask Value")
        p1 = os.path.join(output_dir, "mask_speaker1.png")
        fig.tight_layout()
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths["mask_spk1"] = p1

        # 2. Distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(m1.flatten(), bins=40, alpha=0.7, color=COLOR_SPK1_TGT, label="Mask Values", density=True)
        ax.set_title(f"Mask Value Distribution Histogram {title_suffix}".strip(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Mask Value")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        p4 = os.path.join(output_dir, "mask_distribution.png")
        fig.tight_layout()
        fig.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths["mask_dist"] = p4
    else:
        m1 = m_np[0]
        m2 = m_np[1]
        m_diff = m1 - m2

        # 1. Mask M1
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(m1, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(f"Speaker 1 Latent Mask M_1 {title_suffix}".strip(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Latent Channel (N)")
        ax.set_xlabel("Latent Time Frame (L)")
        plt.colorbar(im, ax=ax, label="Mask Value")
        p1 = os.path.join(output_dir, "mask_speaker1.png")
        fig.tight_layout()
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths["mask_spk1"] = p1

        # 2. Mask M2
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(m2, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(f"Speaker 2 Latent Mask M_2 {title_suffix}".strip(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Latent Channel (N)")
        ax.set_xlabel("Latent Time Frame (L)")
        plt.colorbar(im, ax=ax, label="Mask Value")
        p2 = os.path.join(output_dir, "mask_speaker2.png")
        fig.tight_layout()
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths["mask_spk2"] = p2

        # 3. Mask Contrast
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(m_diff, aspect="auto", cmap="coolwarm", interpolation="nearest")
        ax.set_title(f"Mask Contrast (M_1 - M_2): Blue=Spk2, Red=Spk1 {title_suffix}".strip(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Latent Channel (N)")
        ax.set_xlabel("Latent Time Frame (L)")
        plt.colorbar(im, ax=ax, label="Contrast Value")
        p3 = os.path.join(output_dir, "mask_contrast.png")
        fig.tight_layout()
        fig.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths["mask_contrast"] = p3

        # 4. Mask Distribution Histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(m1.flatten(), bins=40, alpha=0.6, color=COLOR_SPK1_TGT, label="M_1 Values", density=True)
        ax.hist(m2.flatten(), bins=40, alpha=0.6, color=COLOR_SPK1_EST, label="M_2 Values", density=True)
        ax.set_title(f"Mask Value Distribution Histogram {title_suffix}".strip(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Mask Value")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        p4 = os.path.join(output_dir, "mask_distribution.png")
        fig.tight_layout()
        fig.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths["mask_dist"] = p4

    # 5. Summary Grid
    p_summary = os.path.join(output_dir, "mask_analysis_summary.png")
    plot_mask_analysis(masks, save_path=p_summary, title_suffix=title_suffix)
    saved_paths["mask_summary"] = p_summary

    return saved_paths


def plot_snn_raster(
    spike_tensor: torch.Tensor,
    save_path: Optional[str] = None,
):
    """Plots SNN Spike Activity across Simulation Timesteps in a 2-panel figure."""
    spikes_np = spike_tensor.detach().cpu().numpy()
    s_steps = spikes_np.shape[0]
    rate_per_timestep = spikes_np.mean(axis=(1, 2, 3))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    channels_to_plot = min(32, spikes_np.shape[2])
    heatmap_data = spikes_np[:, 0, :channels_to_plot, :].mean(axis=-1)
    
    im = ax1.imshow(heatmap_data.T, aspect="auto", cmap="magma", interpolation="nearest")
    ax1.set_title("SNN Channel Spike Activity Across Simulation Timesteps (Sample 0)")
    ax1.set_xlabel("Simulation Timestep (t)")
    ax1.set_ylabel("Latent Channel")
    plt.colorbar(im, ax=ax1, label="Mean Firing Rate")

    ax2.bar(range(s_steps), rate_per_timestep, color=COLOR_ACCENT, edgecolor="black", alpha=0.8)
    ax2.set_title("Overall Mean Firing Rate per Simulation Timestep")
    ax2.set_xlabel("Simulation Timestep (t)")
    ax2.set_ylabel("Spike Rate")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] Saved spike raster plot to {save_path}")
    plt.close()


def save_separate_snn_raster_plots(
    spike_tensor: torch.Tensor,
    output_dir: str = ".",
) -> Dict[str, str]:
    """
    Saves individual/separate PNG plots for SNN Spike raster dynamics.

    Outputs:
        - snn_spike_raster.png (Channel activity heatmap)
        - snn_firing_rates.png (Bar chart of firing rates)
        - snn_raster_summary.png (Combined 2-panel figure)
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    spikes_np = spike_tensor.detach().cpu().numpy()
    s_steps = spikes_np.shape[0]
    rate_per_timestep = spikes_np.mean(axis=(1, 2, 3))

    # 1. Channel Activity Heatmap
    fig, ax = plt.subplots(figsize=(8, 4.5))
    channels_to_plot = min(32, spikes_np.shape[2])
    heatmap_data = spikes_np[:, 0, :channels_to_plot, :].mean(axis=-1)
    im = ax.imshow(heatmap_data.T, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_title("SNN Channel Spike Activity Across Simulation Timesteps", fontsize=11, fontweight="bold")
    ax.set_xlabel("Simulation Timestep (t)")
    ax.set_ylabel("Latent Channel")
    plt.colorbar(im, ax=ax, label="Mean Firing Rate")
    p1 = os.path.join(output_dir, "snn_spike_raster.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["snn_heatmap"] = p1

    # 2. Firing Rate Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(s_steps), rate_per_timestep, color=COLOR_ACCENT, edgecolor="black", alpha=0.85)
    ax.set_title("Overall Mean Firing Rate per Simulation Timestep", fontsize=11, fontweight="bold")
    ax.set_xlabel("Simulation Timestep (t)")
    ax.set_ylabel("Spike Rate")
    ax.grid(True, alpha=0.3)
    p2 = os.path.join(output_dir, "snn_firing_rates.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["snn_rates"] = p2

    # 3. Summary 2-panel
    p3 = os.path.join(output_dir, "snn_raster_summary.png")
    plot_snn_raster(spike_tensor, save_path=p3)
    saved_paths["snn_summary"] = p3

    return saved_paths


def plot_multi_encoder_comparison(
    mixture: np.ndarray,
    targets: np.ndarray,
    model_predictions: dict,
    sample_rate: int = 8000,
    save_path: Optional[str] = None,
):
    """Plots master side-by-side comparison figure across multiple encoder architectures."""
    n_models = len(model_predictions)
    total_rows = 1 + n_models
    time = np.linspace(0, len(mixture) / sample_rate, len(mixture), endpoint=False)

    fig, axes = plt.subplots(total_rows, 2, figsize=(16, 3.2 * total_rows), sharex="col")

    # Row 0: Ground Truth
    axes[0, 0].plot(time, mixture, color=COLOR_MIX, lw=0.9, alpha=0.6, label="Mixture")
    axes[0, 0].plot(time, targets[0], color=COLOR_SPK1_TGT, lw=1.1, alpha=0.8, label="Spk 1 Target")
    axes[0, 0].plot(time, targets[1], color=COLOR_SPK2_TGT, lw=1.1, alpha=0.8, label="Spk 2 Target")
    axes[0, 0].set_title("Ground Truth Mixture & Targets (Waveform)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].specgram(mixture, NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    axes[0, 1].set_title("Mixture Spectrogram (STFT)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Frequency (Hz)")

    colors = [("#ff7f0e", "#d62728"), ("#9467bd", "#8c564b"), ("#e377c2", "#17becf")]
    for idx, (name, data) in enumerate(model_predictions.items()):
        row = idx + 1
        est = data["est"]
        sisdr = data.get("sisdr", 0.0)
        sisdri = data.get("sisdri", 0.0)
        col1, col2 = colors[idx % len(colors)]

        axes[row, 0].plot(time, targets[0], color=COLOR_SPK1_TGT, lw=0.8, alpha=0.35, label="Spk 1 Target")
        axes[row, 0].plot(time, est[0], color=col1, lw=1.1, linestyle="--", label="Spk 1 Est")
        axes[row, 0].plot(time, targets[1], color=COLOR_SPK2_TGT, lw=0.8, alpha=0.35, label="Spk 2 Target")
        axes[row, 0].plot(time, est[1], color=col2, lw=1.1, linestyle="--", label="Spk 2 Est")
        axes[row, 0].set_title(f"[{name}] Output Waveforms (SI-SDR: {sisdr:+.2f} dB | SI-SNRi: {sisdri:+.2f} dB)", fontsize=11, fontweight="bold")
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 0].legend(loc="upper right", fontsize=8)
        axes[row, 0].grid(True, alpha=0.25)

        axes[row, 1].specgram(est[0], NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
        axes[row, 1].set_title(f"[{name}] Separated Spk 1 Spectrogram", fontsize=11)
        axes[row, 1].set_ylabel("Frequency (Hz)")

    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] Saved 3-way output comparison plot to {save_path}")
    plt.close()


def save_separate_comparison_plots(
    mixture: np.ndarray,
    targets: np.ndarray,
    model_predictions: dict,
    benchmark_rows: Optional[List[dict]] = None,
    sample_rate: int = 8000,
    output_dir: str = ".",
) -> Dict[str, str]:
    """
    Saves individual/separate PNG comparison figures across the 3 encoder models.

    Outputs:
        - comparison_3way_metrics.png (SI-SDR and SI-SNRi performance bar chart)
        - comparison_3way_latency.png (Inference latency & model parameters)
        - comparison_3way_waveforms.png (Waveform separation across models)
        - comparison_3way_spectrograms.png (Spectrogram separation across models)
        - test_3way_encoder_comparison.png (Combined summary sheet)
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}
    time = np.linspace(0, len(mixture) / sample_rate, len(mixture), endpoint=False)

    # 1. Performance Metrics Bar Chart (if benchmark_rows provided)
    if benchmark_rows:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        names = [r["name"] for r in benchmark_rows]
        sisdr_vals = [r["out_sisdr"] for r in benchmark_rows]
        sisdri_vals = [r["sisdri"] for r in benchmark_rows]

        x_pos = np.arange(len(names))
        width = 0.35

        rects1 = ax.bar(x_pos - width / 2, sisdr_vals, width, label="Output SI-SDR (dB)", color="#1f77b4", edgecolor="black")
        rects2 = ax.bar(x_pos + width / 2, sisdri_vals, width, label="SI-SNRi Improvement (dB)", color="#2ca02c", edgecolor="black")

        ax.set_title("Speech Separation Performance by Encoder Architecture", fontsize=12, fontweight="bold")
        ax.set_ylabel("SI-SDR / SI-SNRi (dB)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=10)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        for rect in rects1 + rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3 if h >= 0 else -12),
                        textcoords="offset points",
                        ha="center", va="bottom" if h >= 0 else "top", fontsize=9, fontweight="bold")

        p_metrics = os.path.join(output_dir, "comparison_3way_metrics.png")
        fig.tight_layout()
        fig.savefig(p_metrics, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths["metrics_barchart"] = p_metrics

        # 2. Latency & Parameter Counts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
        latencies = [r["latency"] for r in benchmark_rows]
        params_k = [r["params"] / 1e3 for r in benchmark_rows]

        ax1.bar(names, latencies, color="#e377c2", edgecolor="black", alpha=0.85)
        ax1.set_title("Inference Latency per Sample (ms)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=15, ha="right")
        ax1.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(latencies):
            ax1.text(i, v + 0.5, f"{v:.1f} ms", ha="center", fontweight="bold", fontsize=9)

        ax2.bar(names, params_k, color="#ff7f0e", edgecolor="black", alpha=0.85)
        ax2.set_title("Model Parameters (Thousands)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Parameters (k)")
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=15, ha="right")
        ax2.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(params_k):
            ax2.text(i, v + 5, f"{v:.0f}k", ha="center", fontweight="bold", fontsize=9)

        p_lat = os.path.join(output_dir, "comparison_3way_latency.png")
        fig.tight_layout()
        fig.savefig(p_lat, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths["latency_barchart"] = p_lat

    # 3. Waveforms Comparison
    n_models = len(model_predictions)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(12, 2.5 * (n_models + 1)), sharex=True)
    axes[0].plot(time, mixture, color=COLOR_MIX, label="Mixture", alpha=0.6, lw=0.9)
    axes[0].plot(time, targets[0], color=COLOR_SPK1_TGT, label="Spk 1 Target", lw=1.1)
    axes[0].plot(time, targets[1], color=COLOR_SPK2_TGT, label="Spk 2 Target", lw=1.1)
    axes[0].set_title("Ground Truth Mixture and Targets", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.25)

    colors = [("#ff7f0e", "#d62728"), ("#9467bd", "#8c564b"), ("#e377c2", "#17becf")]
    for idx, (name, data) in enumerate(model_predictions.items()):
        ax = axes[idx + 1]
        est = data["est"]
        sisdr = data.get("sisdr", 0.0)
        c1, c2 = colors[idx % len(colors)]
        ax.plot(time, est[0], color=c1, lw=1.1, label="Spk 1 Est")
        ax.plot(time, est[1], color=c2, lw=1.1, label="Spk 2 Est")
        ax.set_title(f"[{name}] Separated Output Waveforms (SI-SDR: {sisdr:+.2f} dB)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (seconds)")

    p_wavs = os.path.join(output_dir, "comparison_3way_waveforms.png")
    fig.tight_layout()
    fig.savefig(p_wavs, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["comparison_waveforms"] = p_wavs

    # 4. Spectrograms Comparison
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(10, 2.5 * (n_models + 1)), sharex=True, sharey=True)
    axes[0].specgram(mixture, NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    axes[0].set_title("Input Mixture Spectrogram", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Frequency (Hz)")

    for idx, (name, data) in enumerate(model_predictions.items()):
        ax = axes[idx + 1]
        est = data["est"]
        ax.specgram(est[0], NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
        ax.set_title(f"[{name}] Separated Speaker 1 Spectrogram", fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency (Hz)")
    axes[-1].set_xlabel("Time (seconds)")

    p_specs = os.path.join(output_dir, "comparison_3way_spectrograms.png")
    fig.tight_layout()
    fig.savefig(p_specs, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths["comparison_spectrograms"] = p_specs

    # 5. Master Comparison Sheet
    p_master = os.path.join(output_dir, "test_3way_encoder_comparison.png")
    plot_multi_encoder_comparison(mixture, targets, model_predictions, sample_rate, save_path=p_master)
    saved_paths["comparison_master"] = p_master

    return saved_paths
