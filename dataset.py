"""
Datasets and Audio Mixture Loaders for Speech Separation.

Provides:
1. MiniLibriMixDataset: Real 2-speaker speech separation dataset from LibriMix.
2. WavFolderDataset: Dataset loader for custom audio/speech WAV files.
3. get_dataloaders: High-level dataloader factory for training and validation.
"""

import os
import sys
import urllib.request
import zipfile
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Union
from speaker_encoder import get_speaker_cache_manager


class MiniLibriMixDataset(Dataset):
    """
    Dataset loader for Libri2Mix / MiniLibriMix real speech separation and target extraction.

    Expects standard LibriMix structure:
        data_dir/
            train/ (or val/ or dev/)
                mix_clean/ (or mix_both/ or mix/)
                s1/
                s2/
    """

    def __init__(
        self,
        data_dir: str = "./data/Libri2Mix",
        sample_rate: int = 8000,
        segment_length: float = 2.0,
        subset: str = "train",  # 'train' or 'val' or 'dev' or 'test'
        mixture_type: str = "mix_both",  # 'mix_both' (noisy) or 'mix_clean'
        cache_in_memory: bool = False,
        epoch_samples: Optional[int] = None,
        task_mode: str = "tse",          # 'tse' (Target Speech Extraction)
        spk_cache_path: str = "./data/minilibrimix_spk_embeddings.pt",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.task_mode = task_mode.lower()
        self.spk_cache_path = spk_cache_path
        self.cache_manager = get_speaker_cache_manager(cache_file=spk_cache_path)
        self.segment_samples = int(sample_rate * segment_length)
        self.pairs = []
        if subset == "test":
            target_subsets = ["test"]
        elif subset in ["val", "dev"]:
            target_subsets = ["val", "dev"]
        elif subset in ["train", "train-100", "train-360"]:
            target_subsets = ["train"]
        else:
            target_subsets = [subset]

        if os.path.isdir(data_dir):
            for root, _, _ in os.walk(data_dir):
                path_parts = [p.lower() for p in os.path.normpath(root).split(os.sep)]
                # Matches exact subset name or subfolder like 'train-100', 'train-360', 'dev', etc.
                matches_subset = any(
                    any(s in part for s in target_subsets) for part in path_parts
                ) or subset == "all"
                if matches_subset:
                    cand_s1 = os.path.join(root, "s1")
                    cand_s2 = os.path.join(root, "s2")
                    cand_mix = os.path.join(root, mixture_type)
                    if not os.path.isdir(cand_mix):
                        cand_mix = os.path.join(root, "mix_both")
                    if not os.path.isdir(cand_mix):
                        cand_mix = os.path.join(root, "mix_clean")
                    if not os.path.isdir(cand_mix):
                        cand_mix = os.path.join(root, "mix")

                    if os.path.isdir(cand_s1) and os.path.isdir(cand_s2):
                        for f in sorted(os.listdir(cand_s1)):
                            if f.endswith((".wav", ".flac")):
                                p_s1 = os.path.join(cand_s1, f)
                                p_s2 = os.path.join(cand_s2, f)
                                p_mix = os.path.join(cand_mix, f) if os.path.isdir(cand_mix) else None
                                if os.path.exists(p_s1) and os.path.exists(p_s2):
                                    self.pairs.append((p_mix if (p_mix and os.path.exists(p_mix)) else None, p_s1, p_s2))
                        if len(self.pairs) > 0 and subset != "all":
                            break

            # Fallback only when loading all data without subset structure
            if len(self.pairs) == 0 and subset == "all":
                for root, _, _ in os.walk(data_dir):
                    cand_s1 = os.path.join(root, "s1")
                    cand_s2 = os.path.join(root, "s2")
                    cand_mix = os.path.join(root, mixture_type)
                    if not os.path.isdir(cand_mix):
                        cand_mix = os.path.join(root, "mix_both")
                    if not os.path.isdir(cand_mix):
                        cand_mix = os.path.join(root, "mix_clean")
                    if not os.path.isdir(cand_mix):
                        cand_mix = os.path.join(root, "mix")

                    if os.path.isdir(cand_s1) and os.path.isdir(cand_s2):
                        for f in sorted(os.listdir(cand_s1)):
                            if f.endswith((".wav", ".flac")):
                                p_s1 = os.path.join(cand_s1, f)
                                p_s2 = os.path.join(cand_s2, f)
                                p_mix = os.path.join(cand_mix, f) if os.path.isdir(cand_mix) else None
                                if os.path.exists(p_s1) and os.path.exists(p_s2):
                                    self.pairs.append((p_mix if (p_mix and os.path.exists(p_mix)) else None, p_s1, p_s2))
                        if len(self.pairs) > 0:
                            break

        self.subset = subset
        self.cache_in_memory = cache_in_memory
        self._raw_cache = {}
        self.epoch_samples = epoch_samples

    def __len__(self) -> int:
        if self.epoch_samples is not None and self.epoch_samples > 0:
            return self.epoch_samples
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.pairs) == 0:
            raise RuntimeError(f"MiniLibriMixDataset found 0 audio pairs in {self.data_dir}. Please ensure dataset is downloaded.")

        item_idx = idx % len(self.pairs)

        if self.cache_in_memory and item_idx in self._raw_cache:
            s1_full, s2_full, mix_full = self._raw_cache[item_idx]
        else:
            p_mix, p_s1, p_s2 = self.pairs[item_idx]
            s1_full, _ = sf.read(p_s1)
            s2_full, _ = sf.read(p_s2)

            if s1_full.ndim > 1:
                s1_full = s1_full.mean(axis=-1)
            if s2_full.ndim > 1:
                s2_full = s2_full.mean(axis=-1)

            if p_mix and os.path.exists(p_mix):
                mix_full, _ = sf.read(p_mix)
                if mix_full.ndim > 1:
                    mix_full = mix_full.mean(axis=-1)
            else:
                min_len = min(len(s1_full), len(s2_full))
                mix_full = s1_full[:min_len] + s2_full[:min_len]

            # Ensure all match min length
            common_len = min(len(s1_full), len(s2_full), len(mix_full))
            s1_full = s1_full[:common_len]
            s2_full = s2_full[:common_len]
            mix_full = mix_full[:common_len]

            if self.cache_in_memory:
                self._raw_cache[item_idx] = (s1_full, s2_full, mix_full)

        total_len = len(mix_full)
        target_len = self.segment_samples

        if total_len > target_len and self.subset == "train":
            # Dynamic random cropping during training to explore full multi-second utterance
            start_offset = np.random.randint(0, total_len - target_len + 1)
            s1 = s1_full[start_offset : start_offset + target_len]
            s2 = s2_full[start_offset : start_offset + target_len]
            mix = mix_full[start_offset : start_offset + target_len]
        else:
            # Deterministic start for validation / evaluation or short clips
            if total_len < target_len:
                pad_len = target_len - total_len
                s1 = np.pad(s1_full, (0, pad_len))
                s2 = np.pad(s2_full, (0, pad_len))
                mix = np.pad(mix_full, (0, pad_len))
            else:
                s1 = s1_full[:target_len]
                s2 = s2_full[:target_len]
                mix = mix_full[:target_len]

        # Normalization
        max_val = max(np.max(np.abs(mix)), 1e-7)
        mix, s1, s2 = mix / max_val, s1 / max_val, s2 / max_val

        mix_tensor = torch.from_numpy(mix.copy()).unsqueeze(0).float()  # [1, Time]

        # Extract speaker IDs from filenames (LibriMix format: 'spk1_spk2.wav')
        f1 = os.path.basename(p_s1).replace(".wav", "").replace(".flac", "")
        parts1 = f1.split("_")
        spk1_id = parts1[0].split("-")[0]
        spk2_id = parts1[1].split("-")[0] if len(parts1) > 1 else os.path.basename(p_s2).split("-")[0]

        # In training: randomly select target (50% s1, 50% s2). In eval: deterministic
        if self.subset == "train":
            choose_spk1 = (np.random.rand() > 0.5)
        else:
            choose_spk1 = (item_idx % 2 == 0)

        if choose_spk1:
            target_clean = s1
            target_spk_id = spk1_id
            ref_path = p_s1
        else:
            target_clean = s2
            target_spk_id = spk2_id
            ref_path = p_s2

        # Fetch memoized 192-D speaker voiceprint
        spk_embed = self.cache_manager.get_embedding(
            speaker_id=target_spk_id,
            reference_audio_path=ref_path,
            sample_rate=self.sample_rate,
        )

        target_tensor = torch.from_numpy(target_clean.copy()).unsqueeze(0).float()  # [1, Time]
        return mix_tensor, target_tensor, spk_embed


class WavFolderDataset(Dataset):
    """
    Dataset loader for paired audio WAV files in a custom directory.
    """

    def __init__(
        self,
        folder_path: str,
        sample_rate: int = 8000,
        segment_length: float = 2.0,
        epoch_samples: Optional[int] = None,
    ):
        super().__init__()
        self.folder_path = folder_path
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_length)
        self.epoch_samples = epoch_samples
        self.wav_files = []

        if os.path.isdir(folder_path):
            for root, _, files in os.walk(folder_path):
                for f in sorted(files):
                    if f.endswith((".wav", ".flac", ".ogg")):
                        self.wav_files.append(os.path.join(root, f))

    def __len__(self) -> int:
        if self.epoch_samples is not None and self.epoch_samples > 0:
            return min(self.epoch_samples, max(len(self.wav_files) // 2, 0))
        return max(len(self.wav_files) // 2, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.wav_files) < 2:
            raise RuntimeError(f"WavFolderDataset requires at least 2 WAV files in {self.folder_path}, found {len(self.wav_files)}.")

        idx1 = (idx * 2) % len(self.wav_files)
        idx2 = (idx * 2 + 1) % len(self.wav_files)

        data1, _ = sf.read(self.wav_files[idx1])
        data2, _ = sf.read(self.wav_files[idx2])

        if data1.ndim > 1:
            data1 = data1.mean(axis=-1)
        if data2.ndim > 1:
            data2 = data2.mean(axis=-1)

        def pad_crop(arr: np.ndarray, target_len: int) -> np.ndarray:
            if len(arr) < target_len:
                return np.pad(arr, (0, target_len - len(arr)))
            return arr[:target_len]

        s1 = pad_crop(data1, self.segment_samples)
        s2 = pad_crop(data2, self.segment_samples)

        mixture = s1 + s2
        max_val = max(np.max(np.abs(mixture)), 1e-7)
        mixture, s1, s2 = mixture / max_val, s1 / max_val, s2 / max_val

        mix_tensor = torch.from_numpy(mixture).unsqueeze(0).float()
        targets_tensor = torch.from_numpy(np.stack([s1, s2], axis=0)).float()
        return mix_tensor, targets_tensor


def _has_audio_files(directory: str) -> bool:
    """Checks if directory contains extracted .wav or .flac audio files."""
    if not os.path.exists(directory):
        return False
    for _, _, files in os.walk(directory):
        if any(f.endswith((".wav", ".flac")) for f in files):
            return True
    return False


# ---------------------------------------------------------------------------
# MiniLibriMix auto-downloader
# ---------------------------------------------------------------------------

_MINI_LIBRIMIX_ZENODO_URL = "https://zenodo.org/records/3871592/files/MiniLibriMix.zip?download=1"
_MINI_LIBRIMIX_FALLBACK_URL = "https://zenodo.org/record/3871592/files/MiniLibriMix.zip"


def _download_with_progress(url: str, dest_path: str) -> None:
    """Downloads a file from *url* to *dest_path* with a live progress bar."""
    req = urllib.request.Request(url, headers={"User-Agent": "curl/7.81.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total_bytes_str = resp.headers.get("content-length")
            total_bytes = int(total_bytes_str) if total_bytes_str else None
            downloaded = 0
            chunk = 1024 * 1024  # 1 MB
            with open(dest_path, "wb") as out:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    out.write(buf)
                    downloaded += len(buf)
                    if total_bytes:
                        pct = downloaded / total_bytes * 100
                        bar_len = 40
                        filled = int(bar_len * downloaded / total_bytes)
                        bar = "#" * filled + "-" * (bar_len - filled)
                        print(
                            f"\r  [{bar}] {downloaded/(1024**2):.1f}/{total_bytes/(1024**2):.1f} MB ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(f"\r  {downloaded/(1024**2):.1f} MB downloaded", end="", flush=True)
            print()  # newline after progress bar
    except Exception as exc:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise RuntimeError(f"Download failed from {url}: {exc}") from exc


def download_minilibrimix(dest_dir: str = "./data") -> str:
    """
    Downloads and extracts the official MiniLibriMix dataset from Zenodo.

    MiniLibriMix is a ~580 MB subset of Libri2Mix (1000 train + 1000 val pairs @
    8 kHz) released by the original LibriMix authors.  No SoX or extra tools
    are needed — just Python.

    Args:
        dest_dir: Parent folder where ``MiniLibriMix/`` will be created.

    Returns:
        Absolute path to the extracted ``MiniLibriMix`` directory.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "MiniLibriMix.zip")
    out_dir  = os.path.join(dest_dir, "MiniLibriMix")

    # Already there?
    if _has_audio_files(out_dir):
        print(f"[Dataset] MiniLibriMix already present at '{out_dir}'. Skipping download.")
        return os.path.abspath(out_dir)

    print("=" * 68)
    print("  Downloading MiniLibriMix from Zenodo (~580 MB)")
    print("=" * 68)
    print(f"  URL : {_MINI_LIBRIMIX_ZENODO_URL}")
    print(f"  Dest: {zip_path}")

    # Try primary URL, fall back to mirror
    for url in [_MINI_LIBRIMIX_ZENODO_URL, _MINI_LIBRIMIX_FALLBACK_URL]:
        try:
            _download_with_progress(url, zip_path)
            break
        except RuntimeError as exc:
            print(f"  [Warning] {exc}  — trying fallback URL…")
    else:
        raise RuntimeError(
            "Could not download MiniLibriMix from any mirror.\n"
            "Check your internet connection or download manually from:\n"
            f"  {_MINI_LIBRIMIX_ZENODO_URL}\n"
            f"and extract it to '{out_dir}'."
        )

    print(f"  Extracting to '{dest_dir}' …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    os.remove(zip_path)  # free disk space

    if not _has_audio_files(out_dir):
        raise RuntimeError(
            f"Extraction finished but no audio files found in '{out_dir}'.\n"
            "The zip may be corrupt — please delete it and retry."
        )

    print(f"[Dataset] MiniLibriMix downloaded and extracted to '{out_dir}'.")
    return os.path.abspath(out_dir)


def _prompt_dataset_setup(requested_dir: str) -> str:
    """
    Interactive prompt shown when no audio dataset is found.
    Offers to:
      1. Auto-download MiniLibriMix (~580 MB, no SoX needed).
      2. Run the full Libri2Mix generator (requires SoX, ~15 GB).
      3. Abort.

    Returns the folder path to use after setup, or raises SystemExit.
    """
    print()
    print("=" * 68)
    print("  No audio dataset found.")
    print(f"  Looked in: '{requested_dir}'")
    print("=" * 68)
    print()
    print("  Choose how to get the dataset:")
    print("    [1] Download MiniLibriMix (~580 MB, ~1 000 pairs, no SoX needed)")
    print("        Recommended for first run / CPU-only laptops.")
    print("    [2] Generate full Libri2Mix train-100 via setup script")
    print("        (~15–20 GB, requires SoX.  Run the script manually:)")
    print("          python scripts/setup_data.py --storage_dir ./data")
    print("        This will take a long time — start it in a separate terminal.")
    print("    [3] Abort and handle manually")
    print()

    # Non-interactive environments (CI, containers) — default to option 1
    if not sys.stdin.isatty():
        print("  [Non-interactive mode] Defaulting to option 1 (MiniLibriMix auto-download).")
        choice = "1"
    else:
        try:
            choice = input("  Enter choice [1/2/3] (default 1): ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            choice = "3"

    if choice == "1":
        data_parent = os.path.dirname(os.path.abspath(requested_dir))
        # If the user pointed at Libri2Mix, put MiniLibriMix next to it
        return download_minilibrimix(dest_dir=data_parent)
    elif choice == "2":
        print()
        print("  Run the following command in your terminal and then restart training:")
        print("    python scripts/setup_data.py --storage_dir ./data")
        raise SystemExit(0)
    else:
        raise SystemExit(
            "Aborted.  Please provide a dataset directory with audio files.\n"
            "See README.md ➜ 'Step 3 — Get the Dataset' for instructions."
        )


# Alias for Libri2Mix
Libri2MixDataset = MiniLibriMixDataset


def get_dataloaders(
    dataset_type: str = "librimix",
    sample_rate: int = 8000,
    segment_length: float = 2.0,
    batch_size: int = 4,
    data_dir: str = "./data/Libri2Mix",
    wav_folder: Optional[str] = None,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    train_samples_per_epoch: Optional[int] = None,
    mixture_type: str = "mix_clean",
    eval_subset: str = "val",
    task_mode: str = "tse",
    spk_cache_path: str = "./data/minilibrimix_spk_embeddings.pt",
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory creating training and validation/test DataLoaders for Libri2Mix, MiniLibriMix, or custom WAV folders.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    effective_epoch_samples = train_samples_per_epoch if (train_samples_per_epoch and train_samples_per_epoch > 0) else None

    if dataset_type in ["librimix", "libri2mix", "mini_librimix"]:
        folder = data_dir if data_dir else "./data/Libri2Mix"

        # ── Dataset resolution priority ─────────────────────────────────────
        # 1. Use the requested folder directly if it already has audio.
        # 2. If requesting Libri2Mix but only MiniLibriMix is present, auto-
        #    fall back to MiniLibriMix (no user action needed).
        # 3. If nothing is found anywhere, interactively ask the user to either
        #    auto-download MiniLibriMix (~580 MB) or run the full setup script.
        if not _has_audio_files(folder):
            mini_dir = "./data/MiniLibriMix"
            if _has_audio_files(mini_dir):
                print(
                    f"[Dataset] '{folder}' has no audio files. "
                    f"Auto-falling back to existing '{mini_dir}'."
                )
                folder = mini_dir
            else:
                # Nothing found anywhere — ask the user what to do
                folder = _prompt_dataset_setup(folder)

        train_dataset = Libri2MixDataset(
            folder,
            sample_rate=sample_rate,
            segment_length=segment_length,
            subset="train",
            mixture_type=mixture_type,
            epoch_samples=effective_epoch_samples,
            task_mode=task_mode,
            spk_cache_path=spk_cache_path,
        )
        val_dataset = Libri2MixDataset(
            folder,
            sample_rate=sample_rate,
            segment_length=segment_length,
            subset=eval_subset,
            mixture_type=mixture_type,
            task_mode=task_mode,
            spk_cache_path=spk_cache_path,
        )
        # If test split was requested but absent (e.g. MiniLibriMix has train/val), fall back to val/dev
        if len(val_dataset) == 0 and eval_subset == "test":
            for alt in ["val", "dev"]:
                alt_ds = Libri2MixDataset(
                    folder,
                    sample_rate=sample_rate,
                    segment_length=segment_length,
                    subset=alt,
                    mixture_type=mixture_type,
                    task_mode=task_mode,
                    spk_cache_path=spk_cache_path,
                )
                if len(alt_ds) > 0:
                    val_dataset = alt_ds
                    print(
                        f"[Dataset] No dedicated 'test' folder in '{folder}'. "
                        f"Auto-evaluating on held-out '{alt}' split ({len(val_dataset)} samples)."
                    )
                    break
        elif len(val_dataset) == 0 and eval_subset in ["val", "dev"]:
            alt = "dev" if eval_subset == "val" else "val"
            alt_ds = Libri2MixDataset(
                folder,
                sample_rate=sample_rate,
                segment_length=segment_length,
                subset=alt,
                mixture_type=mixture_type,
                task_mode=task_mode,
                spk_cache_path=spk_cache_path,
            )
            if len(alt_ds) > 0:
                val_dataset = alt_ds

        if len(val_dataset) == 0:
            print(f"[Dataset Warning] No validation/test split found. Falling back to training samples.")
            val_dataset = train_dataset

        if len(train_dataset) == 0:
            raise RuntimeError(
                f"No audio samples found in '{folder}'.\n"
                f"Please verify the dataset path or run:\n"
                f"  python scripts/setup_data.py --storage_dir ./data"
            )

    elif dataset_type == "wav_folder":
        folder = wav_folder or data_dir
        if not folder or not os.path.isdir(folder):
            raise ValueError(f"WavFolderDataset directory '{folder}' not found. Please provide a valid --data_dir.")
        train_dataset = WavFolderDataset(folder, sample_rate=sample_rate, segment_length=segment_length, epoch_samples=effective_epoch_samples)
        # Validation uses a fixed 200 samples so validation finishes in seconds instead of 10 minutes
        val_dataset = WavFolderDataset(folder, sample_rate=sample_rate, segment_length=segment_length, epoch_samples=200)
    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'. Choose 'librimix', 'mini_librimix', or 'wav_folder'.")

    persistent = (num_workers > 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(train_dataset) >= batch_size else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("=" * 60)
    print("  Testing MiniLibriMix Dataset & DataLoaders")
    print("=" * 60)
    
    data_directory = "./data/MiniLibriMix"
    sample_rate = 8000
    segment_sec = 2.0
    batch_size = 4
    
    print(f"Loading datasets from: {data_directory}")
    train_loader, val_loader = get_dataloaders(
        dataset_type="mini_librimix",
        sample_rate=sample_rate,
        segment_length=segment_sec,
        batch_size=batch_size,
        data_dir=data_directory,
        num_workers=0,
    )
    
    print(f"[OK] Train samples: {len(train_loader.dataset)} ({len(train_loader)} batches of {batch_size})")
    print(f"[OK] Val samples:   {len(val_loader.dataset)} ({len(val_loader)} batches of {batch_size})")
    
    # Fetch a sample batch
    mix_batch, target_batch, spk_embed = next(iter(train_loader))
    print("\n--- Sample Batch Verification ---")
    print(f"Mixture tensor shape:      {mix_batch.shape} (Expected: [{batch_size}, 1, {int(sample_rate * segment_sec)}])")
    print(f"Target tensor shape:       {target_batch.shape} (Expected: [{batch_size}, 1, {int(sample_rate * segment_sec)}])")
    print(f"Speaker embedding shape:   {spk_embed.shape} (Expected: [{batch_size}, 192])")
    print(f"Mixture value range:       [{mix_batch.min().item():.3f}, {mix_batch.max().item():.3f}]")
    print(f"Target value range:        [{target_batch.min().item():.3f}, {target_batch.max().item():.3f}]")
    
    print("\n[OK] MiniLibriMix Target Speech Extraction Dataset verification completed successfully!")

