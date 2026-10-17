#!/usr/bin/env python3
"""
Speech Separation Dataset Setup & Generator.
Compatible with Windows 11, Ubuntu Linux, and macOS.

Supports 3 Dataset Options:
  1. MiniLibriMix: Pre-mixed lightweight subset (~580 MB download, ~750 MB disk).
     Instant setup, no heavy generation or SoX required.
  2. Libri2Mix Clean: Full train-clean-100 benchmark (~6.3 GB download, ~10-12 GB disk).
     Generates clean 2-speaker speech mixtures (mix_clean) @ 8 kHz via SoX.
  3. Libri2Mix Both: Full benchmark with clean & realistic noise mixtures (~24 GB downloads, ~25-35 GB disk).
     Downloads LibriSpeech + WHAM! background noise and generates mix_clean and mix_both.

Usage:
  # Interactive menu (recommended):
  python scripts/setup_data.py

  # Direct command-line flags:
  python scripts/setup_data.py --mini             # Setup MiniLibriMix (~750 MB)
  python scripts/setup_data.py --clean_only       # Generate Libri2Mix Clean (~11 GB)
  python scripts/setup_data.py --with_noise       # Generate Libri2Mix Clean + Noisy (~30 GB)
  python scripts/setup_data.py --storage_dir /path/to/external_drive
"""

import os
import sys
import shutil
import urllib.request
import tarfile
import zipfile
import subprocess
import argparse
from pathlib import Path


# Zenodo URLs for MiniLibriMix
_MINI_LIBRIMIX_URLS = [
    "https://zenodo.org/records/3871592/files/MiniLibriMix.zip?download=1",
    "https://zenodo.org/record/3871592/files/MiniLibriMix.zip",
]


def print_banner(msg: str):
    print("\n" + "=" * 68)
    print(f"  {msg}")
    print("=" * 68)


def get_disk_free_gb(path: Path) -> float:
    """Returns free space in GB on the filesystem containing path."""
    try:
        usage = shutil.disk_usage(str(path))
        return usage.free / (1024 ** 3)
    except Exception:
        return 0.0


def check_sox():
    """Checks if sox is installed and accessible on system PATH."""
    sox_path = shutil.which("sox")
    if not sox_path:
        # Check standard Windows paths
        candidates = [
            Path(sys.prefix) / "Scripts",
            Path(r"C:\Program Files (x86)\sox-14-4-2"),
            Path(r"C:\Program Files\sox-14-4-2"),
        ]
        for c in candidates:
            if (c / "sox.exe").exists():
                os.environ["PATH"] = str(c) + os.pathsep + os.environ.get("PATH", "")
                sox_path = shutil.which("sox")
                if sox_path:
                    break

    if not sox_path:
        print("\n[ERROR] 'sox' (Sound eXchange) was not found on your system PATH.")
        print("  SoX is required to resample and mix audio at 8 kHz for full Libri2Mix.")
        if sys.platform == "win32":
            print("  Windows 11 Installation Options:")
            print("    1. via Winget:      winget install -e --id SoX.SoX")
            print("    2. via Chocolatey:  choco install sox")
        elif sys.platform == "darwin":
            print("  macOS Installation:")
            print("    brew install sox")
        else:
            print("  Ubuntu / Debian Installation:")
            print("    sudo apt-get update && sudo apt-get install -y sox libsox-fmt-all")
        sys.exit(1)
    print(f"  [OK] Found SoX at: {sox_path}")


def download_file(url: str, dest_path: Path):
    """Downloads a file with a live progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  [OK] Already downloaded: {dest_path.name}")
        return

    print(f"  Downloading {dest_path.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/7.81.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out:
        total_size = resp.headers.get("content-length")
        total_bytes = int(total_size) if total_size else None
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = (downloaded / total_bytes) * 100
                bar_len = 38
                filled = int(bar_len * downloaded / total_bytes)
                bar = "#" * filled + "-" * (bar_len - filled)
                print(
                    f"\r    [{bar}] {downloaded/(1024**2):.1f}/{total_bytes/(1024**2):.1f} MB ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r    {downloaded/(1024**2):.1f} MB downloaded", end="", flush=True)
    print()  # newline after progress bar


def extract_archive(archive_path: Path, extract_dir: Path):
    """Extracts tar.gz or zip archive in pure Python, then removes the archive."""
    print(f"  Extracting {archive_path.name} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif str(archive_path).endswith(".tar.gz") or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    archive_path.unlink()  # free disk space after extraction


def setup_minilibrimix(storage_path: Path):
    """Downloads and extracts the official MiniLibriMix dataset from Zenodo."""
    print_banner("Setting up MiniLibriMix (~580 MB)")
    out_dir = storage_path / "MiniLibriMix"
    zip_path = storage_path / "MiniLibriMix.zip"

    # Check if already present
    has_audio = False
    if out_dir.exists():
        for r, _, files in os.walk(out_dir):
            if any(f.endswith((".wav", ".flac")) for f in files):
                has_audio = True
                break
    if has_audio:
        print(f"  [OK] MiniLibriMix already present at: {out_dir}")
        return out_dir

    print(f"  Destination directory: {storage_path}")
    download_success = False
    for url in _MINI_LIBRIMIX_URLS:
        try:
            download_file(url, zip_path)
            download_success = True
            break
        except Exception as e:
            print(f"  [Warning] Download failed ({e}), trying fallback mirror...")

    if not download_success or not zip_path.exists():
        raise RuntimeError(f"Could not download MiniLibriMix from Zenodo mirrors.")

    extract_archive(zip_path, storage_path)
    print(f"  [OK] MiniLibriMix ready at: {out_dir}")
    return out_dir


def prompt_dataset_choice(storage_path: Path) -> str:
    """
    Displays interactive menu showing choices and current free disk space.
    Returns: 'mini', 'clean', or 'both'
    """
    free_gb = get_disk_free_gb(storage_path)

    print()
    print("=" * 68)
    print("  Speech Separation Dataset Setup — Choose what to install/generate:")
    print("=" * 68)
    print(f"  Target Storage:   {storage_path}")
    print(f"  Free Disk Space:  {free_gb:.1f} GB available")
    print("-" * 68)
    print()
    print("  [1] MiniLibriMix  (~580 MB download, ~750 MB disk space)")
    print("      • Pre-mixed 2-speaker speech subset (800 train + 200 val pairs @ 8 kHz).")
    print("      • Instant download & extraction — no SoX or audio generation needed.")
    print("      • Best for: Fast sanity checks, debugging, and limited disk space.")
    print()
    print("  [2] Libri2Mix Clean  (~6.3 GB download, ~10-12 GB disk space)")
    print("      • Standard benchmark on LibriSpeech train-clean-100 (~13,900 mixtures).")
    print("      • Clean 2-speaker speech mixtures (mix_clean) @ 8 kHz generated via SoX.")
    print("      • Standard academic baseline without noise. Fits within ~16 GB disk space.")
    print()
    print("  [3] Libri2Mix Both / Noisy  (~24 GB downloads, ~25-35 GB disk space)")
    print("      • Full benchmark: clean speech (mix_clean) PLUS realistic background noise (mix_both)")
    print("      • Downloads LibriSpeech (~7 GB) + WHAM! ambient noise (~17.6 GB) from Zenodo.")
    if free_gb < 25.0:
        print(f"      • ⚠️ WARNING: Requires ~25-35 GB space, but only {free_gb:.1f} GB is free!")
    print()

    if not sys.stdin.isatty():
        print("  [Non-interactive mode] Defaulting to Option 1 (MiniLibriMix).")
        return "mini"

    while True:
        try:
            choice = input("  Enter choice [1/2/3] (default 2 – Libri2Mix Clean): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "clean"

        if choice in ["", "2"]:
            return "clean"
        elif choice == "1":
            return "mini"
        elif choice == "3":
            if free_gb < 25.0:
                print(f"\n  [CAUTION] You only have {free_gb:.1f} GB free on {storage_path}.")
                print("  Libri2Mix with WHAM! noise typically requires at least 25-35 GB.")
                confirm = input("  Are you sure you want to proceed? [y/N]: ").strip().lower()
                if confirm != "y":
                    continue
            return "both"
        else:
            print("  Invalid choice. Please enter 1, 2, or 3.")


def setup_libri2mix(storage_path: Path, want_noise: bool):
    """Sets up SoX, clones LibriMix generator, downloads LibriSpeech/WHAM!, and generates mixtures."""
    print_banner(f"Setting up Full Libri2Mix ({'Clean + Noisy' if want_noise else 'Clean Only'})")

    # 1. Audio Tools
    print("\n[1/4] Checking audio tools...")
    check_sox()

    # 2. Generator Repo
    print("\n[2/4] Setting up LibriMix generator...")
    generator_dir = storage_path / "LibriMix_generator"
    if not generator_dir.exists():
        print(f"  Cloning JorisCos/LibriMix into {generator_dir} ...")
        subprocess.check_call(
            ["git", "clone", "https://github.com/JorisCos/LibriMix.git", str(generator_dir)]
        )
    else:
        print(f"  [OK] Generator repo exists at: {generator_dir}")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "pandas", "soundfile", "scipy", "numpy"]
    )

    # 3. Download LibriSpeech
    print("\n[3/4] Downloading LibriSpeech source audio...")
    librispeech_dir = storage_path / "LibriSpeech"
    librispeech_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("http://www.openslr.org/resources/12/dev-clean.tar.gz",        librispeech_dir / "dev-clean"),
        ("http://www.openslr.org/resources/12/test-clean.tar.gz",       librispeech_dir / "test-clean"),
        ("http://www.openslr.org/resources/12/train-clean-100.tar.gz",  librispeech_dir / "train-clean-100"),
    ]

    for url, check_dir in sources:
        if not check_dir.exists():
            archive = storage_path / Path(url).name
            download_file(url, archive)
            extract_archive(archive, storage_path)
        else:
            print(f"  [OK] Found {check_dir.name}")

    # 4. Download WHAM! noise if requested
    wham_check_1 = storage_path / "wham_noise" / "wham_noise"
    wham_check_2 = storage_path / "wham_noise"
    has_wham = wham_check_1.exists() or (wham_check_2.exists() and any(wham_check_2.iterdir()))

    if want_noise and not has_wham:
        wham_zip = storage_path / "wham_noise.zip"
        wham_url = "https://zenodo.org/records/3338999/files/wham_noise.zip?download=1"
        print(f"\n  Downloading WHAM! noise from Zenodo (~17.6 GB)...")
        try:
            download_file(wham_url, wham_zip)
            extract_archive(wham_zip, storage_path)
            has_wham = True
        except Exception as exc:
            print(f"\n  [Warning] WHAM! noise download failed: {exc}")
            print("  Falling back to clean-only mixtures.")
            if wham_zip.exists():
                wham_zip.unlink()

    wham_dir = wham_check_1 if wham_check_1.exists() else wham_check_2

    # 5. Generate Libri2Mix
    types_to_generate = ["mix_clean", "mix_both"] if has_wham else ["mix_clean"]
    print(f"\n[4/4] Generating Libri2Mix ({' & '.join(types_to_generate)} @ 8 kHz via SoX)...")
    librimix_outdir = storage_path / "Libri2Mix"

    metadata_dir  = generator_dir / "metadata" / "Libri2Mix"
    create_script = generator_dir / "scripts" / "create_librimix_from_metadata.py"

    cmd = [
        sys.executable,
        str(create_script),
        "--librispeech_dir", str(librispeech_dir),
        "--wham_dir",        str(wham_dir if has_wham else librispeech_dir),
        "--metadata_dir",    str(metadata_dir),
        "--librimix_outdir", str(storage_path),
        "--n_src",  "2",
        "--freqs",  "8k",
        "--modes",  "min",
        "--types",  *types_to_generate,
    ]

    print(f"  Running generation script: {' '.join(cmd)}\n")
    subprocess.check_call(cmd, cwd=str(generator_dir), env=os.environ.copy())

    print_banner("Libri2Mix Generation Completed Successfully!")
    print(f"  Location: {librimix_outdir}")
    print(f"  Types:    {', '.join(types_to_generate)}")
    return librimix_outdir, ("mix_both" if has_wham else "mix_clean")


def main():
    parser = argparse.ArgumentParser(
        description="Setup Audio Datasets for Speech Separation (MiniLibriMix / Libri2Mix)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_data.py                    # Interactive menu
  python scripts/setup_data.py --mini             # Setup MiniLibriMix (~750 MB)
  python scripts/setup_data.py --clean_only       # Generate Libri2Mix Clean (~11 GB)
  python scripts/setup_data.py --with_noise       # Generate Libri2Mix Clean + Noisy (~30 GB)
  python scripts/setup_data.py --storage_dir /mnt/data --clean_only
        """,
    )
    parser.add_argument(
        "--storage_dir",
        type=str,
        default="./data",
        help="Directory where datasets will be stored (default: ./data)",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mini",
        action="store_true",
        help="Download official MiniLibriMix (~580 MB download, ~750 MB disk).",
    )
    mode_group.add_argument(
        "--clean_only",
        action="store_true",
        help="Generate clean Libri2Mix mixtures only (mix_clean). Fits in ~12 GB.",
    )
    mode_group.add_argument(
        "--with_noise",
        action="store_true",
        help="Generate clean + noisy Libri2Mix mixtures (mix_clean & mix_both). Requires ~30 GB.",
    )

    args = parser.parse_args()

    storage_path = Path(args.storage_dir).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)

    # Determine choice: CLI flag or interactive menu
    if args.mini:
        choice = "mini"
    elif args.clean_only:
        choice = "clean"
    elif args.with_noise:
        choice = "both"
    else:
        choice = prompt_dataset_choice(storage_path)

    # Execute selected setup
    py = "python" if sys.platform == "win32" else "python3"

    if choice == "mini":
        out_dir = setup_minilibrimix(storage_path)
        print_banner("MiniLibriMix Ready!")
        print(f"  Location: {out_dir}")
        print("\n  Train with:")
        print(f"    {py} train.py --dataset_type mini_librimix --data_dir {out_dir} --mixture_type mix_clean")
        print("\n  Evaluate with:")
        print(f"    {py} test_model.py --dataset_type mini_librimix --data_dir {out_dir}")
    else:
        want_noise = (choice == "both")
        out_dir, mix_type = setup_libri2mix(storage_path, want_noise=want_noise)
        print("\n  Train with:")
        print(f"    {py} train.py --dataset_type librimix --data_dir {out_dir} --mixture_type {mix_type}")
        print("\n  Evaluate with:")
        print(f"    {py} test_model.py --dataset_type librimix --data_dir {out_dir} --mixture_type {mix_type}")

    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()

