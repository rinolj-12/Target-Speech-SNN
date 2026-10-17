# 🧠 Spiking Target Speech Extraction (SNN-TSE)

> **TL;DR** — A biologically inspired **Target Speech Extraction** system that listens to a crowded audio scene (2, 3, or 4 overlapping speakers) and extracts only the voice you want — conditioned on a short reference audio from the target speaker. Powered by **Spiking Neural Networks (PLIF neurons)**, **Neuromodulatory FiLM Conditioning**, and a **pre-trained ECAPA-TDNN speaker voiceprint encoder**.

---

## 📖 What Is This?

**SNN-TSE** solves the classic "cocktail party problem" in a novel, energy-efficient way:

| Pillar | Role |
|---|---|
| **Target Speech Extraction (TSE)** | Given a crowded mixture and a reference voice clip, extract *only* that person's clean speech — all others become ignored interference |
| **ECAPA-TDNN Speaker Voiceprint** | A frozen pre-trained speaker verification network encodes a 192-D identity vector from any reference clip of the target speaker |
| **Spiking Neural Network (SNN)** | PLIF neurons fire sparse binary spikes — far more energy-efficient and biologically plausible than standard ReLU/LSTM networks |
| **Neuromodulatory FiLM Conditioning** | The 192-D voiceprint dynamically *modulates* (scales γ and shifts β) the SNN bottleneck, steering it to attend to the right speaker |

> [!NOTE]
> Unlike Blind Source Separation (BSS), TSE does not need to separate *all* speakers. It outputs **a single clean stream** for the chosen target, regardless of how many other voices are in the mixture.

---

## ⚡ Quick Start

### Prerequisites

| Tool | Required? | Notes |
|---|---|---|
| **Python 3.9+** | ✅ Always | `python --version` to check |
| **Git** | ✅ Always | For cloning |
| **SoX** | ⚠️ Full dataset only | Only for Libri2Mix generation — not needed for MiniLibriMix |
| **NVIDIA GPU** | ❌ Optional | CPU works; GPU trains ~20–50× faster |

---

### Step 1 — Clone & Create Environment

```bash
git clone https://github.com/rinolj-12/Target-Speech-SNN.git
cd Target-Speech-SNN
python -m venv .venv
```

Activate the virtual environment:

| Platform | Command |
|---|---|
| **Linux / macOS** | `source .venv/bin/activate` |
| **Windows CMD** | `.venv\Scripts\activate.bat` |
| **Windows PowerShell** | `.\venv\Scripts\Activate.ps1` |

---

### Step 2 — Install Dependencies

**🔵 CPU only (any laptop, no GPU needed):**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

**🟢 NVIDIA GPU (CUDA 12.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> [!TIP]
> Not sure which CUDA version you have? Run `nvidia-smi` and check the top-right corner.
> Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to find the right install command.

---

### Step 3 — Get the Dataset

Run the interactive dataset setup tool:
```bash
python scripts/setup_data.py
```

| # | Dataset | Download Size | Disk Space | Best For |
|---|---|---|---|---|
| **1** | **MiniLibriMix** | ~580 MB | ~750 MB | First run, CPU, quick iteration (no SoX) |
| **2** | **Libri2Mix Clean** | ~6.3 GB | ~10–12 GB | Standard clean benchmark (~13.9k pairs) |
| **3** | **Libri2Mix Noisy** | ~24 GB | ~25–35 GB | Full benchmark with WHAM! background noise |

Or use non-interactive flags:
```bash
python scripts/setup_data.py --mini             # MiniLibriMix
python scripts/setup_data.py --clean_only       # Libri2Mix Clean
python scripts/setup_data.py --with_noise       # Libri2Mix + WHAM! Noise
```

---

### Step 4 — Train

**🔵 CPU (MiniLibriMix — auto-downloaded on first run):**
```bash
python train.py --device cpu --no_amp --batch_size 2 --mixture_type mix_clean
```

**🟢 GPU — MiniLibriMix (~580 MB, ~1,000 pairs):**
```bash
python train.py --device cuda --dataset_type mini_librimix --data_dir ./data/MiniLibriMix --mixture_type mix_clean --batch_size 4 --grad_accum_steps 4 --epochs 70
```

**🟢 GPU — Full Libri2Mix (~13,900 pairs):**
```bash
python train.py --device cuda --dataset_type librimix --data_dir ./data/Libri2Mix --mixture_type mix_clean --batch_size 4 --grad_accum_steps 4 --epochs 70
```

> [!TIP]
> On the first run, `train.py` automatically builds and caches ECAPA-TDNN speaker voiceprints into
> `./data/minilibrimix_spk_embeddings.pt`. This takes ~1–5 min but only runs once.

---

### Step 5 — Test & Extract

**🟢 GPU Evaluation & Multi-Talker Cocktail Party Test:**
```bash
# Evaluate on validation set + extract Speaker #1 from 3 overlapping speakers
python test_model.py --device cuda --num_talkers 3 --target_talker 0

# Extract a specific target from 4 overlapping speakers
python test_model.py --device cuda --num_talkers 4 --target_talker 2
```

**🔵 CPU Evaluation:**
```bash
python test_model.py --device cpu --num_talkers 3 --target_talker 0
```

---

## 📊 Dataset Details

| Property | MiniLibriMix | Libri2Mix train-100 |
|---|---|---|
| Training pairs | ~1,000 | ~13,900 |
| Audio duration | ~3.5 hours | ~58 hours |
| Download size | ~580 MB | ~6.3 GB (clean) / ~23 GB (noisy) |
| Requires SoX | ❌ No | ✅ Yes |
| Best for | First run, CPU, quick iteration | Full benchmark & serious training |
| Sample rate | 8 kHz | 8 kHz |
| Segment length | 2.0 s (16,000 samples) | 2.0 s (16,000 samples) |

---

## 🏗️ Architecture

The system is a 5-stage end-to-end pipeline. The novelty is the SNN separator conditioned by a frozen speaker voiceprint via **Neuromodulatory FiLM Conditioning**.

```
                  REFERENCE AUDIO (any clean clip of the target speaker)
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │  ECAPA-TDNN  (Frozen)   │  Pre-trained SpeechBrain
                        │  Speaker Verification   │  speaker identity network
                        └─────────────┬───────────┘
                                      │  e_target ∈ ℝ^192
                                      │  (192-D speaker voiceprint)
                                      │
────────────────────────┼────────────────────────────────────────────────
                        │  FiLM Conditioning: γ, β ◄────────────────────┐
                        │                                                │
Mixed Audio x(t)        │                                                │
[Batch, 1, T]           │                                                │
      │                 │                                                │
      ▼                 │                                                │
┌──────────────────────────────────────────────────────────────────────┐│
│               STAGE 1 — STFT SPECTROGRAM ENCODER                    ││
│                                                                      ││
│  • Complex STFT: n_fft=256, hop=64  →  8 ms frame rate @ 8 kHz      ││
│  • Magnitude spectrogram  W     [Batch, 129, L]                      ││
│  • Mixture phase saved    θ     [Batch, 129, L]  (for decoder)       ││
│  • 1×1 Conv projection: 129 bins → N=64 latent channels             ││
│  → Output: W  [Batch, N=64, L]                                       ││
└──────────────────────────────┬───────────────────────────────────────┘│
                               │  W [B, 64, L]                          │
                               ▼                                        │
┌──────────────────────────────────────────────────────────────────────┐│
│               STAGE 2 — SPIKE ENCODER                                ││
│                                                                      ││
│  Converts continuous latent W into temporal spike trains             ││
│  • Direct Current (default): W directly drives membrane potential   ││
│  • Bit-Plane: radix-2 decomposition → high temporal precision        ││
│  • Population Coding / Rate Coding (optional)                        ││
│  → Output: Spike Train  [S=6 Timesteps, B, N=64, L]                 ││
└──────────────────────────────┬───────────────────────────────────────┘│
                               │  S [6, B, 64, L]                       │
                               ▼                                        │
┌──────────────────────────────────────────────────────────────────────┐│
│           STAGE 3 — SPIKING TCN SEPARATOR  (FiLM-conditioned)       ││
│                                                                      ││
│  Bottleneck 1×1 Conv + GroupNorm: [N=64 → B=128 channels]           ││
│                           │                                          ││
│  ┌──── Neuromodulatory FiLM Conditioning ─────────────────────────┐ ││
│  │  e_target ──► Linear(192 → 256) ──► split ──► (γ_raw, β)      │ ││
│  │  γ = sigmoid(γ_raw) × 2.0     [range: 0 → 2]                  │ ││
│  │  Modulate: x ← x × γ + β     (per-channel, all timesteps)     │◄┘│
│  └────────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│  ┌──── Stacked Dilated Spiking TCN Blocks (12 total) ─────────────┐  │
│  │  2× repeat of dilations [1, 2, 4, 8, 16, 32]                   │  │
│  │  Each block:                                                    │  │
│  │    ├─ PLIF Neurons: learnable decay β, threshold V_th          │  │
│  │    ├─ Surrogate Gradient (Fast Sigmoid) for backprop           │  │
│  │    ├─ Dilated Depthwise Conv1D (kernel P=3)                    │  │
│  │    └─ Skip connection accumulated across all blocks            │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│  ┌──── Continuous Membrane Readout + Residual Bridge ──────────────┐  │
│  │  Mean(U[t] across S timesteps) + skip features                 │  │
│  │  + Residual bridge from continuous W  (bypass path)            │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│  1×1 Mask Conv ──► Target Mask M  [B, 1, N=64, L]  (Sigmoid)       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │  M [B, 1, 64, L]
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  STAGE 4 — MASKING                                   │
│                                                                      │
│  W_target = M ⊙ W                                                    │
│  [B, 1, 64, L]  =  [B, 1, 64, L]  ×  [B, 64, L]  (broadcast)       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │  W_target [B, 1, 64, L]
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│               STAGE 5 — iSTFT SPECTROGRAM DECODER                   │
│                                                                      │
│  • 1×1 Conv unprojection: N=64 → 129 frequency bins                 │
│  • Reconstruct complex STFT: magnitude × e^(j·θ_mixture)            │
│  • Inverse STFT → waveform trimmed to original input length          │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
              ŝ_target(t)  [Batch, 1, Time]
              ─── Only the Target Speaker's Clean Voice ───
```

---

## 🔬 Key Concepts & Novelty

### 1. Neuromodulatory FiLM Conditioning

Feature-wise Linear Modulation applies a speaker-aware **affine transform** to the SNN bottleneck at every simulation timestep:

```
x_modulated = x × γ(e_target) + β(e_target)
```

- **γ** ∈ [0, 2] — gain gate (sigmoid-scaled): boosts spectral regions where the target speaker dominates
- **β** — bias shift: suppresses frequency bands dominated by interfering speakers
- Both derived from a single linear projection `Linear(192 → 256)` of the ECAPA-TDNN voiceprint

### 2. PLIF Neurons — Learnable Membrane Dynamics

Each PLIF (Parametric Leaky Integrate-and-Fire) neuron has a **per-neuron learnable decay β** updated by surrogate gradients:

```
U[t] = β · U[t-1] + I[t]           (leaky integration over timestep t)
S[t] = H(U[t] − V_threshold)        (binary spike if membrane crosses threshold)
U[t] ← U[t] − S[t] · V_threshold   (soft reset after spike)

Surrogate backprop: dH/dU ≈ 1 / (1 + |U|)²   (Fast Sigmoid)
```

### 3. Why 2-Speaker Training → 3 or 4 Speaker Extraction?

TSE frames the problem as **target vs. noise**, not speaker counting. The SNN learns to preserve whatever matches `e_target` and suppress everything else. A 3rd or 4th interferer just increases the noise floor — the model generalizes naturally without retraining.

---

## 🎯 Loss Function

```
L_total = −SI-SDR(ŝ_target, s_target)  +  λ_STFT · L_MR-STFT(ŝ_target, s_target)
```

| Term | Purpose |
|---|---|
| **Target SI-SDR** | Primary: scale-invariant SDR between extracted stream and clean target |
| **MR-STFT** (optional) | Spectral convergence + log-magnitude L1 at 3 STFT resolutions (λ=0 by default) |

> [!NOTE]
> **No Permutation Invariant Training (uPIT) required.** Since the target is identified by `e_target`,
> the assignment is fully deterministic — no O(K!) label-flip instability.

---

## 🧪 Testing & Multi-Talker Evaluation

### Test CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--num_talkers` | `3` | Speakers in cocktail party test: `2`, `3`, or `4` |
| `--target_talker` | `0` | Index (0-based) of the target speaker to extract |
| `--snn_checkpoint` | auto | Path to `.pt` checkpoint (auto-detects best in `checkpoints/`) |
| `--device` | `auto` | `cpu`, `cuda`, or `cuda:0` |
| `--max_batches` | None | Limit eval batches (for quick debugging) |
| `--no_multitalker` | — | Skip cocktail party test, run standard eval only |

---

## 📂 Test Outputs

Running `test_model.py` generates organized outputs:

```
test_outputs/
├── spectrogram/
│   ├── audio/
│   │   ├── mixture.wav                 — Input mixed audio
│   │   ├── extracted_speaker1.wav      — Target speaker 1 extracted via voiceprint
│   │   ├── extracted_speaker2.wav      — Target speaker 2 extracted via voiceprint
│   │   ├── target_speaker1_clean.wav   — Ground truth speaker 1
│   │   └── target_speaker2_clean.wav   — Ground truth speaker 2
│   ├── waveforms/                      — Waveform overlay comparison plots
│   ├── spectrograms/                   — Time-frequency spectrogram comparisons
│   ├── masks/                          — Predicted target masks (sigmoid output)
│   └── snn/                            — Spike raster plots & per-layer firing rates
└── multitalker/
    ├── crowded_mixture_3speakers.wav   — Synthetic cocktail party (3 or 4 voices)
    ├── target_speaker_1_clean.wav      — Ground truth for target speaker
    └── target_speaker_1_extracted.wav  — SNN-extracted target voice
```

---

## 📁 Project Structure

```
Speech/
├── train.py              # Training loop: AMP, grad accumulation, checkpointing, TSE loss
├── test_model.py         # Evaluation suite + multi-talker cocktail party test
│
├── model.py              # SpikingConvTasNet — full TSE pipeline assembly
├── encoder.py            # STFT Spectrogram encoder (magnitude + phase separation)
├── decoder.py            # iSTFT Spectrogram decoder (phase reconstruction)
├── separator.py          # FiLM-conditioned Spiking TCN separator (core novelty)
├── snn.py                # PLIF/ALIF/LIF neurons, surrogate gradients, tdBN
├── spike_encoder.py      # Spike encoding (Direct, Bit-Plane, Population, Rate)
├── speaker_encoder.py    # Frozen ECAPA-TDNN voiceprint extractor + memoized cache
│
├── losses.py             # TargetSISDRLoss + TSECombinedLoss (+ optional MR-STFT)
├── dataset.py            # Libri2Mix / MiniLibriMix DataLoaders (returns mix, target, embed)
├── config.py             # ModelConfig & TrainConfig dataclasses
│
├── visualize.py          # Waveform, spectrogram, mask & SNN spike-raster diagnostic plots
│
├── scripts/
│   └── setup_data.py     # Interactive Libri2Mix / MiniLibriMix dataset setup
│
├── pretrained_models/
│   └── spkrec-ecapa-voxceleb/  # Downloaded ECAPA-TDNN checkpoint (auto on first run)
│
├── checkpoints/          # Saved SNN-TSE model weights (.pt)
├── test_outputs/         # Audio & visualization outputs (auto-generated)
├── data/                 # Dataset storage (Libri2Mix or MiniLibriMix)
├── requirements.txt      # Python dependencies (install PyTorch separately)
└── .gitattributes        # CRLF/LF line-ending rules
```

---

## 🚀 Training Reference

<details>
<summary><b>All training flags for <code>train.py</code> — click to expand</b></summary>

### Model Architecture

| Flag | Default | Description |
|---|---|---|
| `--neuron_type` | `plif` | SNN neuron: `plif`, `alif`, `aplif`, `fs_neuron`, `lif` |
| `--spike_encoding` | `direct_current` | Spike encoding: `direct_current`, `bit_plane`, `rate`, `population`, `learnable_plif` |
| `--snn_readout` | `membrane` | Mask readout: `membrane`, `weighted_bit`, `rate` |
| `--use_residual_bridge` | `True` | Continuous residual skip to mask head (`--no_residual_bridge` to disable) |
| `--snn_timesteps` | `6` | SNN simulation timesteps S |
| `--snn_beta` | `0.9` | PLIF membrane decay factor β (learnable per-neuron) |
| `--surrogate` | `fast_sigmoid` | Surrogate gradient: `fast_sigmoid`, `atan`, `piecewise` |
| `--num_repeats` | `2` | Dilation stack repeats R (total = R × 6 = 12 blocks) |
| `--bottleneck_channels` | `128` | Bottleneck channels B |
| `--hidden_channels` | `256` | Depthwise conv hidden channels H |
| `--encoder_channels` | `64` | Latent channels N (projected from 129 STFT bins) |
| `--mask_activation` | `sigmoid` | Mask activation: `sigmoid`, `softmax`, `relu` |
| `--stft_n_fft` | `256` | FFT size (256 → 129 bins, 32 ms @ 8 kHz) |
| `--stft_hop_length` | `64` | Hop length (64 → 8 ms frame rate @ 8 kHz) |

### Training Schedule

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `70` | Total training epochs |
| `--batch_size` | `4` | Mini-batch size |
| `--grad_accum_steps` | `4` | Gradient accumulation (effective batch = 16) |
| `--lr` | `1e-3` | Adam initial learning rate |
| `--lr_scheduler` | `cosine` | LR scheduler: `cosine` or `plateau` |
| `--min_lr` | `1e-5` | Minimum LR floor |
| `--weight_decay` | `1e-5` | L2 regularization |
| `--mr_stft_weight` | `0.0` | MR-STFT auxiliary loss weight (0 = disabled) |
| `--mixture_type` | `mix_clean` | `mix_clean` or `mix_both` (noisy) |
| `--segment_length` | `2.0` | Segment crop duration in seconds |
| `--spk_cache_path` | `./data/minilibrimix_spk_embeddings.pt` | Precomputed voiceprint cache path |

### Hardware & Performance

| Flag | Default | Description |
|---|---|---|
| `--device` | `auto` | `auto`, `cuda`, `cuda:0`, or `cpu` |
| `--amp` | `True` | FP16 AMP (`--no_amp` on CPU) |
| `--checkpointing` | `True` | Gradient checkpointing (~80% VRAM savings) |
| `--num_workers` | `0` | DataLoader workers (0 = safe default) |
| `--resume` | — | Resume: `--resume auto` or `--resume path/to/ckpt.pt` |

</details>

---

## 📐 Model Size & Performance

| Architecture | Frontend | Config | Parameters |
|---|---|---|---|
| **Spiking Conv-TasNet (PLIF + FiLM)** | STFT (n_fft=256, hop=64) | N=64, B=128, H=256, R=2, S=6 | ~1.70 M |

| Metric | Value | Notes |
|---|---|---|
| **Inference Latency (CPU)** | ~95 ms / 2s clip | Standard laptop CPU |
| **Inference Latency (GPU)** | ~30 ms / 2s clip | RTX-class GPU |
| **Algorithmic Frame Rate** | 8 ms | hop=64 @ 8 kHz — real-time capable |
| **Speaker Embedding Lookup** | <0.001 ms | Memoized ECAPA-TDNN cache hit |
| **ECAPA-TDNN (Frozen)** | 83 MB | Downloaded once on first run |

---

## 📜 Key References & Foundational Work

- **Conv-TasNet**: Luo & Mesgarani, *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*, IEEE/ACM TASLP 2019.
- **Spiking Neural Networks & PLIF**: Fang et al., *Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks*, ICCV 2021.
- **Neuromodulatory FiLM**: Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018.
- **ECAPA-TDNN Speaker Verification**: Desplanques et al., *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*, Interspeech 2020.
- **SpeechBrain Toolkit**: Ravanelli et al., *SpeechBrain: A General-Purpose Speech Toolkit*, 2021.
- **Target Speech Extraction Foundations**: Wang et al., *VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking*, Interspeech 2019; Ge et al., *SpEx+: A Complete Time-Domain Speaker Extraction Network*, Interspeech 2020.
