"""
Speaker Encoder and Voiceprint Extraction Module for Spiking Target Speech Extraction.

Wraps pre-trained SOTA ECAPA-TDNN (VoxCeleb) to extract fixed 192-dimensional
L2-normalized speaker voiceprint embeddings, with automatic resampling from 8kHz to 16kHz
and an automatic LRU/file memoization cache so extraction overhead is near zero during training.
"""

import os
import argparse
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from tqdm import tqdm


class SpeakerEncoder(nn.Module):
    """
    Pre-trained Speaker Verification Encoder (ECAPA-TDNN).
    Extracts discriminative 192-D speaker voiceprint embeddings.
    """

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str = "pretrained_models/spkrec-ecapa-voxceleb",
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.target_sample_rate = 16000
        self.embed_dim = 192

        from speechbrain.inference.speaker import EncoderClassifier
        self.classifier = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts={"device": str(self.device)},
        )
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        self._resamplers = {}

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.target_sample_rate,
            ).to(self.device)
        return self._resamplers[orig_sr]

    @torch.no_grad()
    def encode(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 8000,
    ) -> torch.Tensor:
        """
        Extracts 192-D L2-normalized speaker embedding.

        Args:
            waveform: Tensor of shape [Batch, Time] or [Batch, 1, Time] or [Time].
            sample_rate: Sample rate of input waveform (e.g. 8000).

        Returns:
            torch.Tensor: Normalized embedding of shape [Batch, 192].
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [T] -> [1, T]
        elif waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)    # [B, 1, T] -> [B, T]

        waveform = waveform.to(self.device).float()

        # Resample to 16 kHz if needed
        if sample_rate != self.target_sample_rate:
            resampler = self._get_resampler(sample_rate)
            waveform = resampler(waveform)

        # Extract embeddings using ECAPA-TDNN: [B, 1, 192]
        emb = self.classifier.encode_batch(waveform)
        emb = emb.squeeze(1)  # [B, 192]
        emb = F.normalize(emb, p=2, dim=-1)
        return emb

    @torch.no_grad()
    def encode_file(self, audio_path: str) -> torch.Tensor:
        """Loads an audio file and returns its 192-D speaker embedding [1, 192]."""
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        wav_tensor = torch.from_numpy(wav).unsqueeze(0).float()
        return self.encode(wav_tensor, sample_rate=sr)


class SpeakerCacheManager:
    """
    Manages in-memory and on-disk memoized speaker voiceprints.
    If an embedding for speaker_id is already cached, returns it in 0.001ms.
    Otherwise, extracts it on-the-fly and memoizes it.
    """

    def __init__(
        self,
        cache_file: str = "./data/minilibrimix_spk_embeddings.pt",
        encoder: Optional[SpeakerEncoder] = None,
        device: str = "cpu",
    ):
        self.cache_file = cache_file
        self.device = device
        self._encoder = encoder
        self.cache: Dict[str, torch.Tensor] = {}

        if os.path.exists(cache_file):
            try:
                self.cache = torch.load(cache_file, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"[SpeakerCache] Warning: Could not load {cache_file}: {e}")

        self._modified_count = 0

    @property
    def encoder(self) -> SpeakerEncoder:
        if self._encoder is None:
            self._encoder = SpeakerEncoder(device=self.device)
        return self._encoder

    def get_embedding(
        self,
        speaker_id: str,
        reference_audio_path: Optional[str] = None,
        reference_waveform: Optional[torch.Tensor] = None,
        sample_rate: int = 8000,
    ) -> torch.Tensor:
        """
        Retrieves 192-D embedding for speaker_id.
        Memoizes upon first computation.
        """
        if speaker_id in self.cache:
            return self.cache[speaker_id]

        # Compute on-the-fly
        if reference_audio_path and os.path.exists(reference_audio_path):
            emb = self.encoder.encode_file(reference_audio_path).squeeze(0).cpu()
        elif reference_waveform is not None:
            emb = self.encoder.encode(reference_waveform, sample_rate=sample_rate).squeeze(0).cpu()
        else:
            raise ValueError(f"Speaker '{speaker_id}' not in cache and no reference audio provided.")

        self.cache[speaker_id] = emb
        self._modified_count += 1

        # Periodic save every 50 new speakers
        if self._modified_count % 50 == 0:
            self.save()

        return emb

    def save(self):
        """Persists current cache to disk."""
        if len(self.cache) > 0 and self.cache_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
            torch.save(self.cache, self.cache_file)


# Global singleton instance helper
_DEFAULT_CACHE_MANAGER: Optional[SpeakerCacheManager] = None

def get_speaker_cache_manager(
    cache_file: str = "./data/minilibrimix_spk_embeddings.pt",
    device: str = "cpu",
) -> SpeakerCacheManager:
    global _DEFAULT_CACHE_MANAGER
    if _DEFAULT_CACHE_MANAGER is None:
        _DEFAULT_CACHE_MANAGER = SpeakerCacheManager(cache_file=cache_file, device=device)
    return _DEFAULT_CACHE_MANAGER
