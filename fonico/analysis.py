"""Misure sul segnale: livelli, rumore di fondo, picchi, saturazione, parlato."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.ndimage import binary_closing, binary_dilation

from .audio_io import SR

EPS = 1e-10

# Specifiche ACX/Audible
ACX_RMS_MIN = -23.0
ACX_RMS_MAX = -18.0
ACX_PEAK_MAX = -3.0
ACX_NOISE_MAX = -60.0


def db(value: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(value, EPS))


def rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    return float(db(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))))


def peak_db(audio: np.ndarray) -> float:
    return float(db(np.max(np.abs(audio)))) if audio.size else -120.0


def true_peak_db(audio: np.ndarray) -> float:
    """Picco reale stimato con sovracampionamento 4x."""
    if audio.size == 0:
        return -120.0
    up = signal.resample_poly(audio, 4, 1)
    return float(db(np.max(np.abs(up))))


def frame_rms_db(audio: np.ndarray, frame_s: float = 0.02, hop_s: float | None = None) -> np.ndarray:
    """Livello in dB di finestre consecutive."""
    frame = max(1, int(frame_s * SR))
    hop = max(1, int((hop_s or frame_s) * SR))
    if audio.size < frame:
        return np.array([rms_db(audio)])
    n = 1 + (audio.size - frame) // hop
    csum = np.concatenate([[0.0], np.cumsum(np.square(audio, dtype=np.float64))])
    starts = hop * np.arange(n)
    energy = np.maximum(csum[starts + frame] - csum[starts], 0.0) / frame
    return db(np.sqrt(energy))


def noise_floor_db(audio: np.ndarray) -> float:
    """Rumore di fondo come lo misura ACX: il mezzo secondo più silenzioso."""
    if audio.size < int(0.5 * SR):
        return rms_db(audio)
    levels = frame_rms_db(audio, frame_s=0.5, hop_s=0.05)
    return float(np.min(levels))


def speech_mask(audio: np.ndarray, frame_s: float = 0.02) -> np.ndarray:
    """True per le finestre in cui c'è voce (con un piccolo margine prima e dopo)."""
    levels = frame_rms_db(audio, frame_s)
    noise = np.percentile(levels, 10)
    loud = np.percentile(levels, 95)
    threshold = max(noise + 10.0, loud - 35.0)
    mask = levels > threshold
    mask = binary_closing(mask, structure=np.ones(int(0.15 / frame_s)))
    mask = binary_dilation(mask, structure=np.ones(int(0.12 / frame_s) * 2 + 1))
    return mask


def mask_to_samples(mask: np.ndarray, n_samples: int, frame_s: float = 0.02) -> np.ndarray:
    frame = int(frame_s * SR)
    out = np.repeat(mask, frame)
    if out.size < n_samples:
        tail = mask[-1] if mask.size else False
        out = np.concatenate([out, np.full(n_samples - out.size, tail)])
    return out[:n_samples]


def regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Intervalli [inizio, fine) dove la maschera è vera."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def periodicity(audio: np.ndarray, frame_s: float = 0.04) -> np.ndarray:
    """Per ogni finestra, quanto il suono è "intonato" (0 = soffio o rumore, 1 = vocale piena).

    È il picco dell'autocorrelazione tra 70 e 400 Hz: la voce ha un'altezza, i rumori no.
    """
    frame = int(frame_s * SR)
    n = audio.size // frame
    if n == 0:
        return np.zeros(0)
    frames = audio[: n * frame].reshape(n, frame).astype(np.float64)
    power = np.abs(np.fft.rfft(frames * np.hanning(frame), n=2 * frame, axis=1)) ** 2
    autocorr = np.fft.irfft(power, axis=1)[:, :frame]
    lags = slice(int(SR / 400), min(frame, int(SR / 70)))
    return autocorr[:, lags].max(axis=1) / np.maximum(autocorr[:, 0], 1e-20)


def digital_silence_ratio(audio: np.ndarray) -> float:
    """Quota di campioni a zero assoluto: alta quando il registratore taglia i silenzi."""
    return float(np.mean(np.abs(audio) < 1e-6)) if audio.size else 0.0


def clipping_regions(audio: np.ndarray, merge_s: float = 0.5) -> list[tuple[float, float]]:
    """Punti in cui la voce è saturata (distorta), in secondi."""
    hot = np.abs(audio) >= 0.98
    hits = []
    for start, end in regions(hot):
        if end - start >= 3 or np.max(np.abs(audio[start:end])) > 1.0:
            hits.append((start / SR, end / SR))
    merged: list[list[float]] = []
    for start, end in hits:
        if merged and start - merged[-1][1] <= merge_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


@dataclass
class Measures:
    duration_s: float
    rms_db: float
    peak_db: float
    noise_db: float
    clipping: list[tuple[float, float]] = field(default_factory=list)

    @property
    def acx_ok(self) -> bool:
        return (
            ACX_RMS_MIN <= self.rms_db <= ACX_RMS_MAX
            and self.peak_db <= ACX_PEAK_MAX
            and self.noise_db <= ACX_NOISE_MAX
        )


def measure(audio: np.ndarray, true_peak: bool = False) -> Measures:
    return Measures(
        duration_s=audio.size / SR,
        rms_db=rms_db(audio),
        peak_db=true_peak_db(audio) if true_peak else peak_db(audio),
        noise_db=noise_floor_db(audio),
        clipping=clipping_regions(audio),
    )


def speech_spectrum(audio: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Spettro medio della sola voce in dB (frequenze, livelli)."""
    if mask is None:
        mask = speech_mask(audio)
    voiced = audio[mask_to_samples(mask, audio.size)]
    if voiced.size < 4096:
        voiced = audio
    freqs, psd = signal.welch(voiced, fs=SR, nperseg=4096)
    return freqs, 10.0 * np.log10(np.maximum(psd, 1e-20))
