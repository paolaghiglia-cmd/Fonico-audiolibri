"""Voce sintetica per i test: sillabe con formanti, sibilanti, respiri, pause, rumore."""

from __future__ import annotations

import numpy as np
from scipy import signal

from fonico.audio_io import SR


def _vowel(duration: float, f0: float, formants: tuple[float, ...], rng: np.random.Generator) -> np.ndarray:
    n = int(duration * SR)
    t = np.arange(n) / SR
    pitch = f0 * (1 + 0.04 * np.sin(2 * np.pi * 4 * t + rng.uniform(0, 6)))
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    source = signal.sawtooth(phase) + 0.05 * rng.standard_normal(n)
    out = np.zeros(n)
    for f in formants:
        b, a = signal.iirpeak(f, Q=8, fs=SR)
        out += signal.lfilter(b, a, source)
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 0.6
    return out * env


def _noise_band(duration: float, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    n = int(duration * SR)
    sos = signal.butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
    env = np.sin(np.pi * np.linspace(0, 1, n))
    return signal.sosfilt(sos, rng.standard_normal(n)) * env


def speech(
    seconds: float = 20.0,
    seed: int = 0,
    level_db: float = -20.0,
    noise_db: float = -55.0,
    hum: float | None = None,
    breaths: bool = True,
    long_pause_s: float = 0.0,
    clicks: int = 0,
    tilt_db: float = 0.0,
    clip: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = [np.zeros(int(0.8 * SR))]
    breath_spots: list[tuple[int, int]] = []
    sentence = 0
    while sum(p.size for p in pieces) < seconds * SR:
        if breaths:
            start = sum(p.size for p in pieces)
            pieces.append(np.zeros(int(0.35 * SR)))
            breath_spots.append((start, int(0.35 * SR)))
            pieces.append(np.zeros(int(0.12 * SR)))
        for _word in range(rng.integers(4, 8)):
            for _syl in range(rng.integers(1, 4)):
                if rng.random() < 0.25:
                    pieces.append(0.35 * _noise_band(0.09, 5000, 9000, rng))
                formants = (rng.uniform(400, 800), rng.uniform(1000, 2000), rng.uniform(2300, 3000))
                pieces.append(_vowel(rng.uniform(0.12, 0.22), rng.uniform(100, 140), formants, rng))
            pieces.append(np.zeros(int(rng.uniform(0.05, 0.12) * SR)))
        sentence += 1
        pause = long_pause_s if (long_pause_s and sentence == 2) else rng.uniform(0.5, 0.8)
        pieces.append(np.zeros(int(pause * SR)))
    voice = np.concatenate(pieces)

    if tilt_db:
        sos = signal.butter(1, 1500, "highpass" if tilt_db > 0 else "lowpass", fs=SR, output="sos")
        voice = voice + (10 ** (abs(tilt_db) / 20) - 1) * signal.sosfilt(sos, voice)

    active = np.abs(voice) > 1e-4
    voice *= 10 ** (level_db / 20) / np.sqrt(np.mean(voice[active] ** 2))
    for start, size in breath_spots:
        puff = _noise_band(size / SR, 500, 4000, rng)
        voice[start : start + size] += puff * 10 ** ((level_db - 16) / 20) / np.sqrt(np.mean(puff**2))

    t = np.arange(voice.size) / SR
    bg = rng.standard_normal(voice.size)
    bg = signal.sosfilt(signal.butter(1, 4000, "lowpass", fs=SR, output="sos"), bg)
    bg *= 10 ** (noise_db / 20) / np.sqrt(np.mean(bg**2))
    out = voice + bg
    if hum:
        out += 10 ** ((noise_db + 12) / 20) * (np.sin(2 * np.pi * hum * t) + 0.5 * np.sin(2 * np.pi * 2 * hum * t))
    # Gli schiocchi di bocca cadono soprattutto nelle pause, tra una parola e l'altra.
    quiet = np.flatnonzero(np.abs(voice) < 1e-4)
    quiet = quiet[(quiet > SR) & (quiet < voice.size - SR)]
    for pos in rng.choice(quiet, size=clicks, replace=False) if clicks else []:
        out[pos : pos + 6] += 0.25 * signal.windows.hann(6) * rng.choice([-1, 1])
    if clip:
        out *= 10 ** (14 / 20)
        out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32)
