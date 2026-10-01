"""Lettura e scrittura dei file audio tramite FFmpeg."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR = 44100


class AudioError(RuntimeError):
    """Errore leggibile dall'attore, senza termini tecnici."""


def ffmpeg_path() -> str:
    """Trova ffmpeg: variabile d'ambiente, copia inclusa nell'installazione, oppure PATH."""
    env = os.environ.get("FONICO_FFMPEG")
    if env and Path(env).exists():
        return env
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    bundle_dirs = []
    if hasattr(sys, "_MEIPASS"):
        bundle_dirs.append(Path(sys._MEIPASS))
    if getattr(sys, "frozen", False):
        bundle_dirs.append(Path(sys.executable).parent)
    for d in bundle_dirs:
        candidate = d / exe
        if candidate.exists():
            return str(candidate)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise AudioError("Manca il componente FFmpeg: reinstalla l'applicazione.")


def _run(args: list[str], input_bytes: bytes | None = None) -> bytes:
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    proc = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-loglevel", "error", *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise AudioError(f"Non riesco a leggere o scrivere il file audio.\n{detail}")
    return proc.stdout


def load(path: str | Path) -> np.ndarray:
    """Decodifica qualsiasi file audio (m4a, wav, mp3...) in mono float32 a 44,1 kHz."""
    raw = _run(["-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"])
    audio = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if audio.size == 0:
        raise AudioError(f"Il file {Path(path).name} sembra vuoto.")
    return audio


def save_wav(path: str | Path, audio: np.ndarray) -> None:
    """Salva un WAV 16 bit per l'ascolto di prova."""
    pcm = np.clip(audio, -1.0, 1.0)
    wavfile.write(str(path), SR, (pcm * 32767).astype(np.int16))


def export_mp3(path: str | Path, audio: np.ndarray, tags: dict[str, str] | None = None) -> None:
    """Esporta un MP3 192 kbps CBR, 44,1 kHz, mono, con metadati ID3."""
    meta: list[str] = []
    for key, value in (tags or {}).items():
        if value:
            meta += ["-metadata", f"{key}={value}"]
    data = np.clip(audio, -1.0, 1.0).astype("<f4").tobytes()
    _run(
        [
            "-y",
            "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
            "-c:a", "libmp3lame", "-b:a", "192k", "-ar", str(SR), "-ac", "1",
            "-id3v2_version", "3", "-write_xing", "1",
            *meta,
            str(path),
        ],
        input_bytes=data,
    )
