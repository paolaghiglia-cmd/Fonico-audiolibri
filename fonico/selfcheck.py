"""Verifica rapida che il programma installato funzioni: usata dopo la costruzione per Windows."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import analysis as an
from . import audio_io
from .audio_io import SR
from .pipeline import Session
from .project import Project


def _voice_like(seconds: float = 8.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SR)) / SR
    voice = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 12))
    syllables = (np.sin(2 * np.pi * 3 * t) > 0) & (np.sin(2 * np.pi * 0.4 * t) > -0.6)
    audio = 0.1 * voice * syllables + 0.003 * rng.standard_normal(t.size)
    return audio.astype(np.float32)


def run() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        source = folder / "prova.wav"
        audio_io.save_wav(source, _voice_like())
        project = Project(titolo="Verifica", autore="Fonico", lettore="Fonico")
        project.add_files([str(source)])
        session = Session(cache=folder / "lavoro")
        session.run(project)
        written = session.export(project, folder / "finali")
        result = an.measure(audio_io.load(written[0]))
        ok = written[0].stat().st_size > 10_000 and result.rms_db > -30
        (Path(tempfile.gettempdir()) / "fonico_verifica.txt").write_text(
            f"ok={ok} rms={result.rms_db:.1f} peak={result.peak_db:.1f} noise={result.noise_db:.1f}\n"
        )
        return 0 if ok else 1
