"""Il fonico al lavoro: mette in fila i passaggi su tutte le tracce del progetto."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import analysis as an
from . import audio_io
from . import processing as proc
from .project import Project, Track
from .report import TrackInfo, Verdict, evaluate

Progress = Callable[[float, str], None]


@dataclass
class TrackResult:
    track: Track
    info: TrackInfo
    verdict: Verdict
    preview: Path  # WAV della traccia sistemata


@dataclass
class Session:
    """Risultati dell'ultima elaborazione, con i file intermedi nella cartella di lavoro."""

    cache: Path
    results: dict[str, TrackResult] = field(default_factory=dict)

    def _key(self, track: Track, *parts: object) -> str:
        src = Path(track.path)
        stat = src.stat()
        raw = json.dumps([str(src.resolve()), stat.st_size, stat.st_mtime_ns, *parts])
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _clean(self, track: Track, project: Project) -> tuple[np.ndarray, np.ndarray, TrackInfo]:
        """Prima fase, uguale per ogni traccia: pulizia, respiri, pause."""
        s = project.settings
        key = self._key(track, "clean", s.pulizia, s.respiri, 3)
        body_f = self.cache / f"{key}_corpo.npy"
        tone_f = self.cache / f"{key}_stanza.npy"
        info_f = self.cache / f"{key}_info.json"
        if body_f.exists() and tone_f.exists() and info_f.exists():
            data = json.loads(info_f.read_text())
            data["before"] = an.Measures(**data["before"])
            return np.load(body_f), np.load(tone_f), TrackInfo(**data)

        audio = audio_io.load(track.path)
        info = TrackInfo(
            before=an.measure(audio),
            speech_db=proc.speech_level_db(audio),
            gated=an.digital_silence_ratio(audio) > 0.02,
        )
        audio, info.hum = proc.remove_hum(audio)
        audio = proc.highpass(audio)
        audio, info.clicks = proc.declick(audio)
        audio, info.noises = proc.remove_isolated_noises(audio)
        audio = proc.denoise(audio, s.pulizia)
        audio = proc.dereverb(audio, s.pulizia)
        audio = proc.deess(audio)
        audio, info.breaths = proc.breaths(audio, s.respiri)
        tone = proc.quietest_segment(audio)
        body, info.pauses = proc.tidy_pauses(audio)

        np.save(body_f, body)
        np.save(tone_f, tone)
        data = {k: v for k, v in info.__dict__.items() if k not in ("before", "after")}
        data["before"] = info.before.__dict__
        info_f.write_text(json.dumps(data))
        return body, tone, info

    def _voice(self, body: np.ndarray, profile: np.ndarray, reference: np.ndarray, timbre: str) -> tuple[np.ndarray, float]:
        """Seconda fase: timbro uguale alle altre tracce, colore scelto, compressione."""
        audio, correction = proc.match_timbre(body, profile, reference)
        audio = proc.tone(audio, timbre)
        return proc.compress(audio), correction

    def run(self, project: Project, progress: Progress | None = None) -> dict[str, TrackResult]:
        progress = progress or (lambda f, m: None)
        self.cache.mkdir(parents=True, exist_ok=True)
        tracks = project.tracks
        total = max(1, len(tracks) * 2)
        cleaned: dict[str, tuple[np.ndarray, np.ndarray, TrackInfo]] = {}
        profiles: dict[str, np.ndarray] = {}

        for i, t in enumerate(tracks):
            progress(i / total, f"Pulisco la voce: {t.name}")
            body, tone, info = self._clean(t, project)
            cleaned[t.path] = (body, tone, info)
            profiles[t.path] = proc.band_profile(body)

        actor = np.median(np.stack(list(profiles.values())), axis=0)
        ref_path = project.settings.riferimento
        if ref_path in profiles:
            reference = proc.reference_profile([profiles[ref_path]])
        else:
            reference = proc.reference_profile(list(profiles.values()))

        self.results = {}
        for i, t in enumerate(tracks):
            progress((len(tracks) + i) / total, f"Uniformo voce e volume: {t.name}")
            body, tone, info = cleaned[t.path]
            voice, _ = self._voice(body, profiles[t.path], reference, project.settings.timbro)
            # quanto questa traccia si allontana dalle altre dell'attore (non dalla voce da studio)
            info.timbre_db = float(np.mean(np.abs(profiles[t.path] - actor)[proc.BAND_CENTERS < 6000]))
            np.save(self._voice_file(t), voice)
            np.save(self._tone_file(t), tone)
            final = proc.finalize([voice], tone, fill_silence=info.gated)
            info.after = an.measure(final, true_peak=True)
            preview = self.cache / f"{self._key(t, 'anteprima')}.wav"
            audio_io.save_wav(preview, final)
            self.results[t.path] = TrackResult(t, info, evaluate(info), preview)
        progress(1.0, "Fatto")
        return self.results

    def _voice_file(self, t: Track) -> Path:
        return self.cache / f"{self._key(t, 'voce')}.npy"

    def _tone_file(self, t: Track) -> Path:
        return self.cache / f"{self._key(t, 'voce')}_stanza.npy"

    def export(self, project: Project, out_dir: str | Path, progress: Progress | None = None) -> list[Path]:
        """Scrive gli MP3 finali: uno per traccia, oppure uno per capitolo."""
        progress = progress or (lambda f, m: None)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        units = project.export_units()
        written = []
        for n, (name, tracks) in enumerate(units, start=1):
            progress((n - 1) / len(units), f"Esporto: {name}")
            missing = [t.name for t in tracks if t.path not in self.results]
            if missing:
                raise audio_io.AudioError(f"Prima premi \"Sistema la voce\": mancano {', '.join(missing)}.")
            bodies = [np.load(self._voice_file(t)) for t in tracks]
            tone = np.load(self._tone_file(tracks[0]))
            gated = any(self.results[t.path].info.gated for t in tracks)
            final = proc.finalize(bodies, tone, fill_silence=gated)
            title = f"{project.titolo} - {name}" if project.titolo else name
            tags = {
                "title": title,
                "album": project.titolo,
                "artist": project.autore,
                "album_artist": project.autore,
                "composer": project.lettore,
                "comment": f"Letto da {project.lettore}" if project.lettore else "",
                "genre": "Audiobook",
                "track": f"{n}/{len(units)}",
            }
            path = out_dir / f"{n:03d} - {_safe(name)}.mp3"
            audio_io.export_mp3(path, final, tags)
            written.append(path)
        progress(1.0, "Esportazione completata")
        return written


def _safe(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", name).strip() or "traccia"
