"""Il progetto dell'audiolibro: dati del libro, tracce, capitoli e scelte dell'attore."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_SUFFIX = ".fonico"
AUDIO_SUFFIXES = {".m4a", ".mp4", ".aac", ".wav", ".mp3", ".flac", ".ogg"}

CLEANING = ("leggera", "normale", "forte")
BREATHS = ("lasciali", "attenuali", "toglili")
TONES = ("caldo", "naturale", "chiaro")


@dataclass
class Settings:
    pulizia: str = "normale"
    respiri: str = "attenuali"
    timbro: str = "naturale"
    riferimento: str | None = None  # percorso della traccia di riferimento; None = automatico


@dataclass
class Track:
    path: str
    capitolo: int | None = None

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def stem(self) -> str:
        return Path(self.path).stem


@dataclass
class Project:
    titolo: str = ""
    autore: str = ""
    lettore: str = ""
    tracks: list[Track] = field(default_factory=list)
    settings: Settings = field(default_factory=Settings)
    file: str | None = None

    def add_files(self, paths: list[str]) -> list[str]:
        """Aggiunge i file audio, ordinati per nome; restituisce quelli scartati."""
        known = {t.path for t in self.tracks}
        rejected = []
        for p in sorted(paths, key=natural_key):
            if Path(p).suffix.lower() not in AUDIO_SUFFIXES:
                rejected.append(p)
            elif p not in known:
                self.tracks.append(Track(path=p))
                known.add(p)
        return rejected

    def merge_into_chapter(self, indexes: list[int]) -> None:
        """Unisce le tracce scelte (contigue o no) in un nuovo capitolo, mettendole vicine."""
        if not indexes:
            return
        indexes = sorted(indexes)
        number = 1 + max((t.capitolo or 0) for t in self.tracks)
        chosen = [self.tracks[i] for i in indexes]
        for t in chosen:
            t.capitolo = number
        first = indexes[0]
        rest = [t for i, t in enumerate(self.tracks) if i not in indexes]
        self.tracks = rest[:first] + chosen + rest[first:]
        self.renumber_chapters()

    def split_chapter(self, indexes: list[int]) -> None:
        for i in indexes:
            self.tracks[i].capitolo = None
        self.renumber_chapters()

    def renumber_chapters(self) -> None:
        mapping: dict[int, int] = {}
        for t in self.tracks:
            if t.capitolo is not None and t.capitolo not in mapping:
                mapping[t.capitolo] = len(mapping) + 1
        for t in self.tracks:
            if t.capitolo is not None:
                t.capitolo = mapping[t.capitolo]

    def export_units(self) -> list[tuple[str, list[Track]]]:
        """Ogni file da esportare: (nome, tracce). Le tracce di uno stesso capitolo diventano un file solo."""
        units: list[tuple[str, list[Track]]] = []
        for t in self.tracks:
            if t.capitolo is not None and units and units[-1][1][0].capitolo == t.capitolo:
                units[-1][1].append(t)
            elif t.capitolo is not None:
                units.append((f"Capitolo {t.capitolo}", [t]))
            else:
                units.append((t.stem, [t]))
        return units

    def save(self, path: str | Path) -> None:
        data = asdict(self)
        data.pop("file")
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.file = str(path)

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        project = cls(
            titolo=data.get("titolo", ""),
            autore=data.get("autore", ""),
            lettore=data.get("lettore", ""),
            tracks=[Track(**t) for t in data.get("tracks", [])],
            settings=Settings(**data.get("settings", {})),
        )
        project.file = str(path)
        return project


def natural_key(path: str) -> list:
    """Ordina "traccia 2" prima di "traccia 10"."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", Path(path).name)]
