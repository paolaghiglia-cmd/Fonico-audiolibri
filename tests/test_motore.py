import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fonico import analysis as an
from fonico import audio_io, processing as proc
from fonico.audio_io import SR
from fonico.pipeline import Session
from fonico.project import Project
from fonico.report import ROSSO, VERDE

from synth import speech

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="serve ffmpeg")


def write_m4a(folder: Path, name: str, audio: np.ndarray) -> str:
    wav = folder / f"{name}.wav"
    audio_io.save_wav(wav, audio)
    m4a = folder / f"{name}.m4a"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "160k", str(m4a)],
        check=True,
    )
    return str(m4a)


@needs_ffmpeg
def test_legge_m4a_in_mono_44k(tmp_path):
    source = speech(seconds=5)
    audio = audio_io.load(write_m4a(tmp_path, "voce", source))
    assert audio.dtype == np.float32
    assert abs(audio.size - source.size) < 0.1 * SR


def test_toglie_il_ronzio_elettrico():
    x = speech(seconds=10, hum=50)
    assert proc.detect_hum(x) == 50
    y, f0 = proc.remove_hum(x)
    assert f0 == 50
    assert proc.detect_hum(y) is None
    assert proc.detect_hum(speech(seconds=10)) is None


def test_pulizia_abbassa_il_rumore_di_fondo():
    x = speech(seconds=10, noise_db=-45)
    before = an.noise_floor_db(x)
    after = an.noise_floor_db(proc.denoise(x, "normale"))
    assert after < before - 10
    # la voce resta: il livello generale cambia poco
    assert abs(an.rms_db(proc.denoise(x)) - an.rms_db(x)) < 1.5


def test_click_attenuati():
    x = speech(seconds=10, clicks=6, noise_db=-60)
    y, count = proc.declick(x)
    assert count >= 5
    _, false_hits = proc.declick(speech(seconds=10, noise_db=-60))
    assert false_hits == 0


def test_trova_i_respiri_e_non_la_voce():
    with_breaths = proc.denoise(proc.highpass(speech(seconds=20, seed=4)))
    without = proc.denoise(proc.highpass(speech(seconds=20, seed=4, breaths=False)))
    _, found = proc.breaths(with_breaths, "attenuali")
    _, false_hits = proc.breaths(without, "attenuali")
    assert found >= 3
    assert false_hits == 0
    same, none = proc.breaths(with_breaths, "lasciali")
    assert none == 0 and same is with_breaths


def test_accorcia_le_pause_lunghe_e_toglie_silenzi_ai_lati():
    x = speech(seconds=15, long_pause_s=4.0, breaths=False)
    y, shortened = proc.tidy_pauses(x)
    assert shortened == 1
    assert x.size - y.size > 2.5 * SR


def test_timbro_avvicinato_al_riferimento():
    ref = speech(seconds=15, seed=1)
    dull = speech(seconds=15, seed=2, tilt_db=-8)
    ref_p, dull_p = proc.band_profile(ref), proc.band_profile(dull)
    matched, correction = proc.match_timbre(dull, dull_p, ref_p)
    assert correction > 1.0
    gap_before = np.abs(dull_p - ref_p).mean()
    gap_after = np.abs(proc.band_profile(matched) - ref_p).mean()
    assert gap_after < gap_before * 0.7


@pytest.mark.parametrize("level_db", [-35.0, -20.0, -12.0])
def test_finale_rispetta_le_specifiche_audible(level_db):
    x = speech(seconds=20, level_db=level_db, noise_db=level_db - 35)
    clean = proc.denoise(proc.highpass(x))
    tone = proc.quietest_segment(clean)
    body, _ = proc.tidy_pauses(clean)
    final = proc.finalize([proc.compress(body)], tone)
    m = an.measure(final, true_peak=True)
    assert an.ACX_RMS_MIN <= m.rms_db <= an.ACX_RMS_MAX
    assert m.peak_db <= an.ACX_PEAK_MAX
    assert m.noise_db <= an.ACX_NOISE_MAX


def test_saturazione_segnalata_in_rosso():
    assert an.clipping_regions(speech(seconds=8, clip=True))
    assert not an.clipping_regions(speech(seconds=8))


@needs_ffmpeg
def test_progetto_completo_con_capitolo(tmp_path):
    paths = [
        write_m4a(tmp_path, "traccia 1", speech(seconds=12, seed=1, level_db=-22)),
        write_m4a(tmp_path, "traccia 2", speech(seconds=12, seed=2, level_db=-30, tilt_db=5)),
        write_m4a(tmp_path, "traccia 10", speech(seconds=12, seed=3, level_db=-18)),
    ]
    project = Project(titolo="Il libro", autore="Autrice", lettore="Attore")
    project.add_files(list(reversed(paths)))
    assert [t.stem for t in project.tracks] == ["traccia 1", "traccia 2", "traccia 10"]
    project.merge_into_chapter([0, 1])

    session = Session(cache=tmp_path / "lavoro")
    results = session.run(project)
    assert len(results) == 3
    for r in results.values():
        assert r.info.after.acx_ok, r.info.after
        assert r.preview.exists()

    written = session.export(project, tmp_path / "finali")
    assert [p.name for p in written] == ["001 - Capitolo 1.mp3", "002 - traccia 10.mp3"]
    for p in written:
        m = an.measure(audio_io.load(p), true_peak=True)
        assert an.ACX_RMS_MIN <= m.rms_db <= an.ACX_RMS_MAX
        assert m.peak_db <= an.ACX_PEAK_MAX + 0.3  # tolleranza della codifica MP3
        assert m.noise_db <= an.ACX_NOISE_MAX

    if shutil.which("ffprobe"):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags:stream=sample_rate,channels,bit_rate",
             "-of", "json", str(written[0])],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(probe.stdout)
        stream = info["streams"][0]
        assert stream["sample_rate"] == "44100" and stream["channels"] == 1
        assert stream["bit_rate"] == "192000"
        assert info["format"]["tags"]["album"] == "Il libro"

    # una seconda elaborazione riusa il lavoro già fatto
    assert any((tmp_path / "lavoro").glob("*_corpo.npy"))
    session.run(project)


def test_verdetti():
    from fonico.report import TrackInfo, evaluate

    good = an.Measures(duration_s=60, rms_db=-25, peak_db=-6, noise_db=-65)
    assert evaluate(TrackInfo(before=good)).color == VERDE
    clipped = an.Measures(duration_s=60, rms_db=-10, peak_db=0, noise_db=-65, clipping=[(80.0, 81.0)])
    verdict = evaluate(TrackInfo(before=clipped))
    assert verdict.color == ROSSO
    assert "1:20" in verdict.messages[0]


def test_progetto_salvato_e_riaperto(tmp_path):
    project = Project(titolo="Titolo", autore="A", lettore="L")
    rejected = project.add_files(["b.m4a", "a.m4a", "note.txt"])
    assert rejected == ["note.txt"]
    project.merge_into_chapter([1])
    project.settings.respiri = "toglili"
    f = tmp_path / "libro.fonico"
    project.save(f)
    again = Project.load(f)
    assert [t.name for t in again.tracks] == ["a.m4a", "b.m4a"]
    assert again.tracks[1].capitolo == 1
    assert again.settings.respiri == "toglili"
    again.split_chapter([1])
    assert again.export_units()[1][0] == "b"


def test_registratore_che_taglia_i_silenzi_riceve_un_fondo_di_stanza():
    # Molti registratori (telefono, Registratore di Windows) mettono a zero assoluto le pause.
    x = speech(seconds=15, level_db=-48, noise_db=-120)
    x[np.abs(x) < 3e-4] = 0.0
    assert an.digital_silence_ratio(x) > 0.02
    body, _ = proc.tidy_pauses(proc.highpass(x))
    final = proc.finalize([proc.compress(body)], np.zeros(0, dtype=np.float32), fill_silence=True)
    m = an.measure(final, true_peak=True)
    assert an.digital_silence_ratio(final) < 0.01
    assert -75.0 < m.noise_db <= an.ACX_NOISE_MAX
    assert an.ACX_RMS_MIN <= m.rms_db <= an.ACX_RMS_MAX
