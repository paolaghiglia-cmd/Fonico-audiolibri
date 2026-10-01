"""I singoli passaggi del "fonico": ognuno riceve e restituisce audio mono float32."""

from __future__ import annotations

import numpy as np
import pedalboard as pb
from scipy import signal
from scipy.ndimage import binary_closing, maximum_filter1d, minimum_filter1d, uniform_filter, uniform_filter1d

from . import analysis as an
from .audio_io import SR

# Livelli di lavoro
SPEECH_TARGET_DB = -18.0  # livello medio della voce prima della compressione
FINAL_RMS_DB = -20.0  # centro della finestra ACX (-23 / -18)
FINAL_PEAK_DB = -3.3  # margine sotto il limite ACX di -3 dB
FINAL_NOISE_DB = -65.0  # margine sotto il limite ACX di -60 dB
ROOM_FILL_DB = -70.0  # fondo di stanza aggiunto quando il registratore lascia silenzi a zero assoluto
HEAD_S = 0.75
TAIL_S = 2.0

DENOISE = {"leggera": (1.2, 10.0), "normale": (1.6, 16.0), "forte": (2.2, 24.0)}
BREATH_GAIN_DB = {"attenuali": -15.0}


def _fade_envelope(gain: np.ndarray, fade_s: float = 0.01) -> np.ndarray:
    size = max(1, int(fade_s * SR))
    return uniform_filter1d(gain.astype(np.float32), size=size, mode="nearest")


# --- Pulizia --------------------------------------------------------------


def highpass(audio: np.ndarray, cutoff: float = 100.0) -> np.ndarray:
    """Toglie i rimbombi sotto la voce (passi, traffico, vibrazioni del microfono)."""
    sos = signal.butter(4, cutoff, "highpass", fs=SR, output="sos")
    return signal.sosfilt(sos, audio).astype(np.float32)


def detect_hum(audio: np.ndarray) -> float | None:
    """Restituisce 50 o 60 se c'è un ronzio elettrico, altrimenti None."""
    if audio.size < 16384:
        return None
    freqs, psd = signal.welch(audio, fs=SR, nperseg=16384)
    level = 10 * np.log10(np.maximum(psd, 1e-20))
    for f0 in (50.0, 60.0):
        prominences = []
        for k in (1, 2, 3):
            fk = f0 * k
            peak = level[np.abs(freqs - fk) <= 1.5].max()
            around = level[(np.abs(freqs - fk) > 4) & (np.abs(freqs - fk) < 12)]
            prominences.append(peak - np.median(around))
        if max(prominences) > 12.0:
            return f0
    return None


def remove_hum(audio: np.ndarray) -> tuple[np.ndarray, float | None]:
    f0 = detect_hum(audio)
    if f0 is None:
        return audio, None
    out = audio.astype(np.float64)
    k = 1
    while f0 * k < 1000:
        b, a = signal.iirnotch(f0 * k, Q=35, fs=SR)
        out = signal.filtfilt(b, a, out)
        k += 1
    return out.astype(np.float32), f0


def deplosive(audio: np.ndarray, max_red_db: float = 30.0) -> tuple[np.ndarray, int]:
    """Toglie i colpi sordi: "p" esplosive sul microfono, urti, mani sul telefono o sul tavolo.

    Si riconoscono perché per un attimo i bassissimi (sotto 160 Hz) superano di molto la voce stessa.
    Una voce maschile profonda può avere i bassi alla pari della voce: quella non va toccata.
    """
    nper, hop = 2048, 512
    freqs, _, spec = signal.stft(audio, fs=SR, nperseg=nper, noverlap=nper - hop)
    power = np.abs(spec) ** 2
    low_bins = freqs < 160
    low = power[low_bins].sum(axis=0) + 1e-20
    ref = power[(freqs >= 200) & (freqs < 2000)].sum(axis=0) + 1e-20
    ratio = 10 * np.log10(low / ref)
    level = 10 * np.log10(power.sum(axis=0) + 1e-20)
    audible = level > np.percentile(level, 95) - 35  # nel silenzio il rapporto non significa nulla
    # La voce profonda ha il picco dei bassi sulla sua nota (120-160 Hz); un colpo sordo sta più in basso
    # (60-100 Hz) e domina di 20-30 dB.
    peak_hz = freqs[low_bins][np.argmax(power[low_bins], axis=0)]
    thump = audible & ((ratio > 20.0) | ((ratio > 10.0) & (peak_hz < 110)))
    if not thump.any():
        return audio, 0
    red = np.where(thump, np.clip(ratio + 6.0, 0.0, max_red_db), 0.0)  # riporta i bassi 6 dB sotto la voce
    red = maximum_filter1d(red, size=3, mode="nearest")
    gains = np.ones(power.shape, dtype=np.float32)
    gains[low_bins] = 10 ** (-red[None, :] / 20)
    _, out = signal.istft(spec * gains, fs=SR, nperseg=nper, noverlap=nper - hop)
    events = len(an.regions(thump))
    return out[: audio.size].astype(np.float32), events


def declick(audio: np.ndarray, max_click_s: float = 0.003) -> tuple[np.ndarray, int]:
    """Attenua i click e gli schiocchi di bocca: impulsi brevissimi e acuti."""
    sos_hp = signal.butter(4, 3000, "highpass", fs=SR, output="sos")
    hp = signal.sosfilt(sos_hp, audio)
    block = int(0.001 * SR)
    n = hp.size // block
    if n < 50:
        return audio, 0
    energy = np.mean(hp[: n * block].reshape(n, block) ** 2, axis=1)
    # Un click spicca anche rispetto al punto più forte dei 3-15 ms intorno: la voce no,
    # perché i suoi impulsi si ripetono a ogni periodo.
    near = maximum_filter1d(energy, size=13, mode="nearest")
    padded = np.pad(near, 9, mode="edge")
    ring = np.maximum(padded[:-18], padded[18:])
    floor = np.percentile(energy, 50) * 4 + 1e-14
    spikes = (energy > 15.0 * ring) & (energy > floor)
    found = [
        (start * block, end * block)
        for start, end in an.regions(spikes)
        if (end - start) <= max_click_s * 1000
    ]
    if not found:
        return audio, 0
    out = audio.copy()
    sos_lp = signal.butter(4, 2500, "lowpass", fs=SR, output="sos")
    pad = int(0.001 * SR)
    ctx = int(0.01 * SR)
    for start, end in found:
        a, b = max(0, start - pad), min(audio.size, end + pad)
        ca, cb = max(0, a - ctx), min(audio.size, b + ctx)
        smooth = signal.sosfiltfilt(sos_lp, audio[ca:cb])[a - ca : b - ca]
        n = b - a
        w = np.ones(n, dtype=np.float32)
        ramp = min(pad, n // 2)
        if ramp:
            w[:ramp] = np.linspace(0, 1, ramp)
            w[-ramp:] = np.linspace(1, 0, ramp)
        out[a:b] = (1 - w) * audio[a:b] + w * smooth
    return out, len(found)


def denoise(audio: np.ndarray, strength: str = "normale") -> np.ndarray:
    """Riduzione del rumore di fondo per sottrazione spettrale, con profilo preso dalle pause."""
    if an.noise_floor_db(audio) < -78.0:
        return audio
    beta, max_red_db = DENOISE[strength]
    _, _, spec = signal.stft(audio, fs=SR, nperseg=2048, noverlap=1536)
    power = np.abs(spec) ** 2
    frame_energy = power.mean(axis=0)
    quiet = frame_energy <= np.percentile(frame_energy, 10)
    noise = power[:, quiet].mean(axis=1, keepdims=True)
    gain = np.sqrt(np.clip(1.0 - beta * noise / np.maximum(power, 1e-20), 0.0, 1.0))
    gain = uniform_filter(gain, size=(3, 5), mode="nearest")
    gain = np.maximum(gain, 10 ** (-max_red_db / 20)).astype(np.float32)
    _, out = signal.istft(spec * gain, fs=SR, nperseg=2048, noverlap=1536)
    return out[: audio.size].astype(np.float32)


def deess(audio: np.ndarray, max_red_db: float = 8.0) -> np.ndarray:
    """Attenua le "s" e le "z" troppo taglienti."""
    freqs, _, spec = signal.stft(audio, fs=SR, nperseg=1024, noverlap=768)
    band = (freqs >= 4500) & (freqs <= 10000)
    power = np.abs(spec) ** 2
    band_e = power[band].sum(axis=0)
    total_e = power.sum(axis=0) + 1e-20
    band_db = 10 * np.log10(band_e + 1e-20)
    total_db = 10 * np.log10(total_e)
    active = total_db > np.percentile(total_db, 95) - 35
    if not active.any():
        return audio
    reference = np.median(band_db[active])
    excess = band_db - (reference + 6.0)
    sibilant = (band_e / total_e > 0.45) & (excess > 0) & active
    red = np.where(sibilant, np.clip(excess * 0.6, 0, max_red_db), 0.0)
    red = uniform_filter1d(red, size=3, mode="nearest")
    gains = np.ones(power.shape, dtype=np.float32)
    gains[band] = 10 ** (-red[None, :] / 20)
    _, out = signal.istft(spec * gains, fs=SR, nperseg=1024, noverlap=768)
    return out[: audio.size].astype(np.float32)


def breaths(audio: np.ndarray, mode: str = "attenuali") -> tuple[np.ndarray, int]:
    """Trova i respiri (soffi senza voce vicini alle pause) e li abbassa o li toglie."""
    if mode == "lasciali":
        return audio, 0
    frame = int(0.02 * SR)
    n = audio.size // frame
    if n < 10:
        return audio, 0
    frames = audio[: n * frame].reshape(n, frame).astype(np.float64)
    levels = an.db(np.sqrt(np.mean(frames**2, axis=1)))
    spectrum = np.fft.rfft(frames * np.hanning(frame), n=2 * frame, axis=1)
    power = np.abs(spectrum) ** 2 + 1e-20
    freqs = np.fft.rfftfreq(2 * frame, 1 / SR)
    sel = (freqs >= 300) & (freqs <= 10000)
    centroid = (power[:, sel] * freqs[sel]).sum(axis=1) / power[:, sel].sum(axis=1)
    # Periodicità: la voce ha un'altezza (picco di autocorrelazione tra 70 e 400 Hz), il respiro no.
    autocorr = np.fft.irfft(power, axis=1)[:, :frame]
    lags = slice(int(SR / 400), int(SR / 70))
    periodicity = autocorr[:, lags].max(axis=1) / np.maximum(autocorr[:, 0], 1e-20)

    noise = np.percentile(levels, 10)
    voiced = (levels > noise + 12) & (periodicity > 0.5)
    speech_level = np.median(levels[voiced]) if voiced.any() else levels.max()
    silent = levels <= noise + 6
    candidate = (
        ~silent
        & (levels < speech_level - 10)
        & (periodicity < 0.4)
        & (centroid < 5000)
    )
    # Riempie i buchi di una finestra dentro lo stesso respiro.
    candidate = binary_closing(candidate, structure=np.ones(3))

    gain = np.ones(n, dtype=np.float32)
    count = 0
    for start, end in an.regions(candidate):
        dur = (end - start) * 0.02
        if not 0.2 <= dur <= 1.2:
            continue
        before = silent[max(0, start - 3) : start].any() or start == 0
        after = silent[end : end + 3].any() or end == n
        if not (before or after):
            continue
        if levels[start:end].max() > speech_level - 4:
            continue
        if mode == "toglili":
            g = 10 ** ((noise - levels[start:end].mean()) / 20)
        else:
            g = 10 ** (BREATH_GAIN_DB["attenuali"] / 20)
        gain[start:end] = min(1.0, max(g, 0.01))
        count += 1
    if not count:
        return audio, 0
    env = np.ones(audio.size, dtype=np.float32)
    env[: n * frame] = np.repeat(gain, frame)
    return (audio * _fade_envelope(env, 0.015)).astype(np.float32), count


DEREVERB = {"leggera": 0.6, "normale": 1.0, "forte": 1.4}


def dereverb(audio: np.ndarray, strength: str = "normale", t60_s: float = 0.5) -> np.ndarray:
    """Riduce il rimbombo della stanza: stima la coda di riverbero dal suono di pochi istanti prima
    e la sottrae (sottrazione spettrale del riverbero tardivo)."""
    beta = DEREVERB[strength]
    nper, hop = 1024, 256
    _, _, spec = signal.stft(audio, fs=SR, nperseg=nper, noverlap=nper - hop)
    power = np.abs(spec) ** 2
    smooth = uniform_filter1d(power, size=3, axis=1, mode="nearest")
    delay = max(1, int(0.05 * SR / hop))  # la coda "tardiva" inizia ~50 ms dopo il suono diretto
    decay = 10 ** (-6.0 * (delay * hop / SR) / t60_s)  # quanto si è spenta la stanza in quel tempo
    late = np.zeros_like(smooth)
    late[:, delay:] = decay * smooth[:, :-delay]
    gain = np.sqrt(np.clip(1.0 - beta * late / np.maximum(power, 1e-20), 0.0, 1.0))
    gain = uniform_filter(gain, size=(3, 3), mode="nearest")
    gain = np.maximum(gain, 10 ** (-12 / 20)).astype(np.float32)
    _, out = signal.istft(spec * gain, fs=SR, nperseg=nper, noverlap=nper - hop)
    return out[: audio.size].astype(np.float32)


def remove_isolated_noises(
    audio: np.ndarray, max_s: float = 0.09, max_unvoiced_s: float = 0.2, gap_s: float = 0.1
) -> tuple[np.ndarray, int]:
    """Toglie i rumori brevi e isolati nelle pause: click del mouse, tasti, colpi sul tavolo.

    Una parola, anche brevissima, dura più di 90 ms; un click del mouse è un lampo circondato dal silenzio.
    """
    frame_s = 0.005
    levels = an.frame_rms_db(audio, frame_s)
    loud = np.percentile(levels, 90)
    active = levels > max(np.percentile(levels, 10) + 8, loud - 40)
    regs = an.regions(active)
    frame = int(frame_s * SR)
    vf = 0.04
    voicing = an.periodicity(audio, vf)
    gain = np.ones(audio.size, dtype=np.float32)
    count = 0
    for i, (a, b) in enumerate(regs):
        dur = (b - a) * frame_s
        seg = voicing[int(a * frame_s / vf) : max(int(a * frame_s / vf) + 1, int(np.ceil(b * frame_s / vf)))]
        voiced = seg.size and np.median(seg) > 0.5
        if dur > (max_s if voiced else max_unvoiced_s):
            continue
        before = (a - regs[i - 1][1]) if i else a + 10**6
        after = (regs[i + 1][0] - b) if i + 1 < len(regs) else 10**6
        if before * frame_s < gap_s or after * frame_s < gap_s:
            continue
        s, e = max(0, (a - 2) * frame), min(audio.size, (b + 4) * frame)
        gain[s:e] = 0.0
        count += 1
    if not count:
        return audio, 0
    return (audio * _fade_envelope(gain, 0.004)).astype(np.float32), count


def clean_edges(audio: np.ndarray, before_s: float = 0.25, after_s: float = 0.4) -> tuple[np.ndarray, bool]:
    """Silenzia quello che c'è prima della prima parola e dopo l'ultima: il click per avviare e
    fermare la registrazione, il fruscio del telefono preso in mano, i rumori di sistemazione."""
    vf = 0.04
    voicing = an.periodicity(audio, vf)
    levels = an.frame_rms_db(audio, vf)[: voicing.size]
    loud = np.percentile(levels, 95)
    voiced = (voicing > 0.6) & (levels > loud - 30)
    voiced = binary_closing(voiced, structure=np.ones(3))
    words = [(a, b) for a, b in an.regions(voiced) if (b - a) * vf >= 0.2]
    if not words:
        return audio, False
    start = max(0, int((words[0][0] * vf - before_s) * SR))
    end = min(audio.size, int((words[-1][1] * vf + after_s) * SR))
    gain = np.zeros(audio.size, dtype=np.float32)
    gain[start:end] = 1.0
    changed = bool(np.any(np.abs(audio[:start]) > 1e-4) or np.any(np.abs(audio[end:]) > 1e-4))
    return (audio * _fade_envelope(gain, 0.03)).astype(np.float32), changed


# --- Pause e ambiente ------------------------------------------------------


def quietest_segment(audio: np.ndarray, seconds: float = 0.5) -> np.ndarray:
    """Il tratto più silenzioso: è il "silenzio della stanza" da usare in testa e in coda."""
    size = int(seconds * SR)
    if audio.size <= size:
        return audio.copy()
    levels = an.frame_rms_db(audio, frame_s=seconds, hop_s=0.05)
    start = int(np.argmin(levels) * 0.05 * SR)
    return audio[start : start + size].copy()


def room_tone(tone: np.ndarray, seconds: float) -> np.ndarray:
    """Allunga il silenzio della stanza alla durata richiesta, con dissolvenze invisibili."""
    size = int(seconds * SR)
    if tone.size < int(0.1 * SR):
        return np.zeros(size, dtype=np.float32)
    fade = min(int(0.05 * SR), tone.size // 4)
    out = tone[: min(size, tone.size)].copy()
    ramp_in = np.linspace(0, 1, fade, dtype=np.float32)
    while out.size < size:
        nxt = tone.copy()
        out[-fade:] = out[-fade:] * ramp_in[::-1] + nxt[:fade] * ramp_in
        out = np.concatenate([out, nxt[fade:]])
    return out[:size].astype(np.float32)


def tidy_pauses(
    audio: np.ndarray, max_pause_s: float = 1.6, keep_pause_s: float = 1.2
) -> tuple[np.ndarray, int]:
    """Toglie i silenzi in testa e in coda e accorcia le pause troppo lunghe."""
    mask = an.mask_to_samples(an.speech_mask(audio), audio.size)
    if not mask.any():
        return audio, 0
    speech = an.regions(mask)
    first, last = speech[0][0], speech[-1][1]
    body = audio[first:last]
    mask = mask[first:last]
    pieces: list[np.ndarray] = []
    cursor = 0
    shortened = 0
    keep_half = int(keep_pause_s * SR / 2)
    fade = int(0.03 * SR)
    for start, end in an.regions(~mask):
        if end - start <= max_pause_s * SR:
            continue
        pieces.append(body[cursor : start + keep_half])
        cursor = end - keep_half
        shortened += 1
    pieces.append(body[cursor:])
    out = pieces[0]
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    for piece in pieces[1:]:
        if out.size < fade or piece.size < fade:
            out = np.concatenate([out, piece])
            continue
        joint = out[-fade:] * ramp[::-1] + piece[:fade] * ramp
        out = np.concatenate([out[:-fade], joint, piece[fade:]])
    return out.astype(np.float32), shortened


# --- Timbro ---------------------------------------------------------------

BAND_CENTERS = 1000.0 * 2.0 ** (np.arange(-10, 11) / 3.0)  # da 100 Hz a 10 kHz, terzi d'ottava


def band_profile(audio: np.ndarray) -> np.ndarray:
    """Impronta del timbro della voce: livello medio per terzo d'ottava, centrato sullo zero."""
    freqs, level = an.speech_spectrum(audio)
    out = []
    for fc in BAND_CENTERS:
        sel = (freqs >= fc / 2 ** (1 / 6)) & (freqs < fc * 2 ** (1 / 6))
        out.append(level[sel].mean() if sel.any() else -120.0)
    profile = np.array(out)
    return profile - profile.mean()


# Spettro medio di una voce parlata ben registrata (LTASS, Byrne et al. 1994, valori arrotondati),
# negli stessi terzi d'ottava di BAND_CENTERS.
SPEECH_TARGET = np.array(
    [49, 54, 58, 59, 60, 60, 60, 59, 56, 53, 51, 50, 49, 47, 46, 45, 44, 43, 41, 40, 38], dtype=float
)
SPEECH_TARGET -= SPEECH_TARGET.mean()


def reference_profile(profiles: list[np.ndarray], pull: float = 0.8) -> np.ndarray:
    """Il timbro a cui portare tutte le tracce: la voce dell'attore, avvicinata a una voce da studio.

    Senza questa spinta un attore che registra in una stanza che rimbomba resterebbe rimbombante
    in tutte le tracce, solo in modo uniforme.
    """
    actor = np.median(np.stack(profiles), axis=0)
    return (1 - pull) * actor + pull * SPEECH_TARGET


def match_timbre(
    audio: np.ndarray,
    profile: np.ndarray,
    reference: np.ndarray,
    strength: float = 0.9,
    max_boost_db: float = 6.0,
    max_cut_db: float = 10.0,
) -> tuple[np.ndarray, float]:
    """Equalizza la voce perché suoni come il riferimento. Restituisce anche la correzione massima."""
    diff = (reference - profile) * strength
    diff -= np.median(diff)  # conta la forma, non il volume
    diff = np.clip(diff, -max_cut_db, max_boost_db)
    diff[BAND_CENTERS < 220] = np.minimum(diff[BAND_CENTERS < 220], 0.0)  # mai aggiungere rimbombo
    diff = np.convolve(np.pad(diff, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    biggest = float(np.max(np.abs(diff)))
    if biggest < 0.5:
        return audio, biggest
    nyq = SR / 2
    freq_points = np.concatenate([[0.0], BAND_CENTERS, [nyq]]) / nyq
    gains_db = np.concatenate([[diff[0]], diff, [diff[-1]]])
    taps = signal.firwin2(2047, freq_points, 10 ** (gains_db / 20))
    out = signal.oaconvolve(audio, taps.astype(np.float32), mode="same")
    return out.astype(np.float32), biggest


TONE_PRESETS = {
    "caldo": [pb.LowShelfFilter(cutoff_frequency_hz=250, gain_db=2.5), pb.HighShelfFilter(cutoff_frequency_hz=6000, gain_db=-2.0)],
    "naturale": [],
    "chiaro": [pb.LowShelfFilter(cutoff_frequency_hz=200, gain_db=-1.5), pb.HighShelfFilter(cutoff_frequency_hz=4500, gain_db=2.5)],
}


def tone(audio: np.ndarray, preset: str = "naturale") -> np.ndarray:
    plugins = TONE_PRESETS[preset]
    if not plugins:
        return audio
    return pb.Pedalboard(plugins)(audio, SR).astype(np.float32)


# --- Dinamica e livelli ---------------------------------------------------


def speech_level_db(audio: np.ndarray) -> float:
    mask = an.speech_mask(audio)
    levels = an.frame_rms_db(audio)
    n = min(levels.size, mask.size)
    voiced = levels[:n][mask[:n]]
    return float(np.median(voiced)) if voiced.size else an.rms_db(audio)


def compress(audio: np.ndarray) -> np.ndarray:
    """Porta la voce a un livello di lavoro e ne riduce gli sbalzi."""
    gain = SPEECH_TARGET_DB - speech_level_db(audio)
    staged = (audio * 10 ** (gain / 20)).astype(np.float32)
    board = pb.Pedalboard([pb.Compressor(threshold_db=-22.0, ratio=2.5, attack_ms=8.0, release_ms=120.0)])
    return board(staged, SR).astype(np.float32)


def _limit(audio: np.ndarray, lookahead_s: float = 0.01) -> np.ndarray:
    """Limiter con anticipo: nessun campione supera il tetto, senza colorare il resto."""
    ceiling = 10 ** ((FINAL_PEAK_DB - 0.7) / 20)
    need = np.minimum(1.0, ceiling / np.maximum(np.abs(audio), 1e-9))
    if need.min() >= 1.0:
        return audio
    size = 2 * int(lookahead_s * SR) + 1
    gain = minimum_filter1d(need, size=size, mode="nearest")
    gain = uniform_filter1d(gain, size=size, mode="nearest")
    return (audio * np.minimum(gain, need)).astype(np.float32)


def quiet_pauses(audio: np.ndarray, target_db: float = FINAL_NOISE_DB) -> np.ndarray:
    """Abbassa il fondo nelle pause se supera il limite."""
    floor = an.noise_floor_db(audio)
    if floor <= target_db + 2:
        return audio
    mask = an.mask_to_samples(an.speech_mask(audio), audio.size)
    pause_gain = 10 ** ((target_db - floor) / 20)
    env = np.where(mask, 1.0, pause_gain).astype(np.float32)
    return (audio * _fade_envelope(env, 0.04)).astype(np.float32)


def room_fill(size: int, level_db: float = ROOM_FILL_DB) -> np.ndarray:
    """Un fondo di stanza sintetico, morbido e costante (rumore rosa-ish, sempre uguale)."""
    rng = np.random.default_rng(12345)
    noise = rng.standard_normal(size)
    noise = signal.sosfilt(signal.butter(1, 2500, "lowpass", fs=SR, output="sos"), noise)
    noise = signal.sosfilt(signal.butter(2, 80, "highpass", fs=SR, output="sos"), noise)
    return (noise * 10 ** (level_db / 20) / np.sqrt(np.mean(noise**2))).astype(np.float32)


def finalize(
    bodies: list[np.ndarray], tone_sample: np.ndarray, gap_s: float = 1.5, fill_silence: bool = False
) -> np.ndarray:
    """Monta i pezzi con silenzio in testa, tra le tracce e in coda, poi porta tutto alle specifiche."""
    parts = [room_tone(tone_sample, HEAD_S)]
    for i, body in enumerate(bodies):
        if i:
            parts.append(room_tone(tone_sample, gap_s))
        parts.append(body)
    parts.append(room_tone(tone_sample, TAIL_S))
    out = np.concatenate(parts).astype(np.float32)

    for _ in range(3):
        out = (out * 10 ** ((FINAL_RMS_DB - an.rms_db(out)) / 20)).astype(np.float32)
        out = _limit(out)
        if abs(an.rms_db(out) - FINAL_RMS_DB) < 0.3:
            break
    out = quiet_pauses(out)
    if fill_silence:
        # Audible scarta i file con silenzi a zero assoluto: serve un fondo di stanza continuo.
        out = (out + room_fill(out.size)).astype(np.float32)
    tp = an.true_peak_db(out)
    if tp > FINAL_PEAK_DB:
        out = (out * 10 ** ((FINAL_PEAK_DB - tp) / 20)).astype(np.float32)
    return out
