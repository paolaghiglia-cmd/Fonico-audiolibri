"""Il riepilogo a semafori, scritto per chi non è un tecnico del suono."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import analysis as an

VERDE, GIALLO, ROSSO = "verde", "giallo", "rosso"
_ORDER = {VERDE: 0, GIALLO: 1, ROSSO: 2}


def _clock(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


@dataclass
class Verdict:
    color: str = VERDE
    messages: list[str] = field(default_factory=list)

    def add(self, color: str, message: str) -> None:
        if _ORDER[color] > _ORDER[self.color]:
            self.color = color
        self.messages.append(message)


@dataclass
class TrackInfo:
    before: an.Measures
    after: an.Measures | None = None
    hum: float | None = None
    clicks: int = 0
    breaths: int = 0
    pauses: int = 0
    timbre_db: float = 0.0
    speech_db: float = -20.0


def evaluate(info: TrackInfo) -> Verdict:
    v = Verdict()
    b = info.before

    for start, end in b.clipping[:5]:
        v.add(ROSSO, f"La voce è distorta tra {_clock(start)} e {_clock(end + 0.5)}: conviene riregistrare quella frase.")
    if len(b.clipping) > 5:
        v.add(ROSSO, f"Ci sono altri {len(b.clipping) - 5} punti distorti: registra più lontano dal microfono o abbassa il volume di registrazione.")

    if info.speech_db < -45:
        v.add(ROSSO, "La registrazione è molto bassa: alzando la voce si alza anche il rumore. Avvicinati al microfono o alza il volume di registrazione.")
    elif info.speech_db < -35:
        v.add(GIALLO, "La registrazione è piuttosto bassa: l'ho alzata io, ma ascolta che il fondo sia pulito.")

    if b.noise_db > -35:
        v.add(ROSSO, "C'è molto rumore di fondo: la pulizia potrebbe rendere la voce artificiale. Meglio registrare in un ambiente più silenzioso.")
    elif b.noise_db > -50:
        v.add(GIALLO, "C'era parecchio rumore di fondo e l'ho tolto: ascolta che la voce suoni naturale.")

    if info.hum:
        v.add(VERDE, f"Ho tolto un ronzio elettrico ({int(info.hum)} Hz): controlla cavi e alimentatori vicino al microfono.")
    if info.clicks:
        v.add(VERDE, f"Ho attenuato {info.clicks} click o schiocchi di bocca.")
    if info.breaths:
        v.add(VERDE, f"Ho sistemato {info.breaths} respiri.")
    if info.pauses:
        v.add(VERDE, f"Ho accorciato {info.pauses} pause troppo lunghe.")
    if info.timbre_db > 4.5:
        v.add(GIALLO, "La voce suonava diversa dalle altre tracce (microfono spostato o giornata diversa): l'ho uniformata, ma ascolta il risultato.")

    a = info.after
    if a is not None and not a.acx_ok:
        v.add(GIALLO, "Il volume finale non rientra perfettamente negli standard Audible: ascolta e, se serve, riregistra.")

    if v.color == VERDE:
        v.messages.insert(0, "Pronta.")
    return v
