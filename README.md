# Fonico audiolibri

Un programma per Windows che fa da fonico a chi registra audiolibri. L'attore registra la voce in brevi tracce (m4a) e il programma la pulisce, la rende uguale in tutte le tracce e produce file MP3 pronti per Audible, Spotify, Storytel o qualunque lettore.

## Per l'attore

1. Installa il programma con `Fonico-Setup.exe` (vedi sotto dove trovarlo).
2. Scrivi titolo, autore e il tuo nome.
3. Trascina nella finestra le tracce della voce. Si mettono in ordine da sole per nome; puoi riordinarle trascinandole.
4. Se vuoi un unico file per capitolo, seleziona le tracce (Ctrl + clic) e premi **Unisci in un capitolo**.
5. Premi **Sistema la voce**. Ogni traccia riceve un semaforo:
   - verde: pronta;
   - giallo: sistemata, ma ascoltala;
   - rosso: c'è un problema che va risolto riregistrando (il programma dice dove e perché).
6. Ascolta **Originale** e **Sistemata** per confrontarle. Se vuoi, cambia pulizia, respiri e colore della voce e premi di nuovo **Sistema la voce**.
7. Premi **Esporta gli MP3** e scegli la cartella.

Puoi salvare il lavoro dal menu File: il progetto (`.fonico`) ricorda tracce, capitoli e scelte.

## Cosa fa il programma su ogni traccia

- Segnala i punti in cui la voce è distorta, troppo bassa o con troppo rumore.
- Toglie ronzii elettrici, rimbombi, click e schiocchi di bocca, rumore di fondo.
- Attenua le "s" taglienti.
- Abbassa o toglie i respiri, a scelta.
- Toglie i silenzi all'inizio e alla fine e accorcia le pause troppo lunghe.
- Rende il timbro della voce uguale in tutte le tracce, prendendo come modello la media delle tracce o quella scelta dall'attore.
- Uniforma il volume e lo porta agli standard Audible/ACX: volume medio tra -23 e -18 dB, picchi sotto -3 dB, fondo sotto -60 dB, 0,75 s di silenzio in testa e 2 s in coda.
- Esporta MP3 192 kbps, 44,1 kHz, mono, con titolo, autore e lettore nei metadati.

Tutto avviene sul computer: nessun file va su internet.

## Dove trovare l'installer

Ogni modifica costruisce automaticamente l'installer per Windows (scheda **Actions** del repository, voce **Fonico-Setup**). Quando si pubblica una versione (tag `v0.1.0`, `v0.2.0`…) l'installer compare anche nella pagina **Releases**.

Il programma non è firmato digitalmente: al primo avvio Windows può mostrare "Windows ha protetto il PC". Premi **Ulteriori informazioni** e poi **Esegui comunque**.

## Per chi sviluppa

```
pip install numpy scipy pedalboard PySide6 pytest
python -m fonico            # avvia il programma
python -m pytest -q         # test del motore audio
```

Serve FFmpeg nel PATH (l'installer lo include già).

- `fonico/processing.py`: i singoli passaggi audio.
- `fonico/pipeline.py`: l'ordine dei passaggi, la cache e l'esportazione.
- `fonico/report.py`: i messaggi a semaforo.
- `fonico/gui/`: la finestra.
- `packaging/` e `.github/workflows/windows.yml`: costruzione dell'installer.

FFmpeg è distribuito con licenza LGPL (https://ffmpeg.org).
