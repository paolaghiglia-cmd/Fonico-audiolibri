"""La finestra dell'attore: tracce, ascolto, scelte semplici, esportazione."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..pipeline import Session, TrackResult
from ..project import AUDIO_SUFFIXES, BREATHS, CLEANING, PROJECT_SUFFIX, TONES, Project
from ..report import GIALLO, ROSSO, VERDE

COLORS = {VERDE: "#2e9d4f", GIALLO: "#e0a400", ROSSO: "#d23b3b", None: "#b8b8b8"}
COLOR_WORDS = {VERDE: "Pronta", GIALLO: "Da ascoltare", ROSSO: "Da riregistrare"}
LABELS = {
    "leggera": "Leggera", "normale": "Normale", "forte": "Forte",
    "lasciali": "Lasciali", "attenuali": "Attenuali", "toglili": "Toglili",
    "caldo": "Più calda", "naturale": "Naturale", "chiaro": "Più chiara",
}


def dot(color: str, size: int = 14) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pix)


class TrackList(QListWidget):
    """Lista che accetta i file trascinati da Esplora risorse e si riordina trascinando."""

    files_dropped = Signal(list)
    reordered = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            files = []
            for p in paths:
                path = Path(p)
                if path.is_dir():
                    files += [str(f) for f in path.iterdir() if f.suffix.lower() in AUDIO_SUFFIXES]
                else:
                    files.append(p)
            self.files_dropped.emit(files)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
            self.reordered.emit()


class Choice(QWidget):
    """Tre pulsanti affiancati, uno solo acceso."""

    changed = Signal(str)

    def __init__(self, values: tuple[str, ...], current: str) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QPushButton] = {}
        for v in values:
            b = QPushButton(LABELS[v])
            b.setCheckable(True)
            b.setMinimumWidth(90)
            b.setStyleSheet("QPushButton:checked { background: #2f6fd6; color: white; font-weight: 600; }")
            self.group.addButton(b)
            self.buttons[v] = b
            layout.addWidget(b)
            b.clicked.connect(lambda _=False, v=v: self.changed.emit(v))
        self.set(current)

    def set(self, value: str) -> None:
        self.buttons[value].setChecked(True)


class Worker(QObject):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, job) -> None:
        super().__init__()
        self.job = job

    def run(self) -> None:
        try:
            self.done.emit(self.job(lambda f, m: self.progress.emit(f, m)))
        except Exception as exc:  # l'attore deve vedere un messaggio, non un crash
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project = Project()
        self.session = Session(cache=Path(tempfile.mkdtemp(prefix="fonico_")))
        self.stale = True  # le scelte sono cambiate dopo l'ultima elaborazione
        self.unsaved = False
        self.thread: QThread | None = None

        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)

        self._build_menu()
        self._build_ui()
        self._refresh_title()
        self.resize(1100, 720)

    # --- costruzione ------------------------------------------------------

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        for text, key, slot in [
            ("Nuovo audiolibro", QKeySequence.New, self.new_project),
            ("Apri…", QKeySequence.Open, self.open_project),
            ("Salva", QKeySequence.Save, self.save_project),
            ("Salva con nome…", QKeySequence.SaveAs, self.save_project_as),
        ]:
            action = QAction(text, self)
            action.setShortcut(key)
            action.triggered.connect(slot)
            menu.addAction(action)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)

        book = QGroupBox("Il libro")
        form = QFormLayout(book)
        self.title_edit = QLineEdit()
        self.author_edit = QLineEdit()
        self.reader_edit = QLineEdit()
        form.addRow("Titolo", self.title_edit)
        form.addRow("Autore", self.author_edit)
        form.addRow("Letto da", self.reader_edit)
        for edit in (self.title_edit, self.author_edit, self.reader_edit):
            edit.textEdited.connect(self._book_edited)
        outer.addWidget(book)

        split = QSplitter(Qt.Horizontal)

        left = QGroupBox("Le tue tracce")
        lv = QVBoxLayout(left)
        hint = QLabel("Trascina qui i file della voce, oppure premi «Aggiungi». Trascinale per cambiare l'ordine.")
        hint.setWordWrap(True)
        lv.addWidget(hint)
        self.list = TrackList()
        self.list.files_dropped.connect(self.add_files)
        self.list.reordered.connect(self._sync_order)
        self.list.currentRowChanged.connect(self._show_track)
        lv.addWidget(self.list)
        row = QGridLayout()
        for i, (text, slot) in enumerate([
            ("Aggiungi…", self.pick_files),
            ("Togli", self.remove_tracks),
            ("Unisci in un capitolo", self.merge_chapter),
            ("Separa dal capitolo", self.split_chapter),
        ]):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b, i // 2, i % 2)
        lv.addLayout(row)
        split.addWidget(left)

        right = QGroupBox("Ascolta e controlla")
        rv = QVBoxLayout(right)
        self.track_title = QLabel("Nessuna traccia scelta")
        self.track_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        rv.addWidget(self.track_title)
        self.verdict_label = QLabel("")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setTextFormat(Qt.RichText)
        self.verdict_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        rv.addWidget(self.verdict_label, 1)

        play_row = QHBoxLayout()
        self.play_orig = QPushButton("▶  Originale")
        self.play_fixed = QPushButton("▶  Sistemata")
        self.stop_btn = QPushButton("■  Ferma")
        self.play_orig.clicked.connect(lambda: self.play(original=True))
        self.play_fixed.clicked.connect(lambda: self.play(original=False))
        self.stop_btn.clicked.connect(self.player.stop)
        for b in (self.play_orig, self.play_fixed, self.stop_btn):
            play_row.addWidget(b)
        rv.addLayout(play_row)
        self.position = QSlider(Qt.Horizontal)
        self.position.sliderMoved.connect(self.player.setPosition)
        self.player.durationChanged.connect(lambda d: self.position.setRange(0, d))
        self.player.positionChanged.connect(self._moved)
        rv.addWidget(self.position)
        self.reference_box = QCheckBox("Usa questa voce come modello per tutte le altre")
        self.reference_box.toggled.connect(self._reference_toggled)
        rv.addWidget(self.reference_box)
        split.addWidget(right)
        split.setSizes([480, 620])
        outer.addWidget(split, 1)

        prefs = QGroupBox("Come vuoi la voce")
        pg = QFormLayout(prefs)
        s = self.project.settings
        self.cleaning = Choice(CLEANING, s.pulizia)
        self.breaths = Choice(BREATHS, s.respiri)
        self.tone = Choice(TONES, s.timbro)
        self.cleaning.changed.connect(lambda v: self._setting("pulizia", v))
        self.breaths.changed.connect(lambda v: self._setting("respiri", v))
        self.tone.changed.connect(lambda v: self._setting("timbro", v))
        pg.addRow("Pulizia dai rumori", self.cleaning)
        pg.addRow("Respiri", self.breaths)
        pg.addRow("Voce", self.tone)
        outer.addWidget(prefs)

        actions = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.status = QLabel("Aggiungi le tracce per cominciare.")
        self.fix_btn = QPushButton("Sistema la voce")
        self.export_btn = QPushButton("Esporta gli MP3…")
        for b in (self.fix_btn, self.export_btn):
            b.setMinimumHeight(44)
            b.setStyleSheet("font-size: 15px; font-weight: 600; padding: 0 22px;")
        self.fix_btn.clicked.connect(self.process)
        self.export_btn.clicked.connect(self.export)
        status_box = QVBoxLayout()
        status_box.addWidget(self.status)
        status_box.addWidget(self.progress)
        actions.addLayout(status_box, 1)
        actions.addWidget(self.fix_btn)
        actions.addWidget(self.export_btn)
        outer.addLayout(actions)

        self.setCentralWidget(root)
        self._show_track(-1)

    # --- progetto ---------------------------------------------------------

    def _refresh_title(self) -> None:
        name = Path(self.project.file).stem if self.project.file else "Nuovo audiolibro"
        self.setWindowTitle(f"{name}{' *' if self.unsaved else ''} — Fonico audiolibri")

    def _changed(self, affects_audio: bool = True) -> None:
        self.unsaved = True
        if affects_audio:
            self.stale = True
        self._refresh_title()

    def _book_edited(self) -> None:
        self.project.titolo = self.title_edit.text().strip()
        self.project.autore = self.author_edit.text().strip()
        self.project.lettore = self.reader_edit.text().strip()
        self._changed(affects_audio=False)

    def _setting(self, name: str, value: str) -> None:
        setattr(self.project.settings, name, value)
        self._changed()
        if self.session.results:
            self.status.setText("Hai cambiato le scelte: premi «Sistema la voce» per sentirle.")

    def _load_into_ui(self) -> None:
        p = self.project
        self.title_edit.setText(p.titolo)
        self.author_edit.setText(p.autore)
        self.reader_edit.setText(p.lettore)
        self.cleaning.set(p.settings.pulizia)
        self.breaths.set(p.settings.respiri)
        self.tone.set(p.settings.timbro)
        self._fill_list()

    def _work_dir(self) -> Path:
        if self.project.file:
            f = Path(self.project.file)
            return f.with_name(f"{f.stem}_lavoro")
        return Path(tempfile.mkdtemp(prefix="fonico_"))

    def _confirm_discard(self) -> bool:
        if not self.unsaved or not self.project.tracks:
            return True
        answer = QMessageBox.question(
            self, "Modifiche non salvate", "Vuoi salvare il progetto prima di continuare?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if answer == QMessageBox.Save:
            return self.save_project()
        return answer == QMessageBox.Discard

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self.project = Project()
        self.session = Session(cache=self._work_dir())
        self.unsaved, self.stale = False, True
        self._load_into_ui()
        self._refresh_title()

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Apri audiolibro", "", f"Audiolibro (*{PROJECT_SUFFIX})")
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        try:
            self.project = Project.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Non riesco ad aprirlo", str(exc))
            return
        missing = [t.name for t in self.project.tracks if not Path(t.path).exists()]
        if missing:
            QMessageBox.warning(self, "File spostati", "Non trovo più questi file:\n" + "\n".join(missing))
            self.project.tracks = [t for t in self.project.tracks if Path(t.path).exists()]
        self.session = Session(cache=self._work_dir())
        self.unsaved, self.stale = False, True
        self._load_into_ui()
        self._refresh_title()
        self.status.setText("Progetto aperto: premi «Sistema la voce».")

    def save_project(self) -> bool:
        if not self.project.file:
            return self.save_project_as()
        self.project.save(self.project.file)
        self.unsaved = False
        self._refresh_title()
        return True

    def save_project_as(self) -> bool:
        suggested = (self.project.titolo or "audiolibro") + PROJECT_SUFFIX
        path, _ = QFileDialog.getSaveFileName(self, "Salva audiolibro", suggested, f"Audiolibro (*{PROJECT_SUFFIX})")
        if not path:
            return False
        if not path.endswith(PROJECT_SUFFIX):
            path += PROJECT_SUFFIX
        self.project.save(path)
        self.unsaved = False
        self._refresh_title()
        return True

    def closeEvent(self, event):  # noqa: N802
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()

    # --- tracce -----------------------------------------------------------

    def _fill_list(self) -> None:
        current = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for i, t in enumerate(self.project.tracks, start=1):
            result = self.session.results.get(t.path) if not self.stale else None
            color = result.verdict.color if result else None
            text = f"{i:02d}   {t.name}"
            if t.capitolo is not None:
                text += f"   · Capitolo {t.capitolo}"
            if t.path == self.project.settings.riferimento:
                text += "   · modello"
            item = QListWidgetItem(dot(COLORS[color]), text)
            item.setData(Qt.UserRole, t.path)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.project.tracks:
            self.list.setCurrentRow(min(max(current, 0), len(self.project.tracks) - 1))
        self._show_track(self.list.currentRow())

    def pick_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(AUDIO_SUFFIXES))
        paths, _ = QFileDialog.getOpenFileNames(self, "Scegli le tracce della voce", "", f"Audio ({exts})")
        if paths:
            self.add_files(paths)

    def add_files(self, paths: list[str]) -> None:
        rejected = self.project.add_files(paths)
        if rejected:
            QMessageBox.information(
                self, "File ignorati", "Questi file non sono audio:\n" + "\n".join(Path(p).name for p in rejected)
            )
        self._changed()
        self._fill_list()
        self.status.setText(f"{len(self.project.tracks)} tracce. Premi «Sistema la voce».")

    def _selected(self) -> list[int]:
        return sorted(self.list.row(i) for i in self.list.selectedItems())

    def remove_tracks(self) -> None:
        rows = self._selected()
        if not rows:
            return
        self.project.tracks = [t for i, t in enumerate(self.project.tracks) if i not in rows]
        self.project.renumber_chapters()
        self._changed()
        self._fill_list()

    def merge_chapter(self) -> None:
        rows = self._selected()
        if not rows:
            QMessageBox.information(self, "Unisci in un capitolo", "Seleziona le tracce da unire (Ctrl + clic).")
            return
        self.project.merge_into_chapter(rows)
        self._changed(affects_audio=False)
        self._fill_list()

    def split_chapter(self) -> None:
        self.project.split_chapter(self._selected())
        self._changed(affects_audio=False)
        self._fill_list()

    def _sync_order(self) -> None:
        by_path = {t.path: t for t in self.project.tracks}
        order = [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]
        self.project.tracks = [by_path[p] for p in order]
        self.project.renumber_chapters()
        self._changed(affects_audio=False)
        self._fill_list()

    def _current_track(self):
        row = self.list.currentRow()
        return self.project.tracks[row] if 0 <= row < len(self.project.tracks) else None

    def _show_track(self, row: int) -> None:
        track = self._current_track()
        enabled = track is not None
        for w in (self.play_orig, self.stop_btn, self.reference_box):
            w.setEnabled(enabled)
        if not track:
            self.track_title.setText("Nessuna traccia scelta")
            self.verdict_label.setText("")
            self.play_fixed.setEnabled(False)
            return
        self.track_title.setText(track.name)
        self.reference_box.blockSignals(True)
        self.reference_box.setChecked(self.project.settings.riferimento == track.path)
        self.reference_box.blockSignals(False)
        result: TrackResult | None = self.session.results.get(track.path)
        self.play_fixed.setEnabled(result is not None)
        if result is None:
            self.verdict_label.setText("Non ancora sistemata.")
            return
        v = result.verdict
        note = "<p><i>Le scelte sono cambiate: premi «Sistema la voce» per aggiornare.</i></p>" if self.stale else ""
        items = "".join(f"<li>{m}</li>" for m in v.messages if m != "Pronta.")
        self.verdict_label.setText(
            f"<p><span style='color:{COLORS[v.color]}; font-size:18px'>●</span> "
            f"<b>{COLOR_WORDS[v.color]}</b></p>{note}<ul>{items}</ul>"
        )

    def _reference_toggled(self, checked: bool) -> None:
        track = self._current_track()
        if not track:
            return
        self.project.settings.riferimento = track.path if checked else None
        self._changed()
        self._fill_list()

    # --- ascolto ----------------------------------------------------------

    def play(self, original: bool) -> None:
        track = self._current_track()
        if not track:
            return
        if original:
            source = Path(track.path)
        else:
            result = self.session.results.get(track.path)
            if not result:
                return
            source = result.preview
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(source)))
        self.player.play()

    def _moved(self, pos: int) -> None:
        if not self.position.isSliderDown():
            self.position.setValue(pos)

    # --- elaborazione -----------------------------------------------------

    def _busy(self, busy: bool) -> None:
        for w in (self.fix_btn, self.export_btn, self.list, self.cleaning, self.breaths, self.tone):
            w.setEnabled(not busy)

    def _start(self, job, on_done) -> None:
        self.player.stop()
        self.player.setSource(QUrl())
        self._busy(True)
        self.thread = QThread(self)
        worker = Worker(job)
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)
        worker.progress.connect(self._progress)
        self._on_done = on_done
        worker.done.connect(self._finished)
        worker.failed.connect(self._failed)
        for sig in (worker.done, worker.failed):
            sig.connect(self.thread.quit)
        self.thread.finished.connect(lambda: self._busy(False))
        self.thread.finished.connect(worker.deleteLater)
        self._worker = worker
        self.thread.start()

    def _finished(self, value) -> None:
        # Metodo della finestra: Qt lo esegue nel thread dell'interfaccia, non in quello del lavoro.
        self._on_done(value)

    def _progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(int(fraction * 1000))
        self.status.setText(message)

    def _failed(self, message: str) -> None:
        self.status.setText("Qualcosa non è andato.")
        QMessageBox.warning(self, "Qualcosa non è andato", message)

    def process(self, then=None) -> None:
        if not self.project.tracks:
            QMessageBox.information(self, "Nessuna traccia", "Aggiungi prima le tracce della voce.")
            return
        project = self.project

        def done(results) -> None:
            self.stale = False
            self._fill_list()
            counts = {c: sum(r.verdict.color == c for r in results.values()) for c in (VERDE, GIALLO, ROSSO)}
            self.status.setText(
                f"Fatto: {counts[VERDE]} pronte, {counts[GIALLO]} da ascoltare, {counts[ROSSO]} da riregistrare."
            )
            if callable(then):
                then()

        self._start(lambda progress: self.session.run(project, progress), done)

    def export(self) -> None:
        if not self.project.tracks:
            QMessageBox.information(self, "Nessuna traccia", "Aggiungi prima le tracce della voce.")
            return
        if self.stale or not self.session.results:
            self.process(then=self.export)
            return
        red = [r.track.name for r in self.session.results.values() if r.verdict.color == ROSSO]
        if red:
            answer = QMessageBox.question(
                self, "Ci sono tracce da riregistrare",
                "Queste tracce hanno problemi che non posso correggere del tutto:\n"
                + "\n".join(red) + "\n\nVuoi esportare comunque?",
            )
            if answer != QMessageBox.Yes:
                return
        folder = QFileDialog.getExistingDirectory(self, "Dove salvo gli MP3?")
        if not folder:
            return
        project = self.project

        def done(written) -> None:
            self.status.setText(f"Esportati {len(written)} file MP3 in {folder}")
            QMessageBox.information(self, "Fatto", f"Ho salvato {len(written)} file MP3 nella cartella:\n{folder}")

        self._start(lambda progress: self.session.export(project, folder, progress), done)
