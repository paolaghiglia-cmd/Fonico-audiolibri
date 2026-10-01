"""Avvio dell'applicazione."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ..project import PROJECT_SUFFIX
from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Fonico audiolibri")
    app.setOrganizationName("Fonico")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    for arg in sys.argv[1:]:
        if arg.lower().endswith(PROJECT_SUFFIX):
            window.open_path(arg)
            break
    return app.exec()
