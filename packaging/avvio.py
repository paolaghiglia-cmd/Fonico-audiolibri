"""Punto di partenza del programma installato (usato da PyInstaller)."""

import sys

if "--verifica" in sys.argv:
    # Senza console un errore aprirebbe una finestra che nessuno chiude: lo scriviamo su file.
    import tempfile
    import traceback
    from pathlib import Path

    try:
        from fonico.selfcheck import run

        code = run()
    except BaseException:
        report = Path(tempfile.gettempdir()) / "fonico_verifica.txt"
        report.write_text(traceback.format_exc())
        code = 2
    sys.exit(code)

from fonico.gui.app import main

sys.exit(main())
