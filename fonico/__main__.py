import sys

if "--verifica" in sys.argv:
    from fonico.selfcheck import run

    sys.exit(run())

from fonico.gui.app import main

sys.exit(main())
