import sys

if len(sys.argv) > 1:
    from booktrans.cli import main
else:
    from booktrans.gui import main

sys.exit(main())
