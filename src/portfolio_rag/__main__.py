"""Allow the CLI to be run as ``python -m portfolio_rag``.

Equivalent to the ``portfolio-rag`` console script, and useful when the package
is on the path but its entry points are not installed.
"""

import sys

from portfolio_rag.cli import main

if __name__ == "__main__":
    sys.exit(main())
