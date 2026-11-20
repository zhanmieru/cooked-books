"""`python -m cookedbooks.verify`: the command the container runs."""

import sys

from cookedbooks.verify import main

if __name__ == "__main__":
    sys.exit(main())
