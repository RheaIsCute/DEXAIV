"""The DEXAIV window: a webview shell around the pages in ``ui/web``.

Started by ``vox gui`` with the URL of the local control API. This process imports nothing
from ``vox``; everything it shows arrives over that URL, and everything the user does goes
back as a command. With ``--browser`` (or when pywebview is not installed) the same pages
open in the default browser instead.
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser

TITLE = "DEXAIV"
DESIGN = (1200, 780)  # the mockups' frame, in CSS pixels
FRAME = (16, 39)  # window border and title bar around the page
GROUND = "#121217"


def window_size() -> tuple[int, int]:
    """Outer window size whose page area is the design frame.

    pywebview takes logical pixels and applies the display scaling itself (measured at
    150%: a 900x600 request gives an 886x563 CSS-pixel page), so no DPI maths belongs here.
    A window smaller than the frame is handled by the page, which scales itself down.
    """
    return DESIGN[0] + FRAME[0], DESIGN[1] + FRAME[1]


def in_browser(url: str) -> int:
    webbrowser.open(url)
    print("UI opened in the browser; press Ctrl+C here to stop the engine.", flush=True)
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args(argv)
    if args.browser:
        return in_browser(args.url)
    try:
        import webview
    except ImportError:
        print("pywebview is not installed (pip install -r requirements/gui.txt); "
              "using the browser.", file=sys.stderr)
        return in_browser(args.url)
    width, height = window_size()
    webview.create_window(TITLE, args.url, width=width, height=height,
                          min_size=(640, 440), background_color=GROUND)
    webview.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
