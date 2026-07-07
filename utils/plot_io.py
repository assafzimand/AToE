"""Shared figure-saving helper: every paper-relevant plot is saved as both
PNG (for quick viewing) and PDF (vector — what actually goes into the paper).

Run metadata (epoch, expert count, rel-L2, ...) belongs in the FILENAME, not
in titles/suptitles: figures stay clean and the caption carries the data.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def save_png_and_pdf(path, fig=None, dpi: int = 200) -> Path:
    """Save the figure to ``path`` (.png) and alongside it as .pdf.

    Args:
        path: Target path; its suffix is normalized to .png.
        fig:  Figure to save (defaults to the current figure).
        dpi:  Raster resolution for the PNG (PDF is vector).

    Returns:
        The PNG path actually written.
    """
    path = Path(path).with_suffix('.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    target = fig if fig is not None else plt
    target.savefig(path, dpi=dpi, bbox_inches='tight')
    target.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    return path
