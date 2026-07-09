"""Shared figure-saving helper: paper-relevant plots are saved as PNG.

Run metadata (epoch, expert count, rel-L2, ...) belongs in the FILENAME, not
in titles/suptitles: figures stay clean and the caption carries the data.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def save_png(path, fig=None, dpi: int = 200) -> Path:
    """Save the figure to ``path`` (.png).

    Args:
        path: Target path; its suffix is normalized to .png.
        fig:  Figure to save (defaults to the current figure).
        dpi:  Raster resolution.

    Returns:
        The PNG path actually written.
    """
    path = Path(path).with_suffix('.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    target = fig if fig is not None else plt
    target.savefig(path, dpi=dpi, bbox_inches='tight')
    return path
