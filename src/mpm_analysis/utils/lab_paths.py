"""
lab_paths.py
------------
Hostname-to-Dropbox-root mapping for all lab computers.


Usage in run.py::

    from mpm_analysis.utils.lab_paths import get_lab_path

    ENS_AVG_FOLDER = get_lab_path(
        r"Experiments\\Zeno\\CavityRecAl6\\...\\scan_experiment_results_50"
    )

Adding a new computer
---------------------
1. Find your hostname:  python -c "import socket; print(socket.gethostname())"
2. Add an entry to ``_DROPBOX_ROOTS`` below with the path to your local
   ``Quantum Circuits Lab`` Dropbox folder.
3. Also add the same entry to ``ZenoPlotting/src/zeno_plotting/base.py``
   so both packages stay in sync.
"""

from __future__ import annotations

import socket
from pathlib import Path

# Map hostname → local path to "Quantum Circuits Lab" Dropbox folder
_DROPBOX_ROOTS: dict[str, Path] = {
    # Noam's desktop
    "DESKTOP-BAEVOT2": Path(
        r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab"
    ),
    # Danielle's desktop
    "DESKTOP-FQN6OUI": Path(
        r"C:\Users\dango\Weizmann Institute Dropbox\Danielle Gov\Quantum Circuits Lab"
    ),
    # Barkay's student PC (E: drive)
    "CMD-SRStudent1": Path(
        r"E:\Weizmann Institute Dropbox\Barkay Guttel\Quantum Circuits Lab"
    ),
    # Barkay's new student laptop
    "CS-TM-LT2": Path(
        r"C:\Users\barkayg\Weizmann Institute Dropbox\Barkay Guttel\Quantum Circuits Lab"
    ),
    # Serge's Lenovo PC (E: drive)
    "cmd-SRLenovoPc2": Path(
        r"E:\Weizmann Institute Dropbox\Serge Rosenblum\Quantum Circuits Lab"
    ),
}

# Sub-paths that are the same on every machine
CAVITY_STR = "CavityRecAl6"
CHIP_STR = "Chip_TransSap46_5"


def get_dropbox_root() -> Path:
    """Return the Dropbox root for this machine.

    Raises
    ------
    ValueError
        If the current hostname is not in ``_DROPBOX_ROOTS``.  Add it there
        (and in ``ZenoPlotting/src/zeno_plotting/base.py``) to fix the error.
    """
    hostname = socket.gethostname()
    root = _DROPBOX_ROOTS.get(hostname)
    if root is None:
        raise ValueError(
            f"Hostname '{hostname}' is not in lab_paths._DROPBOX_ROOTS. "
            f"Add it to mpm_analysis/src/mpm_analysis/utils/lab_paths.py "
            f"(and to ZenoPlotting/src/zeno_plotting/base.py). "
            f"Known hosts: {list(_DROPBOX_ROOTS)}. "
            f'To find your hostname: python -c "import socket; print(socket.gethostname())"'
        )
    return root


def get_lab_path(*parts: str) -> Path:
    """Build a path under the Dropbox root for this machine.

    Parameters
    ----------
    *parts:
        Path components relative to the ``Quantum Circuits Lab`` folder.
        Can be passed as one joined string or as separate segments.

    Returns
    -------
    Path
        Absolute path on this machine.  The path may not yet exist if the
        Dropbox hasn't synced it yet.

    Examples
    --------
    >>> from mpm_analysis.utils.lab_paths import get_lab_path
    >>> get_lab_path(r"Experiments\\Zeno", "CavityRecAl6", "scan50")
    WindowsPath('C:/Users/admin/Weizmann .../scan50')
    """
    return get_dropbox_root().joinpath(*parts)
