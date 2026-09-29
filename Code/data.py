"""Bearing vibration data.

* CWRU (fault diagnosis): MATLAB files with a `*_DE_time` drive-end signal, 12 kHz.
  Expected layout:  <root>/<Class>/*.mat   with Class in {Normal, Ball, InnerRace, OuterRace}
* XJTU-SY (remaining useful life): one folder per bearing containing 1.csv ... N.csv
  (one minute-spaced 1.28 s snapshot each, columns Horizontal/Vertical vibration).
* `synthetic_*` generators produce toy signals only for smoke-testing the pipeline.
"""
from pathlib import Path

import numpy as np

CWRU_CLASSES = ("Normal", "Ball", "InnerRace", "OuterRace")


def segment(signal, window=2048, stride=1024):
    n = (len(signal) - window) // stride + 1
    return np.stack([signal[i * stride: i * stride + window] for i in range(max(n, 0))])


def load_cwru(root, window=2048, stride=1024, classes=CWRU_CLASSES):
    from scipy.io import loadmat

    X, y = [], []
    for label, cls in enumerate(classes):
        for f in sorted((Path(root) / cls).glob("*.mat")):
            mat = loadmat(f)
            key = next(k for k in mat if k.endswith("_DE_time"))
            seg = segment(mat[key].ravel(), window, stride)
            X.append(seg)
            y.append(np.full(len(seg), label))
    if not X:
        raise FileNotFoundError(f"no .mat files under {root}/<{'|'.join(classes)}>/")
    return np.concatenate(X), np.concatenate(y), list(classes)


def load_xjtu_bearing(folder, channel="Horizontal_vibration_signals", failure_threshold=None):
    """Returns (snapshots, rul_fraction) for one run-to-failure bearing.

    RUL is expressed as the fraction of life remaining (1 at start, 0 at failure).
    """
    import pandas as pd

    files = sorted(Path(folder).glob("*.csv"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"no N.csv snapshots in {folder}")
    X = np.stack([pd.read_csv(f)[channel].to_numpy() for f in files])
    rul = np.linspace(1.0, 0.0, len(X))
    return X, rul


# --------------------------------------------------------------------------- #
# Synthetic signals (smoke tests only; not a substitute for real data)
# --------------------------------------------------------------------------- #
def synthetic_cwru(n_per_class=60, window=2048, fs=12000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(window) / fs
    fault_freq = {1: 141.0, 2: 162.0, 3: 107.0}  # ball, inner, outer (Hz), typical CWRU values
    X, y = [], []
    for label in range(4):
        for _ in range(n_per_class):
            s = 0.1 * np.sin(2 * np.pi * 29.95 * t) + 0.05 * rng.standard_normal(window)
            if label:
                period = int(fs / fault_freq[label])
                impulses = np.zeros(window)
                impulses[rng.integers(0, period)::period] = rng.uniform(0.5, 1.5)
                ring = np.exp(-t[:60] * 3000) * np.sin(2 * np.pi * 3000 * t[:60])
                s += np.convolve(impulses, ring, mode="same") * (0.5 + 0.5 * label)
            X.append(s)
            y.append(label)
    return np.array(X), np.array(y), list(CWRU_CLASSES)


def synthetic_run_to_failure(n_snapshots=120, window=2048, seed=0):
    rng = np.random.default_rng(seed)
    life = np.linspace(0, 1, n_snapshots)
    health = np.where(life < 0.6, 0.0, ((life - 0.6) / 0.4) ** 2)
    X = np.stack([rng.standard_normal(window) * (0.1 + 2 * h) + h * np.sin(np.arange(window) * 0.3) for h in health])
    return X, 1 - life
