"""Hand-crafted vibration features and scaling to qubit rotation angles."""
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

FEATURE_NAMES = ["mean", "std", "rms", "peak", "peak_to_peak", "skewness", "kurtosis", "crest_factor",
                 "shape_factor", "impulse_factor", "clearance_factor", "spectral_centroid", "spectral_energy_hi"]


def extract_features(X, fs=12000):
    abs_x = np.abs(X)
    rms = np.sqrt((X**2).mean(1))
    peak = abs_x.max(1)
    mean_abs = abs_x.mean(1)
    spec = np.abs(np.fft.rfft(X, axis=1))
    freqs = np.fft.rfftfreq(X.shape[1], 1 / fs)
    centroid = (spec * freqs).sum(1) / (spec.sum(1) + 1e-12)
    hi = (spec[:, freqs > fs / 8] ** 2).sum(1) / ((spec**2).sum(1) + 1e-12)
    feats = np.column_stack([
        X.mean(1), X.std(1), rms, peak, X.max(1) - X.min(1), skew(X, axis=1), kurtosis(X, axis=1),
        peak / (rms + 1e-12), rms / (mean_abs + 1e-12), peak / (mean_abs + 1e-12),
        peak / (np.sqrt(abs_x).mean(1) ** 2 + 1e-12), centroid, hi,
    ])
    return feats


def angle_encoder(n_qubits):
    """Standardise -> PCA to `n_qubits` components -> scale to [0, pi] for angle encoding."""
    return Pipeline([("std", StandardScaler()), ("pca", PCA(n_components=n_qubits)),
                     ("angle", MinMaxScaler(feature_range=(0, np.pi)))])
