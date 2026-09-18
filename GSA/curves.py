"""Correlated uncertainty model for the length-dependent calcium functions.

The model uses two fitted functions to couple muscle length to calcium
dynamics.  Sampling their polynomial or piecewise coefficients independently
would generate implausible curve shapes and would ignore the covariance
introduced by fitting the same experimental points.  This module instead
bootstraps the digitised data, refits each complete curve and retains the two
leading principal-component directions.  A GSA input is therefore a score
along a joint curve mode, not an isolated coefficient perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class CurveBasis:
    """Serializable PCA representation of the bootstrap curve fits.

    The centres are the coefficients used by the nominal model. ``vectors``
    contains the PCA directions, while ``scales`` contains one standard
    deviation along each direction.  Keeping the bootstrap settings alongside
    the basis makes a resumed run reproducible.
    """

    f1_center: np.ndarray
    f1_vectors: np.ndarray
    f1_scales: np.ndarray
    f1_explained: np.ndarray
    f2_center: np.ndarray
    f2_vectors: np.ndarray
    f2_scales: np.ndarray
    f2_explained: np.ndarray
    n_bootstrap: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        """Convert NumPy arrays to JSON-compatible lists for run metadata."""

        return {
            "f1_center": self.f1_center.tolist(),
            "f1_vectors": self.f1_vectors.tolist(),
            "f1_scales": self.f1_scales.tolist(),
            "f1_explained": self.f1_explained.tolist(),
            "f2_center": self.f2_center.tolist(),
            "f2_vectors": self.f2_vectors.tolist(),
            "f2_scales": self.f2_scales.tolist(),
            "f2_explained": self.f2_explained.tolist(),
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurveBasis":
        """Reconstruct a basis stored in a run's ``metadata.json`` file."""

        return cls(
            f1_center=np.asarray(data["f1_center"], dtype=float),
            f1_vectors=np.asarray(data["f1_vectors"], dtype=float),
            f1_scales=np.asarray(data["f1_scales"], dtype=float),
            f1_explained=np.asarray(data["f1_explained"], dtype=float),
            f2_center=np.asarray(data["f2_center"], dtype=float),
            f2_vectors=np.asarray(data["f2_vectors"], dtype=float),
            f2_scales=np.asarray(data["f2_scales"], dtype=float),
            f2_explained=np.asarray(data["f2_explained"], dtype=float),
            n_bootstrap=int(data["n_bootstrap"]),
            seed=int(data["seed"]),
        )


def _load_xy(path: Path) -> np.ndarray:
    """Load and validate a two-column digitised dataset."""

    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected two columns in {path}.")
    return data


def _f1_values(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Evaluate the constrained piecewise representation of ``f1``.

    ``q`` stores the first breakpoint, two positive interval widths, the
    low-length plateau and the slope of the descending branch.  Expressing
    breakpoints as widths guarantees their ordering during optimisation.
    """

    r1, d12, d23, low, decay = q
    r2 = r1 + d12
    r3 = r2 + d23
    y = np.empty_like(x)
    left = x < r1
    ramp = (x >= r1) & (x < r2)
    plateau = (x >= r2) & (x < r3)
    right = x >= r3
    y[left] = low
    y[ramp] = low + (1.0 - low) * (x[ramp] - r1) / d12
    y[plateau] = 1.0
    y[right] = 1.0 - decay * (x[right] - r3)
    return y


def _fit_f1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit one bootstrap realisation of the piecewise ``f1`` curve."""

    x0 = np.array([1.0, 0.1379, 0.1011, 0.8, 0.4623])
    lower = np.array([0.82, 0.03, 0.03, 0.45, 0.01])
    upper = np.array([1.12, 0.35, 0.40, 0.98, 1.50])
    fit = least_squares(lambda q: _f1_values(x, q) - y, x0=x0, bounds=(lower, upper))
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError("Constrained f1 fit failed.")
    return fit.x


def _f1_to_latent(q: np.ndarray) -> np.ndarray:
    """Move constrained ``f1`` coefficients to an unconstrained PCA space."""

    r1, d12, d23, low, decay = q
    low_fraction = np.clip((low - 0.45) / (0.98 - 0.45), 1e-8, 1 - 1e-8)
    return np.array(
        [r1, np.log(d12), np.log(d23), np.log(low_fraction / (1 - low_fraction)), np.log(decay)]
    )


def _f1_from_latent(q: np.ndarray) -> tuple[float, float, float, float, float]:
    """Convert a latent PCA sample back to ordered, physical coefficients."""

    r1 = float(q[0])
    d12 = float(np.exp(q[1]))
    d23 = float(np.exp(q[2]))
    fraction = 1.0 / (1.0 + np.exp(-q[3]))
    low = float(0.45 + fraction * (0.98 - 0.45))
    decay = float(np.exp(q[4]))
    return r1, r1 + d12, r1 + d12 + d23, low, decay


def _pca(samples: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return leading PCA directions, standard deviations and variance shares."""

    covariance = np.cov(samples, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total = float(np.sum(eigenvalues))
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    return eigenvectors[:, :n_modes].T, np.sqrt(eigenvalues[:n_modes]), explained[:n_modes]


def build_curve_basis(repo_root: Path, n_bootstrap: int = 1000, seed: int = 260826) -> CurveBasis:
    """Estimate covariance of the digitized f1/f2 fits by non-parametric bootstrap.

    PCA modes are centred on the coefficients currently used by ``Params`` so
    that a zero score reproduces the existing model exactly.  Bootstrap is
    used only to estimate plausible, correlated directions of variation.
    """

    data_dir = repo_root / "benchmark_Data" / "Ca_transients"
    blinks = _load_xy(data_dir / "Blinks_l_Capeak.csv")
    konishi_peak = _load_xy(data_dir / "Konishi_l_Capeak.csv")
    konishi_ttp = _load_xy(data_dir / "Konishi_l_Catimetopeak.csv")

    # Convert sarcomere length to the normalised length used by the model and
    # express calcium amplitudes as fractions of their reference maximum.
    blinks = blinks.copy()
    konishi_peak = konishi_peak.copy()
    konishi_ttp = konishi_ttp.copy()
    blinks[:, 0] /= 2.1
    blinks[:, 1] = np.minimum(blinks[:, 1] / 100.0, 1.0)
    konishi_peak[:, 0] /= 2.1
    konishi_peak[:, 1] = np.minimum(konishi_peak[:, 1], 1.0)
    konishi_ttp[:, 0] /= 2.1

    rng = np.random.default_rng(seed)
    f1_samples: list[np.ndarray] = []
    f2_samples: list[np.ndarray] = []

    # Resampling may occasionally leave a poorly conditioned set of points.
    # Failed constrained fits are discarded, with a finite attempt limit so a
    # malformed input dataset cannot leave the analysis in an endless loop.
    attempts = 0
    max_attempts = n_bootstrap * 3
    while len(f1_samples) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        b = blinks[rng.integers(0, len(blinks), len(blinks))]
        k = konishi_peak[rng.integers(0, len(konishi_peak), len(konishi_peak))]
        peak = np.concatenate((b, k), axis=0)
        try:
            f1_samples.append(_f1_to_latent(_fit_f1(peak[:, 0], peak[:, 1])))
        except (RuntimeError, ValueError, FloatingPointError):
            continue

    if len(f1_samples) < n_bootstrap:
        raise RuntimeError(f"Only {len(f1_samples)} valid f1 bootstrap fits were obtained.")

    # f2 is a quadratic fit.  At least three distinct length values are needed
    # after resampling; degenerate draws are skipped.
    for _ in range(n_bootstrap):
        ttp = konishi_ttp[rng.integers(0, len(konishi_ttp), len(konishi_ttp))]
        if np.unique(ttp[:, 0]).size < 3:
            continue
        f2_samples.append(np.polyfit(ttp[:, 0], ttp[:, 1], deg=2))

    if len(f2_samples) < int(0.95 * n_bootstrap):
        raise RuntimeError("Too few valid f2 bootstrap fits.")

    f1_vectors, f1_scales, f1_explained = _pca(np.asarray(f1_samples), 2)
    f2_vectors, f2_scales, f2_explained = _pca(np.asarray(f2_samples), 2)

    # The bootstrap determines covariance and scale, but the centre remains
    # the published model fit.  Consequently, zero mode scores reproduce the
    # baseline model exactly.
    nominal_f1 = _f1_to_latent(np.array([1.0, 0.1379, 0.1011, 0.8, 0.4623]))
    nominal_f2 = np.array([0.3783, -0.8320, 1.1885])

    return CurveBasis(
        f1_center=nominal_f1,
        f1_vectors=f1_vectors,
        f1_scales=f1_scales,
        f1_explained=f1_explained,
        f2_center=nominal_f2,
        f2_vectors=f2_vectors,
        f2_scales=f2_scales,
        f2_explained=f2_explained,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def apply_curve_modes(parameters: dict[str, Any], values: dict[str, float], basis: CurveBasis) -> None:
    """Apply sampled mode scores to one trial's parameter dictionary.

    Missing scores default to zero.  This is useful in the joint blocks, which
    retain only the leading mode of each curve.  The dictionary is modified in
    place because it is already a private copy made for the current model run.
    """

    f1_scores = np.array([values.get("f1_mode_1", 0.0), values.get("f1_mode_2", 0.0)])
    f1_latent = basis.f1_center + (f1_scores * basis.f1_scales) @ basis.f1_vectors
    r1, r2, r3, low, decay = _f1_from_latent(f1_latent)
    slope = (1.0 - low) / (r2 - r1)
    parameters.update(
        {
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "f1_amp_low": low,
            "f1_slope_left": slope,
            "f1_intercept_left": low - slope * r1,
            "f1_amp_plateau": 1.0,
            "f1_decay_slope": decay,
        }
    )

    f2_scores = np.array([values.get("f2_mode_1", 0.0), values.get("f2_mode_2", 0.0)])
    p2 = basis.f2_center + (f2_scores * basis.f2_scales) @ basis.f2_vectors
    l_min = float(parameters.get("f2_l_min", 1.23))
    l_max = float(parameters.get("f2_l_max", 2.04))
    width_min = float(np.polyval(p2, l_min))
    width_max = float(np.polyval(p2, l_max))
    # A non-positive calcium-transient width is non-physical and would also
    # make the ODE ill-defined.  Treat it as a failed model evaluation rather
    # than silently clipping the sampled curve.
    if min(width_min, width_max) <= 0:
        raise ValueError("A sampled f2 curve has non-positive width.")
    parameters.update(
        {
            "p2_1": float(p2[0]),
            "p2_2": float(p2[1]),
            "p2_3": float(p2[2]),
            "f2_width_min": width_min,
            "f2_width_max": width_max,
        }
    )
