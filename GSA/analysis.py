"""Post-process completed Saltelli runs.

The same Jansen estimator is applied to scalar waveform descriptors and to
each saved time point.  The latter produces ``S_i(t)`` and ``S_Ti(t)`` without
rerunning the model.  This module also reports bootstrap confidence intervals,
simple convergence traces and a waveform-based screen for parameter pairs
whose effects may be difficult to distinguish experimentally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from GSA.engine import METRIC_NAMES


def _split_saltelli(values: np.ndarray, n: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split outputs into the A, B and ``C_i`` blocks saved by the runner.

    Row order is ``A`` (N rows), ``B`` (N rows), followed by M consecutive
    ``C_i`` blocks.  Preserving that order is essential for the estimators.
    """

    expected = n * (m + 2)
    if values.shape[0] != expected:
        raise ValueError(f"Expected {expected} model evaluations, found {values.shape[0]}.")
    return values[:n], values[n : 2 * n], values[2 * n :]


def _has_variance(ya: np.ndarray, yb: np.ndarray) -> bool:
    """Reject outputs whose variance is indistinguishable from round-off."""

    y = np.concatenate((np.asarray(ya).ravel(), np.asarray(yb).ravel()))
    y = y[np.isfinite(y)]
    if y.size < 4:
        return False
    scale = max(float(np.max(np.abs(y))), 1.0)
    return float(np.var(y)) > np.finfo(float).eps * scale * scale * 100


def _jansen_point(
    ya: np.ndarray,
    yb: np.ndarray,
    yc: np.ndarray,
    m: int,
) -> tuple[np.ndarray, ...]:
    """Calculate Jansen first- and total-order indices for one output.

    ``yc`` contains the M hybrid-design outputs flattened parameter by
    parameter.  The dummy first-order estimate uses the independent A and B
    matrices to expose finite-sample noise.  Its total-order effect is exactly
    zero by construction because a dummy hybrid matrix is identical to B.
    """

    if not _has_variance(ya, yb):
        empty = np.full(m, np.nan)
        return empty, empty.copy(), np.array(np.nan), np.array(np.nan)

    ya = np.asarray(ya, dtype=float).ravel()
    yb = np.asarray(yb, dtype=float).ravel()
    yc = np.asarray(yc, dtype=float).reshape(m, len(ya))
    base = np.concatenate((ya[np.isfinite(ya)], yb[np.isfinite(yb)]))
    variance = float(np.var(base, ddof=1))
    if not np.isfinite(variance) or variance <= 0:
        empty = np.full(m, np.nan)
        return empty, empty.copy(), np.array(np.nan), np.array(np.nan)

    si = np.full(m, np.nan)
    sti = np.full(m, np.nan)
    for i in range(m):
        mask_s = np.isfinite(ya) & np.isfinite(yc[i])
        mask_st = np.isfinite(yb) & np.isfinite(yc[i])
        if np.count_nonzero(mask_s) >= 4:
            si[i] = 1.0 - np.mean((ya[mask_s] - yc[i, mask_s]) ** 2) / (2.0 * variance)
        if np.count_nonzero(mask_st) >= 4:
            sti[i] = np.mean((yb[mask_st] - yc[i, mask_st]) ** 2) / (2.0 * variance)

    dummy_mask = np.isfinite(ya) & np.isfinite(yb)
    sdummy = (
        1.0 - np.mean((ya[dummy_mask] - yb[dummy_mask]) ** 2) / (2.0 * variance)
        if np.count_nonzero(dummy_mask) >= 4
        else np.nan
    )
    # A truly inactive dummy has C_dummy == B, hence exactly zero total effect.
    return si, sti, np.array(sdummy), np.array(0.0)


def _jansen_bootstrap(
    ya: np.ndarray,
    yb: np.ndarray,
    yc: np.ndarray,
    m: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Paired bootstrap of Saltelli rows for uncertainty intervals.

    A row index is resampled jointly in A, B and every ``C_i`` block.  Sampling
    these arrays independently would destroy the Saltelli pairing and bias the
    resulting confidence intervals.
    """

    rng = np.random.default_rng(seed)
    n = len(ya)
    yc_matrix = np.asarray(yc).reshape(m, n)
    si = np.full((n_bootstrap, m), np.nan)
    sti = np.full((n_bootstrap, m), np.nan)
    sdummy = np.full(n_bootstrap, np.nan)
    stdummy = np.zeros(n_bootstrap)
    for j in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        out = _jansen_point(ya[idx], yb[idx], yc_matrix[:, idx].reshape(-1), m)
        si[j], sti[j] = out[0], out[1]
        sdummy[j] = float(out[2])
    return si, sti, sdummy, stdummy


def _bootstrap_summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bootstrap mean and central 95% interval."""

    values = np.asarray(values, dtype=float)
    return (
        np.nanmean(values, axis=0),
        np.nanquantile(values, 0.025, axis=0),
        np.nanquantile(values, 0.975, axis=0),
    )


def _scalar_indices(
    metrics: np.ndarray,
    n: int,
    m: int,
    n_bootstrap: int,
) -> dict[str, np.ndarray]:
    """Estimate Sobol indices and intervals for every scalar metric."""

    n_metrics = metrics.shape[1]
    si = np.full((n_metrics, m), np.nan)
    sti = np.full((n_metrics, m), np.nan)
    si_lb = np.full_like(si, np.nan)
    si_ub = np.full_like(si, np.nan)
    sti_lb = np.full_like(si, np.nan)
    sti_ub = np.full_like(si, np.nan)
    sdummy = np.full(n_metrics, np.nan)
    stdummy = np.full(n_metrics, np.nan)
    sdummy_ub = np.full(n_metrics, np.nan)
    stdummy_ub = np.full(n_metrics, np.nan)

    for j in range(n_metrics):
        ya, yb, yc = _split_saltelli(metrics[:, j], n, m)
        if not _has_variance(ya, yb):
            continue
        point = _jansen_point(ya, yb, yc, m)
        si[j], sti[j] = point[0], point[1]
        sdummy[j] = float(np.asarray(point[2]).squeeze())
        stdummy[j] = float(np.asarray(point[3]).squeeze())
        if n_bootstrap > 1:
            boot = _jansen_bootstrap(ya, yb, yc, m, n_bootstrap, seed=9401 + j)
            _, si_lb[j], si_ub[j] = _bootstrap_summary(boot[0])
            _, sti_lb[j], sti_ub[j] = _bootstrap_summary(boot[1])
            sdummy_ub[j] = float(np.nanquantile(np.asarray(boot[2]), 0.975))
            stdummy_ub[j] = float(np.nanquantile(np.asarray(boot[3]), 0.975))

    return {
        "Si": si,
        "STi": sti,
        "Si_lb": si_lb,
        "Si_ub": si_ub,
        "STi_lb": sti_lb,
        "STi_ub": sti_ub,
        "Sdummy": sdummy,
        "STdummy": stdummy,
        "Sdummy_ub": sdummy_ub,
        "STdummy_ub": stdummy_ub,
    }


def _time_indices(values: np.ndarray, n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Jansen indices independently at each saved time point."""

    n_time = values.shape[1]
    si = np.full((n_time, m), np.nan)
    sti = np.full((n_time, m), np.nan)
    for j in range(n_time):
        ya, yb, yc = _split_saltelli(values[:, j], n, m)
        if _has_variance(ya, yb):
            out = _jansen_point(ya, yb, yc, m)
            si[j], sti[j] = out[0], out[1]
    return si, sti


def _convergence(metrics: np.ndarray, n: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute scalar indices at four nested base-sample sizes."""

    if n < 8:
        return np.array([], dtype=int), np.empty((0, metrics.shape[1], m)), np.empty((0, metrics.shape[1], m))
    nn = np.unique(np.maximum(4, np.array([n // 4, n // 2, 3 * n // 4, n], dtype=int)))
    si = np.full((len(nn), metrics.shape[1], m), np.nan)
    sti = np.full_like(si, np.nan)
    for j in range(metrics.shape[1]):
        ya, yb, yc = _split_saltelli(metrics[:, j], n, m)
        if not _has_variance(ya, yb):
            continue
        yc_matrix = np.asarray(yc).reshape(m, n)
        for k, sample_size in enumerate(nn):
            out = _jansen_point(
                ya[:sample_size],
                yb[:sample_size],
                yc_matrix[:, :sample_size].reshape(-1),
                m,
            )
            si[k, j, :] = out[0]
            sti[k, j, :] = out[1]
    return nn, si, sti


def _plot_scalar(indices: dict[str, np.ndarray], labels: list[str], trial: str, path: Path) -> None:
    """Plot first- and total-order indices for all scalar descriptors."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    x = np.arange(len(labels))
    for j, (metric, ax) in enumerate(zip(METRIC_NAMES, axes.ravel())):
        s = indices["Si"][j]
        st = indices["STi"][j]
        ax.bar(x - 0.2, s, width=0.4, label="$S_i$", color="#4C78A8")
        ax.bar(x + 0.2, st, width=0.4, label="$S_{Ti}$", color="#F58518")
        if np.isfinite(indices["Sdummy"][j]):
            threshold = max(0.01, abs(float(indices["Sdummy"][j])))
            ax.axhline(threshold, color="0.3", linestyle="--", linewidth=1, label="dummy/MC threshold")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Sobol index")
    axes[1, 0].set_ylabel("Sobol index")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(trial)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_time(sti: np.ndarray, time: np.ndarray, labels: list[str], trial: str, path: Path) -> None:
    """Plot time-resolved total-order indices as a parameter/time heat map."""

    fig, ax = plt.subplots(figsize=(10, max(4, 0.38 * len(labels))))
    shown = np.nan_to_num(sti.T, nan=0.0)
    image = ax.imshow(
        shown,
        aspect="auto",
        origin="lower",
        extent=(float(time[0]), float(time[-1]), -0.5, len(labels) - 0.5),
        vmin=0,
        vmax=max(1.0, float(np.nanpercentile(shown, 99))),
        cmap="viridis",
    )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time [s]")
    ax.set_title(f"{trial}: time-resolved total-order indices (normalised signal)")
    fig.colorbar(image, ax=ax, label="$S_{Ti}(t)$")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _top_parameters(sti: np.ndarray, labels: list[str], count: int = 5) -> list[dict[str, float | str]]:
    """Format the largest finite total-order effects for JSON reporting."""

    scores = np.asarray(sti, dtype=float)
    order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
    return [{"parameter": labels[i], "STi": float(scores[i])} for i in order[:count] if np.isfinite(scores[i])]


def _time_summary(
    sti: np.ndarray,
    time: np.ndarray,
    signal: np.ndarray,
    n: int,
    labels: list[str],
) -> dict[str, Any]:
    """Summarise TVSA only where temporal output variance is informative.

    Sobol indices are undefined before stimulation or whenever all simulations
    give effectively the same output.  The 1% variance mask prevents such
    intervals from dominating trial-level means and reported maxima.
    """

    reference_runs = np.asarray(signal[: 2 * n], dtype=float)
    temporal_variance = np.nanvar(reference_runs, axis=0, ddof=1)
    max_variance = float(np.nanmax(temporal_variance))
    active = np.isfinite(temporal_variance) & (temporal_variance >= 0.01 * max_variance)
    active &= np.any(np.isfinite(sti), axis=1)
    if not np.any(active):
        return {"active_window_s": None, "top_mean_total_effects": []}

    active_indices = np.flatnonzero(active)
    mean_st = np.nanmean(sti[active], axis=0)
    top_order = np.argsort(np.nan_to_num(mean_st, nan=-np.inf))[::-1]
    top: list[dict[str, float | str]] = []
    for parameter_index in top_order[:5]:
        series = sti[:, parameter_index]
        local = active_indices[np.nanargmax(series[active_indices])]
        top.append(
            {
                "parameter": labels[parameter_index],
                "mean_STi": float(mean_st[parameter_index]),
                "max_STi": float(series[local]),
                "time_of_max_s": float(time[local]),
            }
        )
    return {
        "variance_threshold": "1% of maximum temporal output variance",
        "active_window_s": [float(time[active_indices[0]]), float(time[active_indices[-1]])],
        "top_mean_total_effects": top,
    }


def _effect_profile(signal: np.ndarray, samples: np.ndarray, n: int, m: int) -> np.ndarray:
    """Compute a signed temporal fingerprint for each parameter.

    For parameter ``i``, paired B and ``C_i`` simulations differ only in that
    input.  Their waveform difference divided by the physical parameter
    difference is therefore a global finite-difference slope.  The median over
    all N pairs gives a robust signed fingerprint through time.
    """

    ya_x = samples[:n]
    yb_x = samples[n : 2 * n]
    yb = np.asarray(signal[n : 2 * n], dtype=float)
    yc = np.asarray(signal[2 * n :], dtype=float).reshape(m, n, signal.shape[1])
    profiles = np.full((m, signal.shape[1]), np.nan)
    for i in range(m):
        dx = yb_x[:, i] - ya_x[:, i]
        valid = np.isfinite(dx) & (np.abs(dx) > np.finfo(float).eps)
        if np.count_nonzero(valid) < 4:
            continue
        slopes = (yb[valid] - yc[i, valid]) / dx[valid, None]
        profiles[i] = np.nanmedian(slopes, axis=0)
    return profiles


def _identifiability_proxy(profiles: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Flag influential parameters with nearly collinear fingerprints.

    This is a practical confounding screen, not a structural-identifiability
    proof.  Parameters below 1% of the largest profile norm are excluded so
    that correlations between two negligible numerical signals are not
    reported as meaningful.
    """

    norms = np.linalg.norm(np.nan_to_num(profiles, nan=0.0), axis=1)
    influential = norms >= 0.01 * np.max(norms) if np.max(norms) > 0 else np.zeros(len(norms), dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(np.nan_to_num(profiles, nan=0.0))
    pairs: list[dict[str, float | str]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            value = float(correlation[i, j])
            if influential[i] and influential[j] and np.isfinite(value) and abs(value) >= 0.95:
                pairs.append(
                    {"parameter_1": labels[i], "parameter_2": labels[j], "correlation": value}
                )
    pairs.sort(key=lambda item: abs(float(item["correlation"])), reverse=True)
    return {
        "description": (
            "Screening proxy based on correlations between median global finite-difference "
            "waveform fingerprints; high correlation suggests possible practical confounding, "
            "not formal proof of non-identifiability."
        ),
        "potentially_confounded_pairs": pairs,
        "correlation": correlation,
        "profile_norm": norms,
    }


def analyse_run(run_dir: Path, n_bootstrap: int = 500, make_plots: bool = True) -> dict[str, Any]:
    """Analyse all trials in a completed run and write arrays and reports.

    Parameters
    ----------
    run_dir:
        Directory created by :func:`GSA.run_gsa.run_block`.
    n_bootstrap:
        Number of paired Saltelli-row resamples used for scalar confidence
        intervals.  A value of one skips interval estimation.
    make_plots:
        If false, numerical indices and text reports are still written.
    """

    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["analysis_estimator"] = "Jansen first-order and total-order Sobol estimators"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    n = int(metadata["N"])
    labels = list(metadata["parameter_names"])
    m = len(labels)
    report: dict[str, Any] = {
        "block": metadata["block"],
        "N": n,
        "model_evaluations": n * (m + 2),
        "parameter_names": labels,
        "estimator": "Jansen first-order and total-order Sobol estimators on the SAFE Saltelli design",
        "trials": {},
        "interpretation": "pilot" if n < 256 else "pending convergence assessment",
    }

    indices_root = run_dir / "indices"
    figures_root = run_dir / "figures"
    indices_root.mkdir(exist_ok=True)
    if make_plots:
        figures_root.mkdir(exist_ok=True)
    physical_samples = np.load(run_dir / "samples.npy", mmap_mode="r")
    effect_profiles: list[np.ndarray] = []

    for trial in metadata["trials"]:
        trial_dir = run_dir / "trials" / trial
        metrics = np.load(trial_dir / "metrics.npy", mmap_mode="r")
        scalar = _scalar_indices(metrics, n, m, n_bootstrap)
        nn, conv_si, conv_sti = _convergence(metrics, n, m)
        signal = np.load(trial_dir / "signal.npy", mmap_mode="r")
        signal_norm = np.load(trial_dir / "signal_normalized.npy", mmap_mode="r")
        time = np.load(trial_dir / "time.npy")
        time_si, time_sti = _time_indices(signal, n, m)
        time_si_norm, time_sti_norm = _time_indices(signal_norm, n, m)

        out_dir = indices_root / trial
        out_dir.mkdir(exist_ok=True)
        for key, value in scalar.items():
            np.save(out_dir / f"scalar_{key}.npy", value)
        np.save(out_dir / "convergence_N.npy", nn)
        np.save(out_dir / "convergence_Si.npy", conv_si)
        np.save(out_dir / "convergence_STi.npy", conv_sti)
        np.save(out_dir / "time_Si.npy", time_si)
        np.save(out_dir / "time_STi.npy", time_sti)
        np.save(out_dir / "time_normalized_Si.npy", time_si_norm)
        np.save(out_dir / "time_normalized_STi.npy", time_sti_norm)

        trial_report: dict[str, Any] = {"top_total_effects": {}}
        for metric_index, metric in enumerate(METRIC_NAMES):
            trial_report["top_total_effects"][metric] = _top_parameters(
                scalar["STi"][metric_index], labels
            )
        if len(nn) >= 2:
            delta = np.abs(conv_sti[-1] - conv_sti[-2])
            trial_report["max_last_convergence_change"] = {
                metric: float(np.nanmax(delta[j])) if np.any(np.isfinite(delta[j])) else np.nan
                for j, metric in enumerate(METRIC_NAMES)
            }
        trial_report["time_resolved_normalized"] = _time_summary(
            time_sti_norm, time, signal_norm, n, labels
        )
        # Concatenating trial fingerprints later makes the confounding screen
        # require two parameters to look alike across the entire block, rather
        # than during only one contraction.
        effect_profiles.append(_effect_profile(signal_norm, physical_samples, n, m))
        report["trials"][trial] = trial_report

        if make_plots:
            _plot_scalar(scalar, labels, trial, figures_root / f"{trial}_scalar.png")
            _plot_time(time_sti_norm, time, labels, trial, figures_root / f"{trial}_time_STi.png")

    profiles = np.concatenate(effect_profiles, axis=1)
    identifiability = _identifiability_proxy(profiles, labels)
    np.save(indices_root / "identifiability_effect_profiles.npy", profiles)
    np.save(indices_root / "identifiability_profile_correlation.npy", identifiability.pop("correlation"))
    np.save(indices_root / "identifiability_profile_norm.npy", identifiability.pop("profile_norm"))
    report["identifiability_screen"] = identifiability

    # The final two nested estimates provide a deliberately simple convergence
    # flag.  Numerical arrays are retained so users can apply stricter criteria.
    mae_changes = [
        trial.get("max_last_convergence_change", {}).get("mAE_pct_ref", np.nan)
        for trial in report["trials"].values()
    ]
    finite_changes = np.asarray([x for x in mae_changes if np.isfinite(x)], dtype=float)
    if n < 256:
        report["interpretation"] = "pilot"
    elif finite_changes.size and np.max(finite_changes) <= 0.05:
        report["interpretation"] = "stable at the current convergence resolution"
    else:
        report["interpretation"] = f"requires extension (recommended N >= {2 * n})"

    (run_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    _write_markdown_report(run_dir / "report.md", report)
    return report


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    """Write the main numerical findings in a compact, human-readable form."""

    lines = [
        f"# GSA report: {report['block']}",
        "",
        f"Base sample size: **N = {report['N']}** ({report['model_evaluations']} evaluations per trial).",
        f"Interpretation level: **{report['interpretation']}**.",
        f"Estimator: {report['estimator']}.",
        "",
    ]
    if report["interpretation"] == "pilot":
        lines.extend(
            [
                "> This run verifies the workflow and provides preliminary rankings. "
                "Use N >= 256 and inspect convergence/bootstrap intervals before drawing manuscript conclusions.",
                "",
            ]
        )
    elif report["interpretation"].startswith("requires extension"):
        lines.extend(
            [
                "> Rankings are preliminary because the last convergence step changed at least one "
                "mAE total-order index by more than 0.05. Extend the base sample before manuscript use.",
                "",
            ]
        )
    for trial, data in report["trials"].items():
        lines.extend([f"## {trial}", ""])
        for metric in ("mAE_pct_ref", "peak", "time_to_peak_s", "impulse"):
            top = data["top_total_effects"][metric]
            formatted = ", ".join(f"{item['parameter']} ({item['STi']:.3f})" for item in top[:3])
            lines.append(f"- {metric}: {formatted or 'undefined (zero/insufficient variance)'}")
        if "max_last_convergence_change" in data:
            change = data["max_last_convergence_change"].get("mAE_pct_ref", np.nan)
            lines.append(f"- Maximum change in mAE total-order indices over the last convergence step: {change:.3f}")
        tvsa = data.get("time_resolved_normalized", {})
        if tvsa.get("active_window_s"):
            window = tvsa["active_window_s"]
            lines.append(f"- TVSA variance-active window: {window[0]:.4f}-{window[1]:.4f} s")
            for item in tvsa["top_mean_total_effects"][:3]:
                lines.append(
                    f"  - {item['parameter']}: mean STi={item['mean_STi']:.3f}, "
                    f"maximum at t={item['time_of_max_s']:.4f} s"
                )
        lines.append("")
    lines.extend(["## Identifiability screening", "", report["identifiability_screen"]["description"], ""])
    pairs = report["identifiability_screen"]["potentially_confounded_pairs"]
    if pairs:
        for pair in pairs[:10]:
            lines.append(
                f"- {pair['parameter_1']} / {pair['parameter_2']}: "
                f"profile correlation = {pair['correlation']:.3f}"
            )
    else:
        lines.append("- No influential parameter pair exceeded |correlation| = 0.95.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
