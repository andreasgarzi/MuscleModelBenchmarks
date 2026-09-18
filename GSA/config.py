"""Definition of the experiments and uncertain inputs used in the GSA.

The sensitivity analysis is deliberately organised in mechanistic blocks.  A
block contains a small group of benchmark trials and only the parameters that
can affect the mechanism under study.  Parameters not listed in a block keep
the values assigned by :mod:`benchmark_trials`; the benchmark definitions and
the muscle model are therefore never modified by the GSA code.

Ranges are kept here, rather than in the runner, so that the numerical study
can be audited without following the execution logic.  ``rationale`` is also
written to each run's metadata file and records whether a range came from the
original calibration bounds, a physiological interval, or an uncertainty
allowance around a literature value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    """Description of one statistically independent GSA input.

    ``scale`` is either ``linear`` or ``log``.  Sampling is performed by SAFE
    in the unit hypercube and transformed component-wise afterwards.  This
    lets us use log-uniform inputs even though SAFEpython 0.2 does not expose a
    log-uniform distribution in ``AAT_sampling``.  The bounds are inclusive
    and expressed in the units used by the model.
    """

    name: str
    low: float
    high: float
    scale: str = "linear"
    units: str = ""
    rationale: str = ""

    def transform(self, u: np.ndarray | float) -> np.ndarray | float:
        """Map samples from ``[0, 1]`` to this parameter's physical range."""

        if self.scale == "linear":
            return self.low + np.asarray(u) * (self.high - self.low)
        if self.scale == "log":
            return np.exp(np.log(self.low) + np.asarray(u) * np.log(self.high / self.low))
        raise ValueError(f"Unsupported parameter scale '{self.scale}'.")


@dataclass(frozen=True)
class GSABlock:
    """A set of trials and parameters evaluated in one Saltelli design.

    ``signal`` selects force or calcium as the analysed model output.  The
    curve flag requests the bootstrap/PCA representation of the two
    length-dependent calcium functions in :mod:`GSA.curves`.
    """

    name: str
    description: str
    trials: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...]
    signal: str = "force"
    includes_curve_uncertainty: bool = False


def U(name: str, low: float, high: float, units: str = "", rationale: str = "") -> ParameterSpec:
    """Convenience constructor for a uniform input."""

    return ParameterSpec(name, low, high, "linear", units, rationale)


def LU(name: str, low: float, high: float, units: str = "", rationale: str = "") -> ParameterSpec:
    """Convenience constructor for a log-uniform, strictly positive input."""

    return ParameterSpec(name, low, high, "log", units, rationale)


# Parameters shared by several blocks are defined once to keep their ranges
# identical across experiments.
MUAP = (
    LU("b1", 1.6e4, 2.4e4, rationale="literature coefficient, +/-20% screening range"),
    LU("b2", 4.0e7, 6.0e7, rationale="literature coefficient, +/-20% screening range"),
    LU("b3", 7.2e7, 1.08e8, rationale="literature coefficient, +/-20% screening range"),
)

SLOW_CA = (
    LU("c1_s", 1.0e3, 1.0e5, rationale="original calibration bounds"),
    LU("c2_s", 1.0e5, 1.0e6, rationale="original calibration bounds"),
    LU("c3_s", 0.1, 2.0, rationale="original calibration bounds"),
)

FAST_CA = (
    LU("c1_f", 1.0e3, 1.0e5, rationale="original calibration bounds"),
    LU("c2_f", 1.0e5, 1.0e6, rationale="original calibration bounds"),
    LU("c3_f", 0.1, 2.0, rationale="original calibration bounds"),
)

CURVE_MODES = (
    U("f1_mode_1", -2.0, 2.0, rationale="joint bootstrap/PCA mode of f1"),
    U("f1_mode_2", -2.0, 2.0, rationale="joint bootstrap/PCA mode of f1"),
    U("f2_mode_1", -2.0, 2.0, rationale="joint bootstrap/PCA mode of f2"),
    U("f2_mode_2", -2.0, 2.0, rationale="joint bootstrap/PCA mode of f2"),
)


# Each dictionary entry is an independently runnable analysis.  The trial
# names are keys from ``benchmark_trials.all_trials``.
BLOCKS: Mapping[str, GSABlock] = {
    "slow_calcium": GSABlock(
        "slow_calcium",
        "Slow calcium-transient calibration experiment.",
        ("slow_23_100Hz_Ca",),
        SLOW_CA + MUAP,
        signal="Ca",
    ),
    "fast_calcium": GSABlock(
        "fast_calcium",
        "Fast calcium-transient calibration experiment.",
        ("fast_35_125Hz_Ca",),
        FAST_CA + MUAP,
        signal="Ca",
    ),
    "slow_ea": GSABlock(
        "slow_ea",
        "Slow whole-muscle excitation-activation: twitch, unfused and fused tetanus.",
        ("cat_SOL_1Hz_8mm", "cat_SOL_10Hz_8mm", "cat_SOL_40Hz_8mm"),
        (
            U("MVC", 25.0, 31.0, "N", "original calibration bounds"),
            LU("Ca_max_s_M", 1.0e5, 1.0e6, rationale="original calibration bounds"),
            U("k1_s_M", 10.0, 20.0, rationale="original calibration bounds"),
            U("k2_s_M", 10.0, 20.0, rationale="original calibration bounds"),
        )
        + SLOW_CA
        + MUAP,
    ),
    "fast_ea": GSABlock(
        "fast_ea",
        "Fast whole-muscle excitation-activation: twitch, unfused and fused tetanus.",
        ("rat_EDL_isof_1Hz", "rat_EDL_isof_30Hz", "rat_EDL_isof_120Hz"),
        (
            U("MVC", 1.9, 2.49, "N", "original calibration bounds"),
            LU("Ca_max_f_M", 1.0e5, 1.0e6, rationale="original calibration bounds"),
            U("k1_f_M", 10.0, 15.0, rationale="original calibration bounds"),
            U("k2_f_M", 10.0, 15.0, rationale="original calibration bounds"),
        )
        + FAST_CA
        + MUAP,
    ),
    "slow_length": GSABlock(
        "slow_length",
        "Slow force-length and calcium length-dependence at 1, 10 and 40 Hz.",
        tuple(
            f"cat_SOL_{freq}Hz_{length}mm"
            for freq in (1, 10, 40)
            for length in (0, 8, 16)
        ),
        (
            U("a", 0.36, 0.54, rationale="FL width, +/-20%"),
            U("shift", 0.12, 0.18, rationale="FL activation shift, +/-20%"),
            U("l_M_opt_scale", 0.90, 1.10, rationale="fascicle-length uncertainty"),
            U("l_M_0_scale", 0.95, 1.05, rationale="initial fascicle-length uncertainty"),
            U("l_T_slack_scale", 0.95, 1.05, rationale="tendon slack-length uncertainty"),
            U("alpha_0_deg", 5.0, 10.0, "deg", "nominal pennation +/-2.5 deg"),
        )
        + CURVE_MODES,
        includes_curve_uncertainty=True,
    ),
    "fast_length": GSABlock(
        "fast_length",
        "Fast force-length at three equidistant length conditions.",
        ("rat_EDL_isol_0.50", "rat_EDL_isol_2.00", "rat_EDL_isol_3.50"),
        (
            U("a", 0.36, 0.54, rationale="FL width, +/-20%"),
            U("shift", 0.12, 0.18, rationale="FL activation shift, +/-20%"),
            U("l_M_opt_scale", 0.90, 1.10, rationale="fascicle-length uncertainty"),
            U("l_M_0_scale", 0.90, 1.10, rationale="initial fascicle-length uncertainty"),
            U("l_T_slack_scale", 0.95, 1.05, rationale="tendon slack-length uncertainty"),
            U("alpha_0_deg", 7.5, 12.5, "deg", "nominal pennation +/-2.5 deg"),
        )
        + CURVE_MODES,
        includes_curve_uncertainty=True,
    ),
    "slow_fv": GSABlock(
        "slow_fv",
        "Slow dynamic contractions with the largest displacements, including yielding.",
        ("rat_SOL_2.00mm", "cat_SOL_10Hz_c_8mm"),
        (
            U("vmax", 4.0, 12.0, "Lopt/s", "slow-fibre physiological range"),
            LU("af_s", 0.1, 1.0, rationale="original calibration bounds"),
            U("fmax", 1.2, 1.8, rationale="eccentric-force literature range"),
            U("fv1", 0.20, 0.30, rationale="literature coefficient +/-20%"),
            U("kMUs", 0.16, 0.24, rationale="literature coefficient +/-20%"),
            U("eps_0_s", 0.045, 0.075, rationale="slow tendon-strain uncertainty"),
            U("kPE", 4.0, 6.0, rationale="passive-element coefficient +/-20%"),
            U("eps0", 0.5, 0.7, rationale="passive-element strain range"),
            U("cy", 0.245, 0.455, rationale="yielding coefficient +/-30%"),
            LU("Vy", 0.07, 0.13, rationale="yielding velocity +/-30%"),
            LU("Ty", 0.14, 0.26, "s", "yielding time constant +/-30%"),
            U("l_M_opt_scale", 0.90, 1.10, rationale="fascicle-length uncertainty"),
            U("l_T_slack_scale", 0.95, 1.05, rationale="tendon slack-length uncertainty"),
            U("alpha_0_delta_deg", -2.5, 2.5, "deg", "pennation uncertainty"),
        ),
    ),
    "fast_fv": GSABlock(
        "fast_fv",
        "Fast maximum shortening and lengthening contractions.",
        ("cat_CF_120Hz_0.95L0_short", "cat_CF_120Hz_0.95L0_length"),
        (
            U("vmax", 8.0, 16.0, "Lopt/s", "fast-fibre physiological range"),
            LU("af_f", 0.1, 3.0, rationale="original calibration bounds"),
            U("fmax", 1.2, 1.8, rationale="eccentric-force literature range"),
            U("fv1", 0.20, 0.30, rationale="literature coefficient +/-20%"),
            U("kMUf", 0.8, 1.2, rationale="literature coefficient +/-20%"),
            U("eps_0_f", 0.030, 0.050, rationale="fast tendon-strain uncertainty"),
            U("kPE", 4.0, 6.0, rationale="passive-element coefficient +/-20%"),
            U("eps0", 0.5, 0.7, rationale="passive-element strain range"),
            U("l_M_opt_scale", 0.90, 1.10, rationale="fascicle-length uncertainty"),
            U("l_M_0_scale", 0.95, 1.05, rationale="initial fascicle-length uncertainty"),
            U("l_T_slack_scale", 0.95, 1.05, rationale="tendon slack-length uncertainty"),
            U("alpha_0_deg", 0.0, 5.0, "deg", "one-sided pennation range at nominal 0 deg"),
        ),
    ),
    "slow_dynamic_joint": GSABlock(
        "slow_dynamic_joint",
        (
            "Second-stage slow dynamic analysis combining the influential "
            "excitation-activation, length, force-velocity and yielding inputs."
        ),
        (
            "cat_SOL_10Hz_c_8mm",
            "cat_SOL_20Hz_c_8mm",
            "cat_SOL_30Hz_c_8mm",
            "cat_SOL_10Hz_v_8mm",
            "cat_SOL_20Hz_v_8mm",
            "cat_SOL_30Hz_v_8mm",
        ),
        (
            LU("Ca_max_s_M", 1.0e5, 1.0e6, rationale="original calibration bounds"),
        )
        + SLOW_CA
        + (
            U("f1_mode_1", -2.0, 2.0, rationale="primary joint bootstrap/PCA mode of f1"),
            U("f2_mode_1", -2.0, 2.0, rationale="primary joint bootstrap/PCA mode of f2"),
            U("a", 0.36, 0.54, rationale="FL width, +/-20%"),
            U("vmax", 4.0, 12.0, "Lopt/s", "slow-fibre physiological range"),
            LU("af_s", 0.1, 1.0, rationale="original calibration bounds"),
            U("fmax", 1.2, 1.8, rationale="eccentric-force literature range"),
            U("kMUs", 0.16, 0.24, rationale="literature coefficient +/-20%"),
            U("cy", 0.245, 0.455, rationale="yielding coefficient +/-30%"),
            LU("Vy", 0.07, 0.13, rationale="yielding velocity +/-30%"),
            LU("Ty", 0.14, 0.26, "s", "yielding time constant +/-30%"),
            U("alpha_0_delta_deg", -2.5, 2.5, "deg", "pennation uncertainty"),
        ),
        includes_curve_uncertainty=True,
    ),
    "fast_dynamic_joint": GSABlock(
        "fast_dynamic_joint",
        (
            "Second-stage fast dynamic analysis contrasting excitation-activation "
            "and mechanics at partial and maximal stimulation."
        ),
        (
            "cat_CF_20Hz_0.95L0_short",
            "cat_CF_20Hz_0.95L0_length",
            "cat_CF_120Hz_0.95L0_short",
            "cat_CF_120Hz_0.95L0_length",
        ),
        (
            LU("Ca_max_f_M", 1.0e5, 1.0e6, rationale="original calibration bounds"),
        )
        + FAST_CA
        + (
            U("f1_mode_1", -2.0, 2.0, rationale="primary joint bootstrap/PCA mode of f1"),
            U("f2_mode_1", -2.0, 2.0, rationale="primary joint bootstrap/PCA mode of f2"),
            U("a", 0.36, 0.54, rationale="FL width, +/-20%"),
            U("vmax", 8.0, 16.0, "Lopt/s", "fast-fibre physiological range"),
            LU("af_f", 0.1, 3.0, rationale="original calibration bounds"),
            U("fmax", 1.2, 1.8, rationale="eccentric-force literature range"),
        ),
        includes_curve_uncertainty=True,
    ),
    "slow_mu": GSABlock(
        "slow_mu",
        "Slow motor unit: twitch, unfused and fused tetanus.",
        ("cat_LG_1Hz", "cat_LG_12.5Hz", "cat_LG_40Hz"),
        (
            U("MVC", 0.06, 0.08, "N", "original calibration bounds"),
            LU("Ca_max_s_MU", 1.0e5, 1.0e6, rationale="original calibration bounds"),
            U("k1_s_MU", 10.0, 20.0, rationale="original calibration bounds"),
            U("k2_s_MU", 10.0, 20.0, rationale="original calibration bounds"),
        )
        + SLOW_CA
        + MUAP,
    ),
    "fast_mu_cat": GSABlock(
        "fast_mu_cat",
        "Cat fast motor unit: twitch, unfused and fused tetanus, including sag.",
        ("cat_MG_1Hz", "cat_MG_20Hz", "cat_MG_40Hz"),
        (
            U("MVC", 0.2, 1.4, "N", "original calibration bounds"),
            LU("Ca_max_f_MU_catMG", 1.0e5, 1.0e6, rationale="original calibration bounds"),
            LU("k1_f_MU_catMG", 10.0, 100.0, rationale="original calibration bounds"),
            LU("k2_f_MU_catMG", 10.0, 100.0, rationale="original calibration bounds"),
            U("As_peak", 1.30, 2.00, rationale="sag amplitude screening range"),
            LU("Ts", 0.076, 0.142, "s", "sag time constant +/-30%"),
            U("tp", 0.098, 0.182, "s", "sag timing +/-30%"),
        )
        + FAST_CA
        + MUAP,
    ),
    "fast_mu_rat": GSABlock(
        "fast_mu_rat",
        "Rat fast motor unit at low, intermediate and saturating frequencies.",
        ("rat_MG_25Hz", "rat_MG_40Hz", "rat_MG_150Hz"),
        (
            U("MVC", 0.06, 0.08, "N", "original calibration bounds"),
            LU("Ca_max_f_MU_ratMG", 1.0e5, 1.0e6, rationale="original calibration bounds"),
            LU("k1_f_MU_ratMG", 10.0, 100.0, rationale="original calibration bounds"),
            LU("k2_f_MU_ratMG", 10.0, 100.0, rationale="original calibration bounds"),
            U("As_peak", 1.30, 2.00, rationale="sag amplitude screening range"),
            LU("Ts", 0.076, 0.142, "s", "sag time constant +/-30%"),
            U("tp", 0.098, 0.182, "s", "sag timing +/-30%"),
        )
        + FAST_CA
        + MUAP,
    ),
}


def get_block(name: str) -> GSABlock:
    """Return a configured block and report the valid names on failure."""

    try:
        return BLOCKS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown GSA block '{name}'. Available: {', '.join(BLOCKS)}") from exc
