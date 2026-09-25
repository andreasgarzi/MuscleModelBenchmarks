"""Numerical regression test for the Sobol estimator used by the GSA."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.stats as st
from safepython import VBSA
from safepython.sampling import AAT_sampling

from GSA.analysis import _effect_profile, _jansen_point


class JansenEstimatorTests(unittest.TestCase):
    """Compare the implementation with a problem having analytical indices."""

    def test_ishigami_indices(self) -> None:
        """Recover published first- and total-order Ishigami sensitivities."""

        # N is intentionally larger than a production screening run because
        # this test checks the estimator itself, not small-sample convergence.
        np.random.seed(20260826)
        n = 8192
        unit_ab = AAT_sampling("lhs", 3, st.uniform, [0.0, 1.0], 2 * n)
        a, b, c = VBSA.vbsa_resampling(unit_ab)

        def scale(x: np.ndarray) -> np.ndarray:
            """Map SAFE's unit samples to the conventional [-pi, pi] domain."""

            return -np.pi + 2 * np.pi * x

        def ishigami(x: np.ndarray) -> np.ndarray:
            """Ishigami function with a=7 and b=0.1."""

            return np.sin(x[:, 0]) + 7 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])

        si, sti, _, _ = _jansen_point(
            ishigami(scale(a)), ishigami(scale(b)), ishigami(scale(c)), 3
        )
        np.testing.assert_allclose(si, [0.3139, 0.4424, 0.0], atol=0.06)
        np.testing.assert_allclose(sti, [0.5576, 0.4424, 0.2437], atol=0.06)

    def test_effect_profiles_use_unit_input_scale(self) -> None:
        """Recover comparable slopes for parameters with different bounds."""

        n = 4
        m = 2
        unit_a = np.array(
            [[0.1, 0.8], [0.2, 0.6], [0.3, 0.4], [0.4, 0.2]], dtype=float
        )
        unit_b = np.array(
            [[0.9, 0.1], [0.8, 0.3], [0.7, 0.5], [0.6, 0.7]], dtype=float
        )
        unit_samples = np.vstack((unit_a, unit_b))

        def output(samples: np.ndarray) -> np.ndarray:
            return np.column_stack(
                (
                    2.0 * samples[:, 0] + 3.0 * samples[:, 1],
                    -4.0 * samples[:, 0] + 0.5 * samples[:, 1],
                )
            )

        c0 = unit_b.copy()
        c0[:, 0] = unit_a[:, 0]
        c1 = unit_b.copy()
        c1[:, 1] = unit_a[:, 1]
        signal = np.vstack((output(unit_a), output(unit_b), output(c0), output(c1)))

        profiles = _effect_profile(signal, unit_samples, n, m)
        np.testing.assert_allclose(profiles, [[2.0, -4.0], [3.0, 0.5]])


if __name__ == "__main__":
    unittest.main()
