"""Numerical regression test for the Sobol estimator used by the GSA."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.stats as st
from safepython import VBSA
from safepython.sampling import AAT_sampling

from GSA.analysis import _jansen_point


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


if __name__ == "__main__":
    unittest.main()
