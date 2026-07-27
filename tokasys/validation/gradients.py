"""Finite-difference checks for JAX derivatives."""

from __future__ import annotations

import numpy as np


def central_difference_jacobian(fun, x: np.ndarray, step: float = 1.0e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y0 = np.atleast_1d(np.asarray(fun(x), dtype=float))
    jac = np.empty((y0.size, x.size), dtype=float)
    for i in range(x.size):
        delta = np.zeros_like(x)
        delta[i] = step
        yp = np.atleast_1d(np.asarray(fun(x + delta), dtype=float))
        ym = np.atleast_1d(np.asarray(fun(x - delta), dtype=float))
        jac[:, i] = (yp - ym) / (2.0 * step)
    return jac.squeeze(axis=0) if y0.size == 1 else jac
