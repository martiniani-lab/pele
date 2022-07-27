import numpy as np
import pytest

from ...potentials import Harmonic
from ..rk import BacktrackingLineSearch


def test_align():
    ctol = 1e-4
    max_iter = 10
    dec_scale = 0.5
    bk_ls = BacktrackingLineSearch(ctol, max_iter, dec_scale)
    origin = np.array([0, 0, 0], dtype=float)
    pot = Harmonic(origin, 1, bdim=3)
    start_x = np.array([0, 1, 0], dtype=float)
    overshot_step = np.array([0, -3, 0])
    f_old, grad_old = pot.getEnergyGradient(start_x)
    f_new, grad_new = pot.getEnergyGradient(start_x + overshot_step)

    (f_at_x, grad_at_x, step_scale, step_final) = bk_ls.line_search(
        start_x, f_new, f_old, grad_new, grad_old, overshot_step, pot.getEnergyGradient
    )

    f_at_x_calc, grad_at_x_calc = pot.getEnergyGradient(start_x + step_final)

    assert f_at_x_calc == f_at_x
    assert np.all(grad_at_x_calc == grad_at_x)
    # armijo condition
    assert f_at_x <= f_old + ctol * np.dot(step_final, grad_old)
