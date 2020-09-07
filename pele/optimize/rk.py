"""
This is contains a modified version of Runge-Kutta ode solvers in scipy with
a backtracking line search check. We want a line search to
make sure we converge to a higher tolerance near the minima
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.lib.function_base import gradient, select
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
from scipy.integrate._ivp.common import (validate_max_step,
                                         validate_tol, select_initial_step,
                                         norm, warn_extraneous)
import numpy as np
from scipy.optimize.linesearch import line_search


# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 2.1 # Maximum allowed increase in a step size.


class BacktrackingLineSearch:
    """
    @brief      backtracking line search method

    @details    Backtracking line search method that
                decreases the step size until it satisfies the armijo condition
                f(x+\\alpha step) <= f(x) + ctol*\\grad{f(x)}.step
    """

    def __init__(self, ctol=1e-4, max_iter=10, dec_scale=0.5):
        """
        ctol: double
              tolerance factor for the armijo condition (sufficient decrease)
        max_iter: int
              max number of linesearch iterations for backtracking
        dec_scale: double
              scale with which the step is multiplied to make it smaller
        """
        self.ctol = ctol
        self.max_iter = max_iter
        self.dec_scale = dec_scale

    def line_search(self, x_init, energy, grad, step, get_energy_gradient):
        """ function that performs the line search
        Parameters
        ----------
        x_init: array[double]
              initial x
        fnew: double
              new energy
        f: double
              old energy
        gradnew: array[double]
              new gradient
        grad: array[double]
              old gradient
        step: array[double]
              step direction
        energy_gradient: callable
              function that returns energy and gradient
        Returns
        ----------
        f_at_x function value at the final step
        grad_at_x gradient value at the final step
        step_scale scale with which the step is multiplied
        step*step_scale scaled step to be taken satisfying sufficent decrease
        """
        dg_init = np.dot(grad, step)
        dec = self.ctol * dg_init
        step_scale = 1               # scale of the step in the beginning
        
        if (dg_init > 0):
            raise Exception(
                "the moving direction increases the function value")

        for i in range(self.max_iter):
            x = x_init + step_scale*step
            energy_at_x, grad_at_x = get_energy_gradient(x)
            if energy_at_x > energy + step_scale*dec:
                step_scale *= self.dec_scale
            else:
                return energy_at_x, grad_at_x, step_scale, step_scale*step

        raise Exception(
            "max iterations exceeded, increase max iter for convergence")
        return None


def rk_step(fun, t, y, f, h, A, B, C, E, K):
    """Perform a single Runge-Kutta step.

    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.

    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Current value of the derivative, i.e. ``fun(x, y)``.
    h : float
        Step to use.
    A : list of ndarray, length n_stages - 1
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients above the main diagonal
        are zeros, so `A` is stored as a list of arrays of increasing lengths.
        The first stage is always just `f`, thus no coefficients for it
        are required.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages - 1,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero, thus it is not stored.
    E : ndarray, shape (n_stages + 1,)
        Coefficients for estimating the error of a less accurate method. They
        are computed as the difference between b's in an extended tableau.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.

    Returns
    -------
    step : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.
    error : ndarray, shape (n,)
        Error estimate of a less accurate method.
    step : ndarray, shape (n,)
        step taken. this is important for linesearch issues
    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    K[0] = f
    for s, (a, c) in enumerate(zip(A, C)):
        dy = np.dot(K[:s + 1].T, a) * h
        K[s + 1] = fun(t + c * h, y + dy)
    step = h * np.dot(K[:-1].T, B)
    y_new = y + step
    f_new = fun(t + h, y_new)
    K[-1] = f_new
    error = np.dot(K.T, E)
    return y_new, f_new, error, step


class RungeKutta_LS(OdeSolver):
    """Base class for explicit Runge-Kutta methods with backtracking linesearch
       Note that this support descent directions only
       Note that adding a line search does not make the
       differential equation trajectory more accurate, but instead increases
       the accuracy with which it converges to a minima at the end of the attractor
       """
    C = NotImplemented
    A = NotImplemented
    B = NotImplemented
    E = NotImplemented
    P = NotImplemented
    order = NotImplemented
    n_stages = NotImplemented

    def __init__(self, fun, get_energy_gradient, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6,
                 ls_ctol=1e-4, ls_max_iter=10, ls_dec_scale=0.5,
                 vectorized=False,
                 line_search_class=None, **extraneous):
        warn_extraneous(extraneous)
        super(RungeKutta_LS, self).__init__(fun, t0, y0, t_bound, vectorized,
                                            support_complex=True)
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)
        self.h_abs = select_initial_step(
            self.fun, self.t, self.y, self.f, self.direction,
            self.order, self.rtol, self.atol)
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        self.ls_ctol = ls_ctol
        self.ls_max_iter = ls_max_iter
        self.ls_dec_scale = ls_dec_scale
        self.get_energy_gradient = get_energy_gradient
        self.line_search_class = line_search_class

    def _step_impl(self):
        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.line_search_class == None:
            self.line_search_class = BacktrackingLineSearch(ctol=self.ls_ctol,
                                                       max_iter=self.ls_max_iter,
                                                       dec_scale=self.ls_dec_scale)
        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        order = self.order
        step_accepted = False
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h
            
            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, f_new, error, step = rk_step(self.fun, t, y, self.f, h, self.A,
                                                self.B, self.C, self.E, self.K)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = norm(error / scale)
            
            
            if error_norm == 0.0:
                h_abs *= MAX_FACTOR
                step_accepted = True
            elif error_norm < 1:
                h_abs *= min(MAX_FACTOR,
                             max(1, SAFETY * error_norm ** (-1 / (order + 1))))
                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** (-1 / (order + 1)))
            if step_accepted:
                # check whether the step doesn't increase the energy value
                # This is basically an afterthought, but we can
                # refine the idea.
                # Note that the notation here becomes optimizerish
                # but this takes into account the difference
                energy, gradient = self.get_energy_gradient(y)
                (energy_at_x, grad_at_x,
                 step_scale, new_step) = self.line_search_class.line_search(y, energy,
                                                                       gradient, step,
                                                                       self.get_energy_gradient)
                y_new = y + new_step
                f_new = -grad_at_x




        self.y_old = y
        
        self.t = t_new
        self.y = y_new
        
        self.h_abs = h_abs
        self.f = f_new
        
        return True, None

    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)


class RK23_LS(RungeKutta_LS):
    """Explicit Runge-Kutta method of order 3(2).

    This uses the Bogacki-Shampine pair of formulas [1]_. The error
    is controlled
    assuming accuracy of the second-order method, but steps are taken using the
    third-order accurate formula (local extrapolation is done). A cubic Hermite
    polynomial is used for the dense output.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar and there are two options for ndarray ``y``.
        It can either have shape (n,), then ``fun`` must return array_like with
        shape (n,). Or alternatively it can have shape (n, k), then ``fun``
        must return array_like with shape (n, k), i.e. each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e. the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
        Is always 0 for this solver as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    """
    order = 2
    n_stages = 3
    C = np.array([1/2, 3/4])
    A = [np.array([1/2]),
         np.array([0, 3/4])]
    B = np.array([2/9, 1/3, 4/9])
    E = np.array([5/72, -1/12, -1/9, 1/8])
    P = np.array([[1, -4 / 3, 5 / 9],
                  [0, 1, -2/3],
                  [0, 4/3, -8/9],
                  [0, -1, 1]])


class RK45_LS(RungeKutta_LS):
    """Explicit Runge-Kutta method of order 5(4).

    This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method accuracy, but steps are taken
    using the fifth-order accurate formula (local extrapolation is done).
    A quartic interpolation polynomial is used for the dense output [2]_.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e. each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e. the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
        Is always 0 for this solver as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    """
    order = 4
    n_stages = 6
    C = np.array([1/5, 3/10, 4/5, 8/9, 1])
    A = [np.array([1/5]),
         np.array([3/40, 9/40]),
         np.array([44/45, -56/15, 32/9]),
         np.array([19372/6561, -25360/2187, 64448/6561, -212/729]),
         np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656])]
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                  1/40])
    # Corresponds to the optimum value of c_6 from [2]_.
    P = np.array([
        [1, -8048581381/2820520608, 8663915743/2820520608,
         -12715105075/11282082432],
        [0, 0, 0, 0],
        [0, 131558114200/32700410799, -68118460800/10900136933,
         87487479700/32700410799],
        [0, -1754552775/470086768, 14199869525/1410260304,
         -10690763975/1880347072],
        [0, 127303824393/49829197408, -318862633887/49829197408,
         701980252875 / 199316789632],
        [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
        [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])


class RkDenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, Q):
        super(RkDenseOutput, self).__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q
        self.order = Q.shape[1] - 1
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        if t.ndim == 0:
            p = np.tile(x, self.order + 1)
            p = np.cumprod(p)
        else:
            p = np.tile(x, (self.order + 1, 1))
            p = np.cumprod(p, axis=0)
        y = self.h * np.dot(self.Q, p)
        if y.ndim == 2:
            y += self.y_old[:, None]
        else:
            y += self.y_old
        return y
