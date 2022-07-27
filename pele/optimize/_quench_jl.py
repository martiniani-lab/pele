#!/home/praharsh/anaconda3/envs/changebranch3 python
"""
This file manages quenches from the DifferentialEquations package in julia.
this is written as a separate package since you might need to use python-jl
"""
import julia
import numpy as np
from diffeqpy import de

from pele.optimize import Result

j = julia.Julia()


# def get_negative_grad(x, p, t):
#     return - 2*x

solvers = [
    de.Rosenbrock23(autodiff=False),
    de.Rosenbrock32(autodiff=False),
    de.CVODE_BDF(),
]
solvers_explicit = [de.CVODE_Adams()]
solvers_stiffness = [de.AutoTsit5(de.Rosenbrock23(autodiff=False))]


def ode_julia_naive(
    coords,
    pot,
    tol=1e-4,
    nsteps=2000,
    convergence_check=None,
    solver_type=de.ROCK2(),
    reltol=1e-5,
    abstol=1e-5,
    **kwargs
):
    class feval_pot:
        """wrapper class that makes sure the function is right"""

        def __init__(self):
            self.nfev = 0

        def get_negative_grad(self, x, p, t):
            self.nfev += 1
            return -pot.getEnergyGradient(x.copy())[1]

        def get_energy_gradient(self, x):
            self.nfev += 1
            return pot.getEnergyGradient(x.copy())

    function_evaluate_pot = feval_pot()
    converged = False
    n = 0
    if convergence_check == None:
        convergence_check = lambda g: np.max(g) < tol

    # initialize ode problem
    tspan = (0.0, 10.0)
    # tspan
    # tspan = (0.0, 10.0)
    prob = de.ODEProblem(function_evaluate_pot.get_negative_grad, coords, tspan)
    integrator = de.init(prob, solver_type, reltol=reltol, abstol=abstol)
    x_ = np.full(len(coords), np.nan)
    while not converged and n < nsteps:
        xold = x_
        de.step_b(integrator)
        x_ = integrator.u
        n += 1
        converged = convergence_check(de.get_du(integrator))
        if converged:
            print(de.get_du(integrator))
    res = Result()
    res.coords = x_
    res.energy = pot.getEnergy(x_)
    res.rms = 0
    res.grad = 0
    res.nfev = function_evaluate_pot.nfev
    res.nsteps = n
    res.success = converged
    return res
