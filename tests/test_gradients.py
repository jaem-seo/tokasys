import numpy as np

from tokasys import default_problem
from tokasys.core.variables import normalize_reactor_variables
from tokasys.solvers.interfaces import make_vector_functions
from tokasys.validation.gradients import central_difference_jacobian


def test_objective_gradient_matches_finite_difference():
    problem = default_problem()
    funcs = make_vector_functions(problem)
    u0 = np.asarray(
        normalize_reactor_variables(problem.initial_variables, problem.variable_bounds),
        dtype=float,
    )
    _, ad_grad = funcs["objective_value_and_grad"](u0)
    fd_grad = central_difference_jacobian(
        lambda u: np.asarray(funcs["objective"](u)),
        u0,
        step=1.0e-4,
    )
    np.testing.assert_allclose(np.asarray(ad_grad), fd_grad, rtol=2.0e-3, atol=2.0e-4)
