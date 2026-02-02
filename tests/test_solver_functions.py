import os 
import sys

import numpy as np 
import matplotlib.pyplot as plt


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo = path_to_file + "/../../"
sys.path.insert(0, path_to_repo)

from Hydro_Numerics_1D.code import analytic_functions, hydrodynamic_equations, solver_functions 


def test_solve_equations():
    #Begin by testing forward Euler integration of diffusive equations

    diffusive_xrange = np.linspace(-5, 5, 1000)
    diffusive_xdiff = np.diff(diffusive_xrange)[0]
    diffusive_tdiff = 1e-4
    diffusive_num_steps = 20000
    diffusive_sample_D = 1.0 
    diffusive_sample_sigma_0 = 1.0 

    #Define a solver-wrapped version of the diffusive equation 
    def diffusive_equations_solver_wrapped(state_vars_stack, eval, pderiv_x, 
                                           pderiv_xx):
        A, = state_vars_stack 
        A_pderiv_xx = pderiv_xx(A) 
        A_rhs = hydrodynamic_equations.diffusive_equation(A_pderiv_xx, diffusive_sample_D)
        return np.array([A_rhs, ])

    initial_state = analytic_functions.diffusive_gaussian(0, diffusive_xrange, 
                                            diffusive_sample_sigma_0, diffusive_sample_D)
    
    initial_state_reshaped = np.expand_dims(initial_state, axis = 0)

    times, state_vars = solver_functions.solve_equations(
        diffusive_equations_solver_wrapped, initial_state_reshaped, diffusive_xdiff, 
        diffusive_tdiff, diffusive_num_steps, method = "forward_euler", 
        deriv_order = 2, output_increment = 10000, print_progress = True)
    
    expected_final_state = analytic_functions.diffusive_gaussian(
        times[-1], diffusive_xrange, diffusive_sample_sigma_0, 
        diffusive_sample_D)
    
    A_vals, = state_vars 

    initial_A_vals = A_vals[0] 
    final_A_vals = A_vals[-1] 

    plt.plot(diffusive_xrange, expected_final_state) 
    plt.plot(diffusive_xrange, final_A_vals)
    plt.show()

    assert np.allclose(final_A_vals, expected_final_state, rtol = 1e-2, atol = 1e-5)
    