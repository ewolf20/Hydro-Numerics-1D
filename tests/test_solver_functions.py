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

    times_forward_euler, state_vars_forward_euler = solver_functions.solve_equations(
        diffusive_equations_solver_wrapped, initial_state_reshaped, diffusive_xdiff, 
        diffusive_tdiff, diffusive_num_steps, method = "forward_euler", 
        deriv_order = 2, output_increment = 10000, print_progress = True)
    
    expected_final_state_fe = analytic_functions.diffusive_gaussian(
        times_forward_euler[-1], diffusive_xrange, diffusive_sample_sigma_0, 
        diffusive_sample_D)
    
    A_vals_fe, = state_vars_forward_euler 

    initial_A_vals_fe = A_vals_fe[0] 
    final_A_vals_fe = A_vals_fe[-1] 

    assert np.allclose(initial_A_vals_fe, initial_state)
    assert np.allclose(final_A_vals_fe, expected_final_state_fe, rtol = 1e-2, atol = 1e-5)

    #Now try leapfrog integration of the (advective) 1D Euler equations
    #Restrict interest to areas of nonzero density, and away from cusps
    advective_x_range = np.linspace(-0.8, 2.5, 3000) 
    advective_xdiff = np.diff(advective_x_range)[-1]
    #Set t step smaller than x step
    advective_tdiff = 1e-5
    advective_num_steps = 100000

    euler_gamma = 5/3 
    
    initial_time = 1.0 

    initial_rho = analytic_functions.polytropic_riemann_expansion_density_profile(
        initial_time, advective_x_range, gamma = euler_gamma)
    
    initial_velocity = analytic_functions.polytropic_riemann_expansion_velocity_profile(
        initial_time, advective_x_range, gamma = euler_gamma
    )

    initial_state_advective = np.stack((initial_rho, initial_velocity))

    def gamma_specified_euler(state_vars_stack, eval, pderiv_x):
        return hydrodynamic_equations.euler_equations_isentropic_polytropic_solver_wrapped(
            state_vars_stack, eval, pderiv_x, euler_gamma
        )
        


    times_leapfrog, state_vars_leapfrog = solver_functions.solve_equations(
        gamma_specified_euler, 
        initial_state_advective, advective_xdiff, advective_tdiff, advective_num_steps, 
        deriv_order = 1, output_increment = 10000, print_progress = True)

    final_time_leapfrog = times_leapfrog[-1] 

    expected_final_rho_leapfrog = analytic_functions.polytropic_riemann_expansion_density_profile(
        initial_time + final_time_leapfrog, advective_x_range, gamma = euler_gamma
    )
    expected_final_velocity_leapfrog = analytic_functions.polytropic_riemann_expansion_velocity_profile(
        initial_time + final_time_leapfrog, advective_x_range, gamma = euler_gamma
    )

    final_state_leapfrog = state_vars_leapfrog[:, -1] 
    final_rho_leapfrog, final_velocity_leapfrog = final_state_leapfrog 

    #Numerics are very dicey, but seem to be capturing the correct behavior...
    assert np.allclose(expected_final_rho_leapfrog, final_rho_leapfrog, atol = 2e-3, rtol = 1e-2)
    assert np.allclose(expected_final_velocity_leapfrog, final_velocity_leapfrog, atol = 1e-2, rtol = 5e-2)

