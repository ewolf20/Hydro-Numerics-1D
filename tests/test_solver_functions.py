import os 
import sys

import numpy as np 
import matplotlib.pyplot as plt
import scipy


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo = path_to_file + "/../../"
sys.path.insert(0, path_to_repo)

from Hydro_Numerics_1D.code import analytic_functions, hydrodynamic_equations, solver_functions 

TEST_TEMP_PATH = "Test_Temp"


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
        deriv_order = 2, output_increment = 10000,
        check_finite = False)
    
    expected_final_state_fe = analytic_functions.diffusive_gaussian(
        times_forward_euler[-1], diffusive_xrange, diffusive_sample_sigma_0, 
        diffusive_sample_D)
        
    A_vals_fe, = state_vars_forward_euler 

    initial_A_vals_fe = A_vals_fe[0]
    final_A_vals_fe = A_vals_fe[-1]


    assert np.allclose(initial_A_vals_fe, initial_state)
    assert np.allclose(final_A_vals_fe, expected_final_state_fe, rtol = 1e-2, atol = 1e-5)

    #Try (modified) leapfrog integration of the diffusive equation above 

    times_diffusive_leapfrog, state_vars_diffusive_leapfrog = solver_functions.solve_equations(
        diffusive_equations_solver_wrapped, initial_state_reshaped, diffusive_xdiff, 
        diffusive_tdiff, diffusive_num_steps, method = "leapfrog", deriv_order = 2, 
        output_increment = 10000, check_finite = False)
    
    A_vals_diffusive_leap, = state_vars_diffusive_leapfrog
    
    initial_A_vals_diffusive_leap = A_vals_diffusive_leap[0] 
    final_A_vals_diffusive_leap = A_vals_diffusive_leap[-1] 

    expected_final_state_diffusive_leap = analytic_functions.diffusive_gaussian(
        times_diffusive_leapfrog[-1], diffusive_xrange, diffusive_sample_sigma_0, 
        diffusive_sample_D)
    
    assert np.allclose(initial_A_vals_diffusive_leap, initial_state)
    assert np.allclose(final_A_vals_diffusive_leap, expected_final_state_diffusive_leap, rtol = 1e-2, atol = 1e-5)

    #Now try leapfrog integration of the (advective) 1D Euler equations
    #Restrict interest to areas of nonzero density, and away from cusps
    advective_x_range = np.linspace(-0.8, 2.5, 3000)
    advective_xdiff = np.diff(advective_x_range)[-1]
    #Set t step smaller than x step
    advective_tdiff = 1e-5
    advective_num_steps = 10000

    euler_gamma = 5/3 
    
    initial_time = 1.0 

    initial_rho = analytic_functions.polytropic_riemann_expansion_density_profile(
        initial_time, advective_x_range, gamma = euler_gamma)
    
    initial_velocity = analytic_functions.polytropic_riemann_expansion_velocity_profile(
        initial_time, advective_x_range, gamma = euler_gamma
    )

    initial_state_advective = np.stack((initial_rho, initial_velocity))

    continuity_equation_solver_wrapped = hydrodynamic_equations.continuity_equation_solver_wrapped 
    momentum_equation_solver_wrapped = hydrodynamic_equations.momentum_equation_solver_wrapped_factory(
        pressure_term_type = "polytropic", gamma_val = euler_gamma
    )

    euler_hydro_system = hydrodynamic_equations.hydro_system_function_factory(
        continuity_equation_solver_wrapped, momentum_equation_solver_wrapped
    )
        


    times_leapfrog, state_vars_leapfrog = solver_functions.solve_equations(
        euler_hydro_system, 
        initial_state_advective, advective_xdiff, advective_tdiff, advective_num_steps, 
        method = "leapfrog", deriv_order = 1, output_increment = 1000, check_finite = False)

    final_time_leapfrog = times_leapfrog[-1] 

    expected_final_rho_leapfrog = analytic_functions.polytropic_riemann_expansion_density_profile(
        initial_time + final_time_leapfrog, advective_x_range, gamma = euler_gamma
    )
    expected_final_velocity_leapfrog = analytic_functions.polytropic_riemann_expansion_velocity_profile(
        initial_time + final_time_leapfrog, advective_x_range, gamma = euler_gamma
    )

    final_state_leapfrog = state_vars_leapfrog[:, -1] 
    final_rho_leapfrog, final_velocity_leapfrog = final_state_leapfrog 

    #Numerics are imperfect, but captures correct behavior...
    assert np.allclose(expected_final_rho_leapfrog, final_rho_leapfrog, atol = 1e-4, rtol = 1e-3)
    assert np.allclose(expected_final_velocity_leapfrog, final_velocity_leapfrog, atol = 1e-4, rtol = 1e-3)

    #Verify that the incremental output is working... 
    if not os.path.exists(TEST_TEMP_PATH):
        os.mkdir(TEST_TEMP_PATH)

    time_file_path = os.path.join(TEST_TEMP_PATH, "Times.npy")
    state_var_file_path = os.path.join(TEST_TEMP_PATH, "State_Vars.npy")

    try:
        times_leapfrog, state_vars_leapfrog = solver_functions.solve_equations(
        euler_hydro_system,
        initial_state_advective, advective_xdiff, advective_tdiff, advective_num_steps,
        method = "leapfrog", deriv_order = 1, output_increment = 1000, check_finite = False,
        incremental_output = True, time_numpy_path = time_file_path,
        state_var_numpy_path = state_var_file_path)

        times_saved = np.load(time_file_path)
        state_vars_saved = np.load(state_var_file_path) 

        assert np.allclose(times_saved, times_leapfrog) 
        assert np.allclose(state_vars_saved, state_vars_leapfrog) 

    finally:
        os.remove(time_file_path)
        os.remove(state_var_file_path)
        os.rmdir(TEST_TEMP_PATH)

    #Check resumption after interruption... 
    if not os.path.exists(TEST_TEMP_PATH):
        os.mkdir(TEST_TEMP_PATH)

    time_file_path = os.path.join(TEST_TEMP_PATH, "Times.npy")
    state_var_file_path = os.path.join(TEST_TEMP_PATH, "State_Vars.npy")

    try:
        half_times_leapfrog, half_state_vars_leapfrog = solver_functions.solve_equations(
        euler_hydro_system,
        initial_state_advective, advective_xdiff, advective_tdiff, advective_num_steps // 2,
        method = "leapfrog", deriv_order = 1, output_increment = 1000, check_finite = False,
        incremental_output = True, time_numpy_path = time_file_path,
        state_var_numpy_path = state_var_file_path)

        assert np.allclose(half_times_leapfrog, times_leapfrog[:len(half_times_leapfrog)])
        assert np.allclose(half_state_vars_leapfrog, state_vars_leapfrog[:, :len(half_times_leapfrog)])

        #Now resume the outputs... 
        full_times_leapfrog, full_state_vars_leapfrog = solver_functions.solve_equations(
        euler_hydro_system,
        initial_state_advective, advective_xdiff, advective_tdiff, advective_num_steps,
        method = "leapfrog", deriv_order = 1, output_increment = 1000, check_finite = False,
        resume_existing = True, time_numpy_path = time_file_path,
        state_var_numpy_path = state_var_file_path)

        #Output should be the same as for unbroken evaluation... 
        assert np.allclose(full_times_leapfrog, times_leapfrog)
        assert np.allclose(full_state_vars_leapfrog, state_vars_leapfrog)

    finally:
        os.remove(time_file_path)
        os.remove(state_var_file_path)
        os.rmdir(TEST_TEMP_PATH)


    #Check evolution for an irregular number of steps 
    advective_irregular_num_steps = 1337
    times_irregular, _ = solver_functions.solve_equations(
        euler_hydro_system, 
        initial_state_advective, advective_xdiff, advective_tdiff, advective_irregular_num_steps, 
        method = "leapfrog", deriv_order = 1, output_increment = 1000, check_finite = False, 
        guarantee_last = True)
    
    assert np.isclose(times_irregular[-1], advective_tdiff * advective_irregular_num_steps)


    #Now deliberately engineer an unstable evolution of the equations... 
    num_x_samples_unstable = 2000
    advective_x_range_unstable = np.linspace(-2.0, 5.0, num_x_samples_unstable)
    advective_xdiff_unstable = np.diff(advective_x_range)[-1]
    #Set t step smaller than x step
    advective_tdiff_unstable = 1e-4
    advective_num_steps_unstable = 10000

    initial_time_unstable = 0.1

    euler_gamma = 5/3 

    ghost_density_unstable = 1e-2


    initial_rho_unstable = -0.5 * scipy.special.erf(advective_x_range_unstable / initial_time_unstable) + 0.5
    initial_velocity_unstable = np.zeros(advective_x_range_unstable.size)

    #Pad the initial rho distribution with a 'ghost density'
    initial_rho_unstable = (1 - ghost_density_unstable) * initial_rho_unstable + ghost_density_unstable

    initial_state_unstable = np.stack((initial_rho_unstable, initial_velocity_unstable))

    times_leapfrog_unstable, state_vars_leapfrog_unstable = solver_functions.solve_equations(
            euler_hydro_system,
            initial_state_unstable, advective_xdiff_unstable, advective_tdiff_unstable, advective_num_steps_unstable, 
            method = "leapfrog", deriv_order = 1, check_finite = False)
    
    assert not np.all(np.isfinite(state_vars_leapfrog_unstable))

    success_checked, times_leapfrog_unstable_checked, state_vars_leapfrog_unstable_checked = solver_functions.solve_equations(
            euler_hydro_system,
            initial_state_unstable, advective_xdiff_unstable, advective_tdiff_unstable, advective_num_steps_unstable, 
            method = "leapfrog", deriv_order = 1, check_finite = True)
    
    
    assert not success_checked
    assert np.all(np.isfinite(state_vars_leapfrog_unstable_checked))
    assert times_leapfrog_unstable[-1] > times_leapfrog_unstable_checked[-1]
    
