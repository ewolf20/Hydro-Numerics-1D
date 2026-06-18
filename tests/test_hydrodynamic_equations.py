import os 
import sys

import numpy as np 
import matplotlib.pyplot as plt


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo = path_to_file + "/../../"
sys.path.insert(0, path_to_repo)

from Hydro_Numerics_1D.code import analytic_functions, hydrodynamic_equations

REFERENCE_DELTA_T = 0.001
REFERENCE_X_SAMPS = 1000
REFERENCE_RANGE_LOWER = -0.8
REFERENCE_RANGE_UPPER = 2.8
x_range = np.linspace(REFERENCE_RANGE_LOWER, REFERENCE_RANGE_UPPER, REFERENCE_X_SAMPS)
x_diff = np.diff(x_range)[-1]
T_CENTER = 1.0


SCALE_INVARIANT_GAMMA = 5/3 


#Get reference profiles from the polytropic riemann expansion, which satisfies the Euler equations, 
#Use these profiles to validate that the Euler equations are written down correctly!
def _get_reference_profiles():
    #Sample only from the range away from cusps
    x_range = np.linspace(REFERENCE_RANGE_LOWER, REFERENCE_RANGE_UPPER, REFERENCE_X_SAMPS)

    t_range = np.array([T_CENTER - REFERENCE_DELTA_T, T_CENTER, T_CENTER + REFERENCE_DELTA_T])

    t_grid, x_grid = np.meshgrid(t_range, x_range, indexing = "ij") 

    density_profile_stack = analytic_functions.polytropic_riemann_expansion_density_profile(t_grid, x_grid, 
                                                                                      gamma = SCALE_INVARIANT_GAMMA)
    velocity_profile_stack = analytic_functions.polytropic_riemann_expansion_velocity_profile(t_grid, x_grid, 
                                                                                        gamma = SCALE_INVARIANT_GAMMA)
    
    return(density_profile_stack, velocity_profile_stack)

def _get_euler_reference_vals_and_derivs(): 
    density_profile_stack, velocity_profile_stack = _get_reference_profiles()
    rho = density_profile_stack[1] 
    u = velocity_profile_stack[1] 
    rho_pderiv_t = 1 / (2.0 * REFERENCE_DELTA_T) * (density_profile_stack[2] - density_profile_stack[0])
    u_pderiv_t = 1 / (2.0 * REFERENCE_DELTA_T) * (velocity_profile_stack[2] - velocity_profile_stack[0])

    #Now do the derivatives 
    rho_pderiv_x = 1 / (2.0 * x_diff) * (rho[2:] - rho[:-2]) 
    u_pderiv_x = 1 / (2.0 * x_diff) * (u[2:] - u[:-2])

    #Also need u_pderiv_xx for viscosity terms 
    u_pderiv_xx = 1 / (np.square(x_diff)) * (u[2:] - 2 * u[1:-1] + u[:-2])


    #Clip to match with spatial derivatives
    rho = rho[1:-1] 
    u = u[1:-1] 
    rho_pderiv_t = rho_pderiv_t[1:-1] 
    u_pderiv_t = u_pderiv_t[1:-1]

    return (rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, rho_pderiv_t, u_pderiv_t)



def test_continuity_equation():
    (rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx,
      rho_pderiv_t, u_pderiv_t) = _get_euler_reference_vals_and_derivs() 
    
    continuity_rhs = hydrodynamic_equations.continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x)
    #Loose tolerances because of discretization error... 
    assert np.allclose(rho_pderiv_t, continuity_rhs, rtol = 1e-5, atol = 1e-6)


#Now test the momentum function factory under various conditions... 


# def test_momentum_euler_isentropic_polytropic_equation():
#     (rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx,
#      rho_pderiv_t, u_pderiv_t) = _get_euler_reference_vals_and_derivs() 
    
#     momentum_euler_rhs = hydrodynamic_equations.momentum_euler_isentropic_polytropic_equation(
#         rho, u, rho_pderiv_x, u_pderiv_x, SCALE_INVARIANT_GAMMA
#     )

#     assert np.allclose(u_pderiv_t, momentum_euler_rhs, rtol = 1e-5, atol = 1e-6)


# def test_momentum_euler_isentropic_equation():
#     (rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx,
#       rho_pderiv_t, u_pderiv_t) = _get_euler_reference_vals_and_derivs()
    
#     c_values = analytic_functions.normalized_speed_of_sound_isentropic_polytropic_eos(
#         rho, gamma = SCALE_INVARIANT_GAMMA
#     )

#     momentum_euler_rhs = hydrodynamic_equations.momentum_euler_isentropic_equation(
#         rho, u, rho_pderiv_x, u_pderiv_x, c_values
#     )

#     assert np.allclose(u_pderiv_t, momentum_euler_rhs, rtol = 1e-5, atol = 1e-6)

# def test_momentum_euler_generic_equation():
#     (rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx,
#      rho_pderiv_t, u_pderiv_t) = _get_euler_reference_vals_and_derivs()
    
#     pressure_values = analytic_functions.normalized_pressure_isentropic_polytropic_eos(
#         rho, gamma = SCALE_INVARIANT_GAMMA
#     )

#     P_pderiv_x = 1.0 / (2 * x_diff) * (pressure_values[2:] - pressure_values[:-2])

#     rho = rho[1:-1]
#     u = u[1:-1]
#     u_pderiv_x = u_pderiv_x[1:-1] 
#     u_pderiv_t = u_pderiv_t[1:-1] 

#     momentum_euler_rhs = hydrodynamic_equations.momentum_euler_generic_equation(
#         rho, u, u_pderiv_x, P_pderiv_x, 
#     )

#     #Slightly looser tolerance - maybe the pressure derivative is the issue...
#     assert np.allclose(u_pderiv_t, momentum_euler_rhs, rtol = 1e-4, atol = 1e-5)


#Just test viscosity for the polytropic case... 

# #Const viscosity
# def test_momentum_euler_const_viscosity_isentropic_polytropic_equation(): 
#     (rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx,
#      rho_pderiv_t, u_pderiv_t) = _get_euler_reference_vals_and_derivs() 
    
#     sample_eta = 0.1
    
#     momentum_euler_rhs_viscous = hydrodynamic_equations.momentum_euler_const_viscosity_isentropic_polytropic_equation(
#         rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, SCALE_INVARIANT_GAMMA, sample_eta
#     )

#     momentum_euler_rhs_viscous_subtracted = momentum_euler_rhs_viscous - 1.0 / rho * (4/3) * u_pderiv_xx * sample_eta

#     assert np.allclose(u_pderiv_t, momentum_euler_rhs_viscous_subtracted, rtol = 1e-5, atol = 1e-6)



# #Var viscosity
# def test_momentum_euler_var_viscosity_isentropic_polytropic_equation():
#     (rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx,
#      rho_pderiv_t, u_pderiv_t) = _get_euler_reference_vals_and_derivs()
    
#     sample_nu = 0.1
#     eta = rho * sample_nu
#     eta_pderiv_x = rho_pderiv_x * sample_nu
    
#     momentum_euler_rhs_viscous = hydrodynamic_equations.momentum_euler_var_viscosity_isentropic_polytropic_equation(
#         rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, eta, eta_pderiv_x, SCALE_INVARIANT_GAMMA)

#     viscous_correction = 1.0 / rho * (4/3) * (u_pderiv_xx * eta + eta_pderiv_x * u_pderiv_x)
#     momentum_euler_rhs_viscous_subtracted = momentum_euler_rhs_viscous - viscous_correction

#     assert np.allclose(u_pderiv_t, momentum_euler_rhs_viscous_subtracted, rtol = 1e-5, atol = 1e-6)


def test_diffusivity_equation():
    sample_x_values = np.linspace(-3, 3, 10000)
    sample_x_diff = np.diff(sample_x_values)[0]
    sample_t = 1.5
    sample_D = 1.2
    sample_sigma_0 = 1.3
    t_range = np.array([sample_t - REFERENCE_DELTA_T, sample_t, sample_t + REFERENCE_DELTA_T])
    t_grid, x_grid = np.meshgrid(t_range, sample_x_values, indexing = "ij")
    function_values = analytic_functions.diffusive_gaussian(t_grid, x_grid, sample_sigma_0, sample_D)

    fun_pderiv_t = 1.0 / (2 * REFERENCE_DELTA_T) * (function_values[2] - function_values[0]) 

    fun_values = function_values[1] 
    fun_pderiv_x = np.gradient(fun_values, sample_x_diff, edge_order = 2)
    fun_pderiv_xx = np.gradient(fun_pderiv_x, sample_x_diff, edge_order = 2)

    expected_fun_pderiv_t = hydrodynamic_equations.diffusive_equation(fun_pderiv_xx, sample_D) 

    assert np.allclose(expected_fun_pderiv_t, fun_pderiv_t, rtol = 1e-4, atol = 1e-5)


#Now verify correct behavior of solver-wrapped functions... 


#For testing, define implementation of eval, pderiv_x, etc...
def _sample_eval(state_var):
    return state_var[0] 


def _sample_pderiv_x(state_var):
    current_var = state_var[0]
    return np.gradient(current_var, x_diff, edge_order = 2)

def _sample_pderiv_xx(state_var):
    current_var = state_var[0]
    return np.gradient(
        np.gradient(current_var, x_diff, edge_order = 2),
    x_diff, edge_order = 2)



def test_momentum_equation_solver_wrapped_factory():

    (rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx,
     rho_pderiv_t, u_pderiv_t) = _get_euler_reference_vals_and_derivs()
    
    state_vars_stack = np.stack((rho, u))
    #Reshape to accommodate required timestep axis for solver-wrapped function...
    state_vars_stack_reshaped = np.expand_dims(state_vars_stack, axis = 1)

    expected_rhs = u_pderiv_t

    #First do it with an explicitly polytropic EOS and make it inviscid
    momentum_equation_solver_wrapped_polytropic_inviscid = hydrodynamic_equations.momentum_equation_solver_wrapped_factory(
        pressure_term_type = "polytropic", gamma_val = SCALE_INVARIANT_GAMMA, 
        viscous_term_type = "inviscid"
    )

    def _validate_rhs(expected_rhs, returned_rhs):
        assert np.allclose(expected_rhs, returned_rhs, rtol = 3e-4, atol = 1e-4)

    
    returned_rhs_polytropic_inviscid = momentum_equation_solver_wrapped_polytropic_inviscid(state_vars_stack_reshaped, 
                                                    _sample_eval, _sample_pderiv_x)
    
    _validate_rhs(expected_rhs, returned_rhs_polytropic_inviscid)
    

    #Now do it with an isentropic speed of sound, and specify c as for a polytropic EOS... 
    def c_func(state_vars):
        rho, *_ = state_vars 
        return np.power(rho, (SCALE_INVARIANT_GAMMA - 1) / 2.0)
    
    momentum_equation_solver_wrapped_isentropic_inviscid = hydrodynamic_equations.momentum_equation_solver_wrapped_factory(
        pressure_term_type = "isentropic", c_func = c_func, viscous_term_type = "inviscid")
    
    returned_rhs_isentropic_inviscid = momentum_equation_solver_wrapped_isentropic_inviscid(state_vars_stack_reshaped, 
                                                                            _sample_eval, _sample_pderiv_x)
    
    _validate_rhs(expected_rhs, returned_rhs_isentropic_inviscid)

    #Now with an arbitrary pressure function, again encoding polytropic interactions 
    def pressure_func(state_vars):
        rho, *_ = state_vars
        return 1.0 / SCALE_INVARIANT_GAMMA * np.power(rho, SCALE_INVARIANT_GAMMA)

    momentum_equation_solver_wrapped_arbitrary_inviscid = hydrodynamic_equations.momentum_equation_solver_wrapped_factory(
        pressure_term_type = "arbitrary", pressure_func = pressure_func, viscous_term_type = "inviscid")

    returned_rhs_arbitrary_inviscid = momentum_equation_solver_wrapped_arbitrary_inviscid(state_vars_stack_reshaped, 
                                                                        _sample_eval, _sample_pderiv_x)
    _validate_rhs(expected_rhs, returned_rhs_arbitrary_inviscid)


    #Now add a viscous term
    SAMPLE_ETA_VAL = 0.1
    momentum_equation_solver_wrapped_polytropic_const_eta = hydrodynamic_equations.momentum_equation_solver_wrapped_factory(
        pressure_term_type = "polytropic", gamma_val = SCALE_INVARIANT_GAMMA, viscous_term_type = "const_eta", eta_val = SAMPLE_ETA_VAL)

    returned_rhs_polytropic_const_eta = momentum_equation_solver_wrapped_polytropic_const_eta(state_vars_stack_reshaped, 
                                                                                _sample_eval, _sample_pderiv_x, _sample_pderiv_xx)
    #subtract off the viscous term 
    const_eta_viscous_term = hydrodynamic_equations._momentum_constant_eta_viscous_term(rho, u_pderiv_xx, SAMPLE_ETA_VAL)
    returned_rhs_polytropic_const_eta_viscous_corrected = returned_rhs_polytropic_const_eta - const_eta_viscous_term
    _validate_rhs(expected_rhs, returned_rhs_polytropic_const_eta_viscous_corrected)

    #Now use a constant nu... 
    SAMPLE_NU_VAL = 0.1
    momentum_equation_solver_wrapped_polytropic_const_nu = hydrodynamic_equations.momentum_equation_solver_wrapped_factory(
        pressure_term_type = "polytropic", gamma_val = SCALE_INVARIANT_GAMMA, viscous_term_type = "const_nu", nu_val = SAMPLE_NU_VAL)

    returned_rhs_polytropic_const_nu = momentum_equation_solver_wrapped_polytropic_const_nu(state_vars_stack_reshaped, 
                                                                        _sample_eval, _sample_pderiv_x, _sample_pderiv_xx)
    const_nu_viscous_term = hydrodynamic_equations._momentum_constant_nu_viscous_term(rho, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, SAMPLE_NU_VAL)
    returned_rhs_polytropic_const_nu_viscous_corrected = returned_rhs_polytropic_const_nu - const_nu_viscous_term
    _validate_rhs(expected_rhs, returned_rhs_polytropic_const_nu_viscous_corrected)

    #Now with an arbitrary viscosity, though we replicate the constant nu... 
    def eta_func(state_vars):
        rho, *_ = state_vars
        return rho * SAMPLE_NU_VAL

    momentum_equation_solver_wrapped_polytropic_arbitrary = hydrodynamic_equations.momentum_equation_solver_wrapped_factory(
        pressure_term_type = "polytropic", gamma_val = SCALE_INVARIANT_GAMMA, viscous_term_type = "arbitrary", eta_func = eta_func
    )

    returned_rhs_polytropic_arbitrary = momentum_equation_solver_wrapped_polytropic_arbitrary(state_vars_stack_reshaped, 
                                                                                _sample_eval, _sample_pderiv_x, _sample_pderiv_xx)
    #Result should be identical to that obtained for a constant nu...
    _validate_rhs(returned_rhs_polytropic_const_nu, returned_rhs_polytropic_arbitrary)



def test_hydro_system_function_factory():
    (rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx,
     rho_pderiv_t, u_pderiv_t) = _get_euler_reference_vals_and_derivs()
    
    state_vars_stack = np.stack((rho, u))
    #Reshape to accommodate required timestep axis for solver-wrapped function...
    state_vars_stack_reshaped = np.expand_dims(state_vars_stack, axis = 1)
    
    continuity_equation_solver_wrapped = hydrodynamic_equations.continuity_equation_solver_wrapped
    momentum_equation_solver_wrapped = hydrodynamic_equations.momentum_equation_solver_wrapped_factory(
        pressure_term_type = "polytropic", gamma_val = SCALE_INVARIANT_GAMMA, viscous_term_type = "inviscid")
    
    hydro_system_function = hydrodynamic_equations.hydro_system_function_factory(
        continuity_equation_solver_wrapped, momentum_equation_solver_wrapped
    )

    hydro_system_rhs = hydro_system_function(state_vars_stack_reshaped, _sample_eval, _sample_pderiv_x)

    expected_rhs = np.stack((rho_pderiv_t, u_pderiv_t))
    assert np.allclose(expected_rhs, hydro_system_rhs, rtol = 1e-4, atol = 1e-5)

    #Now test adding a force 
    SAMPLE_ACCEL = 0.1
    hydro_system_function_const_accel = hydrodynamic_equations.hydro_system_function_factory(
        continuity_equation_solver_wrapped, momentum_equation_solver_wrapped, 
        ext_accel_type = "constant_accel", ext_accel_val = SAMPLE_ACCEL
    )

    hydro_system_rhs_const_accel = hydro_system_function_const_accel(state_vars_stack_reshaped, _sample_eval, _sample_pderiv_x)
    expected_rhs_const_accel = expected_rhs + np.expand_dims(np.array([0, SAMPLE_ACCEL]), axis = 1)
    assert np.allclose(expected_rhs_const_accel, hydro_system_rhs_const_accel, rtol = 1e-4, atol = 1e-5)

    sample_x_vals = np.arange(state_vars_stack_reshaped.shape[-1])
    SAMPLE_T = 0.0
    sample_t_vals = np.array([SAMPLE_T])

    t_vals_expanded = np.expand_dims(sample_t_vals, axis = 1) 
    x_vals_expanded = np.expand_dims(sample_x_vals, axis = 0)

    def accel_function(t, x):
        return np.cos(t) * x

    expected_accel_vals = accel_function(SAMPLE_T, sample_x_vals)

    hydro_system_function_arb_accel = hydrodynamic_equations.hydro_system_function_factory(
        continuity_equation_solver_wrapped, momentum_equation_solver_wrapped,
        ext_accel_type = "arbitrary", ext_accel_func = accel_function)
    

    hydro_system_rhs_arb_accel = hydro_system_function_arb_accel(t_vals_expanded, x_vals_expanded, 
                                        state_vars_stack_reshaped, _sample_eval, _sample_pderiv_x)
    
    expected_rhs_arb_accel = expected_rhs + np.stack((np.zeros(len(expected_accel_vals)), expected_accel_vals))
    assert np.allclose(expected_rhs_arb_accel, hydro_system_rhs_arb_accel, rtol = 1e-4, atol = 1e-5)





def test_get_hydro_total_energy():
    #Use the analytic result to verify constant energy for a simple case 
    def sample_epsilon_func(state_vars):
        rho, *_ = state_vars 
        epsilon = 1.0 / (SCALE_INVARIANT_GAMMA * (SCALE_INVARIANT_GAMMA - 1)) * np.power(rho, SCALE_INVARIANT_GAMMA - 1)
        return epsilon 

    time_range = np.linspace(0.1, 1, 10)
    X_RANGE_MIN = -3 
    X_RANGE_MAX = 6
    x_range = np.linspace(-3, 6, 100000)
    x_diff = np.diff(x_range)[0]
    time_grid, x_grid = np.meshgrid(time_range, x_range, indexing = "ij")
    analytic_rho_values = analytic_functions.polytropic_riemann_expansion_density_profile(
        time_grid, x_grid, gamma = SCALE_INVARIANT_GAMMA
    )
    analytic_u_values = analytic_functions.polytropic_riemann_expansion_velocity_profile(
        time_grid, x_grid, gamma = SCALE_INVARIANT_GAMMA
    )
    state_vars_stack = np.stack((analytic_rho_values, analytic_u_values))
    energies = hydrodynamic_equations.get_hydro_total_energy(state_vars_stack, sample_epsilon_func, 
                                                             x_diff = x_diff)
    expected_energy = 1.0 / (SCALE_INVARIANT_GAMMA * (SCALE_INVARIANT_GAMMA - 1)) * np.abs(X_RANGE_MIN)
    assert np.allclose(energies, expected_energy)

