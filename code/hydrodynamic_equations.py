import numpy as np 

"""
The functions in this file represent the differential equations of interest which are to be solved. 
There are three types of functions here for external calling: *_equation(),  *_solver_wrapped(), and *_factory().

The first two types of function encode the RHS of hydrodynamic equations which have been arranged in the form: 

partial A/partial t = ... 

Where the right hand side contains no derivatives with respect to t. 

*_equation() functions are meant to be human-readable, and encode a single equation, e.g. continuity. 

    They have the calling signature f(*expanded_vars, *params), where:
        *expanded_vars: Array-likes representing the values of  
            linearly-independent quantities in the equations 
            (e.g. u, rho but also partial u / partial x or pressure P)

        *params: scalars required to define the equation - e.g. a variable viscosity. 

    These functions return an array-like of the same shape as the elements of expanded_vars, representing 
    the RHS of the relevant equation broadcast over any present axes (e.g. spatial)

    
*_solver_wrapped functions are meant to be passed to the differential equation solver, and are not as easily readable.

    They have calling signature f(state_vars_stack, *fin_diff_funcs) or f(t, x, state_vars_stack, *fin_diff_funcs)
        
        state_vars_stack: A 3D ND array of shape (k, l, N). 
            The first dimension encodes the different state variables - for 1D hydrodynamics, the quantities rho, u, and (optionally) s. 
            The second dimension encodes a time axis, to accommodate steppers which are not first-order in time. By convention, 
                the smallest index is the most recent time. 
            The third dimension encodes a spatial axis.

        fin_diff_funcs: Functions which allow stepper-defined finite-difference evaluation of the quantities on the RHS of the equations. 

            Different finite difference schemes will evaluate e.g. partial u/partial x in different ways; 
            to avoid implementation-specific details on the level of the equations, we simply pass a function 
            pderiv_x, which the stepper-wrapped function can call. 
            
                Each element of *fin_diff_funcs must have call signature 
                fin_diff_func(vars[i]); that is, it takes the shape (l, N) for a given state variable.
                It should have a return of shape (N).

        t, x: Arrays encoding the times t and positions x at which the solver-wrapped functions are to be evaluated. Currently used only 
        for situations where time- or position-dependent forces are applied. These must yield an array of shape (l, N) when broadcast against 
        each other.  


    These functions return an ND array of shape N), representing the RHS of the equation for the relevant state variable, broadcast over the 
    spatial axis. 

    Note that the solver-wrapped functions do not support other parameters - where these must be specified, they should be inserted via closures.
"""

#CONTINUITY

def continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x):
    return -1.0 * (u * rho_pderiv_x + rho * u_pderiv_x)

def continuity_equation_solver_wrapped(state_vars, *fin_diff_funcs):
    rho_vals, u_vals, *_ = state_vars 
    eval, pderiv_x, *_ = fin_diff_funcs
    rho = eval(rho_vals)
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals)
    return continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x)


#MOMENTUM

#Advective term
def _momentum_advective_term(u, u_pderiv_x):
    return -1.0 * (u * u_pderiv_x)

#Pressure terms

def _momentum_generic_pressure_term(rho, P_pderiv_x):
    return -1.0 * (1.0 / rho) * P_pderiv_x

def _momentum_isentropic_pressure_term(rho, rho_pderiv_x, c):
    return -1.0 * (1.0 / rho) * np.square(c) * rho_pderiv_x

def _momentum_isentropic_polytropic_pressure_term(rho, rho_pderiv_x, gamma): 
    return -1.0 * np.power(rho, gamma - 2) * rho_pderiv_x

#Viscous terms 
def _momentum_constant_eta_viscous_term(rho, u_pderiv_xx, eta): 
    return 1.0 / (rho) * (4/3 * eta) * u_pderiv_xx

def _momentum_constant_nu_viscous_term(rho, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, nu): 
    return 4/3 * (nu * u_pderiv_xx + 1.0 / rho * nu * rho_pderiv_x * u_pderiv_x)

def _momentum_arb_eta_viscous_term(rho, u_pderiv_x, u_pderiv_xx, eta, eta_pderiv_x): 
    return 1.0 / rho * 4/3 * (eta * u_pderiv_xx + eta_pderiv_x * u_pderiv_x)


#Function factory for the momentum equation. Note: External forces are not included at this stage. 
def momentum_equation_solver_wrapped_factory(pressure_term_type = "arbitrary", pressure_func = None, c_func = None, gamma_val = None, 
                                             viscous_term_type = "inviscid", eta_func = None, eta_val = None, nu_val = None):
    ALLOWED_PRESSURE_TERM_TYPES = ["arbitrary", "isentropic", "polytropic"]
    if not pressure_term_type in ALLOWED_PRESSURE_TERM_TYPES:
        raise ValueError("Unrecognized pressure term type. Allowed values are: {0}".format(ALLOWED_PRESSURE_TERM_TYPES))
    
    ALLOWED_VISCOUS_TERM_TYPES = ["inviscid", "const_eta", "const_nu", "arbitrary"]
    if not viscous_term_type in ALLOWED_VISCOUS_TERM_TYPES:
        raise ValueError("Unrecognized viscous term type. Allowed values are: {0}".format(ALLOWED_VISCOUS_TERM_TYPES))
    

    def momentum_equation_solver_wrapped(state_vars, *fin_diff_funcs):
        rho_vals, u_vals, *_ = state_vars
        #Pderiv_xx is only used in viscous terms 
        if viscous_term_type == "inviscid":
            eval, pderiv_x, *_ = fin_diff_funcs 
        else:
            eval, pderiv_x, pderiv_xx, *_ = fin_diff_funcs
        
        #Advective term 
        u = eval(u_vals) 
        u_pderiv_x = pderiv_x(u_vals)
        advective_term = _momentum_advective_term(u, u_pderiv_x) 

        #Pressure term 
        rho = eval(rho_vals)
        #Not always necessary, but often; we predefine it to not waste evaluations
        rho_pderiv_x = pderiv_x(rho_vals)
        if pressure_term_type == "arbitrary":
            P_vals = pressure_func(state_vars) 
            P_pderiv_x = pderiv_x(P_vals)
            pressure_term = _momentum_generic_pressure_term(rho, P_pderiv_x)
        elif pressure_term_type == "isentropic":
            c_vals = c_func(state_vars) 
            c = eval(c_vals) 
            pressure_term = _momentum_isentropic_pressure_term(rho, rho_pderiv_x, c)
        elif pressure_term_type == "polytropic":
            pressure_term = _momentum_isentropic_polytropic_pressure_term(rho, rho_pderiv_x, gamma_val)


        #Viscous Term
        if viscous_term_type == "inviscid":
            viscous_term = 0.0 
        elif viscous_term_type == "const_eta":
            u_pderiv_xx = pderiv_xx(u_vals)
            viscous_term = _momentum_constant_eta_viscous_term(rho, u_pderiv_xx, eta_val)
        elif viscous_term_type == "const_nu":
            u_pderiv_xx = pderiv_xx(u_vals)
            viscous_term = _momentum_constant_nu_viscous_term(rho, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, nu_val)
        elif viscous_term_type == "arbitrary":
            u_pderiv_xx = pderiv_xx(u_vals)
            eta_vals = eta_func(state_vars) 
            eta = eval(eta_vals) 
            eta_pderiv_x = pderiv_x(eta_vals)
            viscous_term = _momentum_arb_eta_viscous_term(rho, u_pderiv_x, u_pderiv_xx, eta, eta_pderiv_x)

        return advective_term + pressure_term + viscous_term 

    return momentum_equation_solver_wrapped

#ENTROPY

def _entropy_advective_term(u, s_pderiv_x):
    return -u * s_pderiv_x

def _entropy_dissipative_visc_term(rho, T, u_pderiv_x, eta):
    return 1.0 / (rho * T) * 4/3 * eta * np.square(u_pderiv_x)

def _entropy_dissipative_thermal_term_const_kappa(rho, T, T_pderiv_xx, kappa):
    return 1.0 / (rho * T) * kappa * T_pderiv_xx 

def _entropy_dissipative_thermal_term_const_kappa_prime(rho, T, rho_pderiv_x, T_pderiv_x, T_pderiv_xx, kappa_prime):
    return kappa_prime * (1.0 / T * T_pderiv_xx + 1.0 / (rho * T) * rho_pderiv_x * T_pderiv_x)

def _entropy_dissipative_thermal_term_arb_kappa(rho, T, T_pderiv_x, T_pderiv_xx, kappa, kappa_pderiv_x):
    return 1.0 / (rho * T) * (kappa_pderiv_x * T_pderiv_x + kappa * T_pderiv_xx)


def entropy_equation_solver_wrapped_factory(T_func,
                                             viscous_term_type = "arbitrary", eta_val = None, nu_val = None, eta_func = None,
                                             thermal_term_type = "arbitrary", kappa_val = None, kappa_prime_val = None, kappa_func = None):

    ALLOWED_VISCOUS_TERM_TYPES = ["arbitrary", "const_eta", "const_nu", "inviscid"]
    if not viscous_term_type in ALLOWED_VISCOUS_TERM_TYPES:
        raise ValueError("Unrecognized viscous term type. Allowed values are: {0}".format(ALLOWED_VISCOUS_TERM_TYPES))
    
    ALLOWED_THERMAL_TERM_TYPES = ["arbitrary", "const_kappa", "const_kappa_prime", "none"]
    if not thermal_term_type in ALLOWED_THERMAL_TERM_TYPES:
        raise ValueError("Unrecognized thermal term type. Allowed values are: {0}".format(ALLOWED_THERMAL_TERM_TYPES))


    def entropy_equation_solver_wrapped(state_vars, *fin_diff_funcs):
        rho_vals, u_vals, s_vals = state_vars 
        eval, pderiv_x, pderiv_xx = fin_diff_funcs
        u = eval(u_vals) 
        s_pderiv_x = pderiv_x(s_vals) 
        advective_term = _entropy_advective_term(u, s_pderiv_x) 

        T_vals = T_func(state_vars) 
        T = eval(T_vals) 
        rho = eval(rho_vals)

        if viscous_term_type == "inviscid":
            dissipative_term_viscous = 0.0 
        else:
            u_pderiv_x = pderiv_x(u_vals)
            if viscous_term_type == "arbitrary":
                eta_vals = eta_func(state_vars)
                eta = eval(eta_vals)
            elif viscous_term_type == "const_eta":
                eta = eta_val
            elif viscous_term_type == "const_nu":
                eta_vals = rho_vals * nu_val 
                eta = eval(eta_vals) 
            dissipative_term_viscous = _entropy_dissipative_visc_term(rho, T, u_pderiv_x, eta)

        if thermal_term_type == "none":
            dissipative_term_thermal = 0.0
        elif thermal_term_type == "const_kappa":
            T_pderiv_xx = pderiv_xx(T_vals)
            dissipative_term_thermal = _entropy_dissipative_thermal_term_const_kappa(rho, T, T_pderiv_xx, kappa_val)
        elif thermal_term_type == "const_kappa_prime":
            rho_pderiv_x = pderiv_x(rho_vals) 
            T_pderiv_x = pderiv_x(T_vals)
            T_pderiv_xx = pderiv_xx(T_vals)
            dissipative_term_thermal = _entropy_dissipative_thermal_term_const_kappa_prime(rho, T, rho_pderiv_x, T_pderiv_x, 
                                                                                           T_pderiv_xx, kappa_prime_val)
        elif thermal_term_type == "arbitrary": 
            kappa_vals = kappa_func(state_vars) 
            kappa = eval(kappa_vals) 
            kappa_pderiv_x = pderiv_x(kappa_vals)
            dissipative_term_thermal = _entropy_dissipative_thermal_term_arb_kappa(rho, T, T_pderiv_x, T_pderiv_xx, kappa, kappa_pderiv_x)

        return advective_term + dissipative_term_viscous + dissipative_term_thermal

    return entropy_equation_solver_wrapped


#PURELY DIFFUSIVE DYNAMICS
#Included mostly for testing purposes...

def diffusive_equation(A_pderiv_xx, diffusivity):
    return diffusivity * A_pderiv_xx

#Wrap functions into a complete system, optionally adding an externally-applied acceleration.
#If specified, ext_accel_func has call signature (t, x), where t and x are assumed to be able to broadcast together to a (l, N) array. 

def hydro_system_function_factory(continuity_solver_wrapped, momentum_solver_wrapped, entropy_solver_wrapped = None, 
                                  ext_accel_type = "none", ext_accel_val = None, ext_accel_func = None):
    
    ALLOWED_EXT_ACCEL_TYPES = ["none", "constant_accel", "arbitrary"]
    if not ext_accel_type in ALLOWED_EXT_ACCEL_TYPES:
        raise ValueError("Unrecognized ext_accel_type; allowed values are: {0}".format(ALLOWED_EXT_ACCEL_TYPES))
    
    def _hydro_system_function_no_force(state_vars, *fin_diff_funcs):
        continuity_rhs = continuity_solver_wrapped(state_vars, *fin_diff_funcs)
        momentum_rhs_no_force = momentum_solver_wrapped(state_vars, *fin_diff_funcs)
        if entropy_solver_wrapped is None: 
            return np.stack((continuity_rhs, momentum_rhs_no_force))
        else:
            entropy_rhs = entropy_solver_wrapped(state_vars, *fin_diff_funcs)
            return np.stack((continuity_rhs, momentum_rhs_no_force, entropy_rhs))

    #Add the external acceleration; do it this way because it changes the signature... 

    if ext_accel_type == "arbitrary":
        def hydro_system_function(t, x, state_vars, *fin_diff_funcs):
            eval, *_ = fin_diff_funcs
            rhs = _hydro_system_function_no_force(state_vars, *fin_diff_funcs) 
            accel_vals = ext_accel_func(t, x)
            accel = eval(accel_vals)
            #Add the accel to the momentum equation only. 
            rhs[1] += accel
            return rhs
    elif ext_accel_type == "constant_accel":
        def hydro_system_function(state_vars, *fin_diff_funcs):
            rhs = _hydro_system_function_no_force(state_vars, *fin_diff_funcs) 
            rhs[1] += ext_accel_val
            return rhs
    elif ext_accel_type == "none":
        def hydro_system_function(state_vars, *fin_diff_funcs):
            return _hydro_system_function_no_force(state_vars, *fin_diff_funcs)

    return hydro_system_function

"""Given input state vars, calculate the total energy. 

Params: 
    state_vars_stack: An ND array of shape (k, ..., N), with k >= 2. The first axis is assumed to encode the state variables
        (rho, u, ...), and the last encodes the 1D position axis - any other axes, if present, are broadcast over. 

    epsilon_func: A function epsilon_func(state_vars) which takes as input the (k, ...) state vars ND array and returns 
        the (normalized) energy per unit mass, according to the normalization: 

        epsilon' = epsilon / c_0^2

        If more axes than the first are present, they are broadcast over.

    x_diff: Default 1.0. If specified, the x-difference (in sim units) between adjacent points along the last axis of 
        state_vars_stack. Used as input to np.trapz.
"""
def get_hydro_total_energy(state_vars_stack, epsilon_func, x_diff = 1.0): 
    rho, u, *_ = state_vars_stack 
    epsilon = epsilon_func(state_vars_stack) 
    integrand = 0.5 * rho * np.square(u) + rho * epsilon
    return np.trapz(integrand, dx = x_diff, axis = -1)