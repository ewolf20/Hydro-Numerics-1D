import numpy as np 

"""
The functions in this file represent the differential equations of interest which are to be solved. 
There are two types of functions here for external calling: *_readable() and *_solver_wrapped().

Both functions encode the RHS of hydrodynamic equations which have been arranged in the form: 

partial A/partial t = ... 

Where the right hand side contains at most derivatives with respect to x. 

*_readable() functions are meant to be human-readable, and encode a single equation, e.g. continuity. 

    They have the calling signature f(*expanded_vars, *params), where:
        *expanded_vars: Array-likes representing the values of  
            linearly-independent quantities in the Euler equations 
            (e.g. u, rho but also partial u / partial x or pressure P)

        *params: scalars required to define the equation - e.g. a variable viscosity. 

    These functions return an array-like of the same shape as the elements of expanded_vars, representing 
    the RHS of the relevant equation broadcast over some axis (e.g. spatial)

    
*_solver_wrapped functions are meant to be passed to the differential equation solver. They encode a 
system of equations, e.g. the full Euler equations. 

    They have calling signature f(state_vars, *num_funcs, *params). 
        
        state_vars: A 3D ND array of shape (k, l, N). 
            The first dimension encodes the different state variables - for 1D hydrodynamics, the quantities rho, u, and (optionally) s. 
            The second dimension encodes a time axis, to accommodate steppers which are not first-order in time. By convention, 
                the smallest index is the most recent time. 
            The third dimension encodes a spatial axis.

        num_funcs: Functions which allow stepper-defined numerical evaluation of the quantities on the RHS of the equations. 

            Different finite difference schemes will evaluate e.g. partial u/partial x in different ways; 
            to avoid implementation-specific details on the level of the equations, we simply pass a function 
            pderiv_x, which the stepper-wrapped function can call. Each element of *num_funcs must have call signature 
            num_func(vars[i]); that is, it takes the shape (l, N) array for a given state variable.

            Remark: For some stepper-wrapped functions, it will be highly natural to pass functions unrelated 
            to derivative taking - e.g. a function mapping the state variables to pressure. These should not 
            be passed with num_funcs, but instead params, below. 

        *params: Any additional scalar parameters or functions necessary for the equation, e.g. tunable viscosity
        as above, or a pressure equation of state.

            Remark: While *params may be present here, they must be defined in a closure before passing to the 
            solver, which will only accept functions of the form *_solver_wrapped(vars, *num_funcs) 

    These functions return a 2-dimensional ND array of shape (k, N), representing the RHS of the system of equations, 
    broadcast over state variables and spatial axis (but not time). 
"""

#*_READABLE()

#GENERIC

def continuity_readable(rho, u, rho_pderiv_x, u_pderiv_x):
    return -1.0 * (u * rho_pderiv_x + rho * u_pderiv_x)

#INVISCID EQUATIONS

def momentum_Euler_generic_readable(rho, u, u_pderiv_x, P_pderiv_x):
    return -1.0 * (u * u_pderiv_x + (1.0 / rho) * P_pderiv_x)


def momentum_Euler_isentropic_readable(rho, u, rho_pderiv_x, u_pderiv_x, c):
    return -1.0 * (u * u_pderiv_x + (1.0 / rho) * np.square(c) * rho_pderiv_x)

def momentum_Euler_isentropic_polytropic_readable(rho, u, rho_pderiv_x, u_pderiv_x, gamma):
    return -1.0 * (u * u_pderiv_x + np.power(rho, gamma - 2) * rho_pderiv_x)


#STEPPER_WRAPPED 

def euler_equations_generic_solver_wrapped(state_vars, eval, pderiv_x, pressure_func):
    rho_vals, u_vals = state_vars 
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals)

    pressures = pressure_func(state_vars) 
    P_pderiv_x = pderiv_x(pressures) 

    rho_rhs = continuity_readable(rho, u, rho_pderiv_x, u_pderiv_x) 
    u_rhs = momentum_Euler_generic_readable(rho, u, u_pderiv_x, P_pderiv_x)

    return np.stack((rho_rhs, u_rhs))

def euler_equations_isentropic_solver_wrapped(state_vars, eval, pderiv_x, c_func):
    rho_vals, u_vals = state_vars 
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals)

    c_vals = c_func(state_vars) 
    c = eval(c_vals) 

    rho_rhs = continuity_readable(rho, u, rho_pderiv_x, u_pderiv_x)
    u_rhs = momentum_Euler_isentropic_readable(rho, u, rho_pderiv_x, u_pderiv_x, c)

    return np.stack((rho_rhs, u_rhs))


def euler_equations_isentropic_polytropic_solver_wrapped(state_vars, eval, pderiv_x, gamma): 
    rho_vals, u_vals = state_vars 
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals)

    rho_rhs = continuity_readable(rho, u, rho_pderiv_x, u_pderiv_x)
    u_rhs = momentum_Euler_isentropic_polytropic_readable(rho, u, rho_pderiv_x, u_pderiv_x)

    return np.stack((rho_rhs, u_rhs))
