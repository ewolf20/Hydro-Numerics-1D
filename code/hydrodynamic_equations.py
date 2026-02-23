import numpy as np 

"""
The functions in this file represent the differential equations of interest which are to be solved. 
There are two types of functions here for external calling: *_equation() and *_solver_wrapped().

Both functions encode the RHS of hydrodynamic equations which have been arranged in the form: 

partial A/partial t = ... 

Where the right hand side contains at most derivatives with respect to x. 

*_equation() functions are meant to be human-readable, and encode a single equation, e.g. continuity. 

    They have the calling signature f(*expanded_vars, *params), where:
        *expanded_vars: Array-likes representing the values of  
            linearly-independent quantities in the equations 
            (e.g. u, rho but also partial u / partial x or pressure P)

        *params: scalars required to define the equation - e.g. a variable viscosity. 

    These functions return an array-like of the same shape as the elements of expanded_vars, representing 
    the RHS of the relevant equation broadcast over any present axes (e.g. spatial)

    
*_solver_wrapped functions are meant to be passed to the differential equation solver. They encode a 
system of equations, e.g. the full Euler equations. 

    They have calling signature f(state_vars_stack, *fin_diff_funcs, *params). 
        
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

            Remark: For some stepper-wrapped functions, it will be highly natural to pass functions unrelated 
            to derivative taking - e.g. a function mapping the state variables to pressure. These should not 
            be passed with fin_diff_funcs, but instead params, below. 

        *params: Any additional scalar parameters or functions necessary for the equation, e.g. tunable viscosity
        as above, or a pressure equation of state.

            Remark: While *params may be present here, they must be defined in a closure before passing to the 
            solver, which will only accept functions of the form *_solver_wrapped(vars, *fin_diff_funcs). By convention, 
            any functions passed here should broadcast over all axes of state_vars besides the first.

    These functions return a 2-dimensional ND array of shape (k, N), representing the RHS of the system of equations, 
    broadcast over state variables and spatial axis (but not time). 
"""

#*_EQUATION()

#GENERIC

def continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x):
    return -1.0 * (u * rho_pderiv_x + rho * u_pderiv_x)

#INVISCID EQUATIONS

def momentum_euler_generic_equation(rho, u, u_pderiv_x, P_pderiv_x):
    return -1.0 * (u * u_pderiv_x + (1.0 / rho) * P_pderiv_x)


def momentum_euler_isentropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, c):
    return -1.0 * (u * u_pderiv_x + (1.0 / rho) * np.square(c) * rho_pderiv_x)

def momentum_euler_isentropic_polytropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, gamma):
    return -1.0 * (u * u_pderiv_x + np.power(rho, gamma - 2) * rho_pderiv_x)

#VISCOUS EQUATIONS

#NOTE: Generically, there are two relevant viscosities in the hydrodynamic equations. However, for 1D flow, 
#only a certain linear combination of them is relevant: 4/3 eta + zeta. We will then condense these into one 
#quantity, which we call eta but is actually eta + 3/4 zeta. 

#CONSTANT VISCOSITY

def _const_viscosity_term(rho, u_pderiv_xx, eta):
    return 1.0 / (rho) * (4/3 * eta) * u_pderiv_xx

def momentum_euler_const_viscosity_generic_equation(rho, u, u_pderiv_x, u_pderiv_xx, P_pderiv_x, eta):
    return momentum_euler_generic_equation(rho, u, u_pderiv_x, P_pderiv_x) + _const_viscosity_term(rho, u_pderiv_xx, eta)

#Contradiction ahoy - we ignore the entropic effects of viscosity and only model the momentum damping
def momentum_euler_const_viscosity_isentropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, c, eta):
    return momentum_euler_isentropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, c) + _const_viscosity_term(rho, u_pderiv_xx, eta)


def momentum_euler_const_viscosity_isentropic_polytropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, gamma, eta):
    return (momentum_euler_isentropic_polytropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, gamma) + 
            _const_viscosity_term(rho, u_pderiv_xx, eta))


#VARIABLE VISCOSITY 

#Here, no longer assume that the viscosity is constant. This makes the form of the viscous term more annoying.
#We manually expand the derivative partial_x (4/3 eta partial_x u) = 4/3 (partial_x eta partial_x u + eta partial_xx u)

def _var_viscosity_term(rho, u_pderiv_x, u_pderiv_xx, eta, eta_pderiv_x):
    return 4 / (3 * rho) * (u_pderiv_x * eta_pderiv_x + eta * u_pderiv_xx)

def momentum_euler_var_viscosity_generic_equation(rho, u, u_pderiv_x, u_pderiv_xx, eta, eta_pderiv_x, P_pderiv_x):
    return momentum_euler_generic_equation(rho, u, u_pderiv_x, P_pderiv_x) + _var_viscosity_term(rho, u_pderiv_x, u_pderiv_xx, 
                                                                                    eta, eta_pderiv_x)

def momentum_euler_var_viscosity_isentropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, eta, eta_pderiv_x, c):
    return momentum_euler_isentropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, c) + _var_viscosity_term(rho, u_pderiv_x, u_pderiv_xx, eta, 
                                                                                                         eta_pderiv_x)

def momentum_euler_var_viscosity_isentropic_polytropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, eta, eta_pderiv_x, gamma): 
    return momentum_euler_isentropic_polytropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, gamma) + _var_viscosity_term(
        rho, u_pderiv_x, u_pderiv_xx, eta, eta_pderiv_x)


#ENTROPY
#We include a single equation, giving the evolution of the entropy

def entropy_generic_equation(rho, u, s_pderiv_x, u_pderiv_x, eta, kappa, kappa_pderiv_x, T, T_pderiv_x, T_pderiv_xx):
    advective_term = -u * s_pderiv_x 
    dissipative_part_visc = 8/9 * eta * np.square(u_pderiv_x) 
    dissipative_part_therm = kappa * T_pderiv_xx + kappa_pderiv_x * T_pderiv_x
    dissipative_term = 1.0 / (rho * T) * (dissipative_part_visc + dissipative_part_therm)
    return advective_term + dissipative_term


#PURELY DIFFUSIVE DYNAMICS
#Included mostly for testing purposes...

def diffusive_equation(A_pderiv_xx, diffusivity):
    return diffusivity * A_pderiv_xx


#STEPPER_WRAPPED 

#INVISCID

def euler_equations_generic_solver_wrapped(state_vars_stack, eval, pderiv_x, pressure_func):
    rho_vals, u_vals = state_vars_stack
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals)

    pressures = pressure_func(state_vars_stack) 
    P_pderiv_x = pderiv_x(pressures) 

    rho_rhs = continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x) 
    u_rhs = momentum_euler_generic_equation(rho, u, u_pderiv_x, P_pderiv_x)

    return np.stack((rho_rhs, u_rhs))

def euler_equations_isentropic_solver_wrapped(state_vars_stack, eval, pderiv_x, c_func):
    rho_vals, u_vals = state_vars_stack
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals)

    c_vals = c_func(state_vars_stack) 
    c = eval(c_vals) 

    rho_rhs = continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x)
    u_rhs = momentum_euler_isentropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, c)

    return np.stack((rho_rhs, u_rhs))


def euler_equations_isentropic_polytropic_solver_wrapped(state_vars_stack, eval, pderiv_x, gamma): 
    rho_vals, u_vals = state_vars_stack
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals)

    rho_rhs = continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x)
    u_rhs = momentum_euler_isentropic_polytropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, gamma)

    return np.stack((rho_rhs, u_rhs))


#VISCOUS EULER 

#Const viscosity

def euler_equations_const_viscosity_generic_solver_wrapped(state_vars_stack, eval, pderiv_x, pderiv_xx, pressure_func, 
                                           eta):
    rho_vals, u_vals = state_vars_stack
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals)
    u_pderiv_xx = pderiv_xx(u_vals)

    pressures = pressure_func(state_vars_stack) 
    P_pderiv_x = pderiv_x(pressures) 

    rho_rhs = continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x) 
    u_rhs = momentum_euler_const_viscosity_generic_equation(rho, u, u_pderiv_x, u_pderiv_xx, 
                                                            P_pderiv_x, eta)

    return np.stack((rho_rhs, u_rhs))


def euler_equations_const_viscosity_isentropic_solver_wrapped(state_vars_stack, eval, pderiv_x, pderiv_xx,
                                                              c_func, eta):
    rho_vals, u_vals = state_vars_stack
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals)
    u_pderiv_xx = pderiv_xx(u_vals)

    c_vals = c_func(state_vars_stack)
    c = eval(c_vals)

    rho_rhs = continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x)
    u_rhs = momentum_euler_const_viscosity_isentropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, 
                                                               u_pderiv_xx, c, eta)

    return np.stack((rho_rhs, u_rhs))


def euler_equations_const_viscosity_isentropic_polytropic_solver_wrapped(state_vars_stack, eval, 
                                                            pderiv_x, pderiv_xx, gamma, eta):
    rho_vals, u_vals = state_vars_stack
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals)
    u_pderiv_xx = pderiv_xx(u_vals)

    rho_rhs = continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x)
    u_rhs = momentum_euler_const_viscosity_isentropic_polytropic_equation(rho, u, rho_pderiv_x, 
                                                    u_pderiv_x, u_pderiv_xx, gamma, eta)

    return np.stack((rho_rhs, u_rhs))
    

#Variable viscosity

def euler_equations_var_viscosity_generic_solver_wrapped(state_vars_stack, eval, pderiv_x, pderiv_xx, pressure_func, 
                                           eta_func):
    rho_vals, u_vals = state_vars_stack
    rho = eval(rho_vals)
    u = eval(u_vals)
    rho_pderiv_x = pderiv_x(rho_vals)
    u_pderiv_x = pderiv_x(u_vals)
    u_pderiv_xx = pderiv_xx(u_vals)

    eta_vals = eta_func(state_vars_stack)
    eta = eval(eta_vals)
    eta_pderiv_x = pderiv_x(eta_vals)

    pressures = pressure_func(state_vars_stack)
    P_pderiv_x = pderiv_x(pressures)

    rho_rhs = continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x)
    u_rhs = momentum_euler_var_viscosity_generic_equation(rho, u, u_pderiv_x, u_pderiv_xx, eta, 
                                                          eta_pderiv_x, P_pderiv_x)

    return np.stack((rho_rhs, u_rhs))


def euler_equations_var_viscosity_isentropic_solver_wrapped(state_vars_stack, eval, pderiv_x, pderiv_xx,
                                                              c_func, eta_func):
    rho_vals, u_vals = state_vars_stack
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals)
    u_pderiv_xx = pderiv_xx(u_vals)

    eta_vals = eta_func(state_vars_stack)
    eta = eval(eta_vals)
    eta_pderiv_x = pderiv_x(eta_vals)

    c_vals = c_func(state_vars_stack)
    c = eval(c_vals)

    rho_rhs = continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x)
    u_rhs = momentum_euler_var_viscosity_isentropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, 
                                                             eta, eta_pderiv_x, c)

    return np.stack((rho_rhs, u_rhs))

def euler_equations_var_viscosity_isentropic_polytropic_solver_wrapped(state_vars_stack, eval, 
                                                                       pderiv_x, pderiv_xx, gamma, eta_func):
    
    rho_vals, u_vals = state_vars_stack 
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals) 
    u_pderiv_xx = pderiv_xx(u_vals)

    eta_vals = eta_func(state_vars_stack) 
    eta = eval(eta_vals)
    eta_pderiv_x = pderiv_x(eta_vals)

    rho_rhs = continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x)
    u_rhs = momentum_euler_var_viscosity_isentropic_polytropic_equation(rho, u, rho_pderiv_x, u_pderiv_x, u_pderiv_xx, 
                                                                        eta, eta_pderiv_x, gamma)
    
    return np.stack((rho_rhs, u_rhs))

#Full Navier Stokes 
def navier_stokes_equations_generic_solver_wrapped(state_vars_stack, eval, pderiv_x, pderiv_xx, pressure_func, temperature_func, 
                                                   eta_func, kappa_func):
    rho_vals, u_vals, s_vals = state_vars_stack 
    rho = eval(rho_vals) 
    u = eval(u_vals) 
    s = eval(s_vals) 

    rho_pderiv_x = pderiv_x(rho_vals) 
    u_pderiv_x = pderiv_x(u_vals) 
    u_pderiv_xx = pderiv_xx(u_vals) 
    s_pderiv_x = pderiv_x(s_vals)

    pressure_vals = pressure_func(state_vars_stack)
    P_pderiv_x = pderiv_x(pressure_vals)

    temperature_vals = temperature_func(state_vars_stack)
    T = eval(temperature_vals) 
    T_pderiv_x = pderiv_x(temperature_vals) 
    T_pderiv_xx = pderiv_xx(temperature_vals)

    eta_vals = eta_func(state_vars_stack) 
    eta = eval(eta_vals) 
    eta_pderiv_x = pderiv_x(eta_vals) 

    kappa_vals = kappa_func(state_vars_stack) 
    kappa = eval(kappa_vals) 
    kappa_pderiv_x = pderiv_x(kappa_vals) 

    rho_rhs = continuity_equation(rho, u, rho_pderiv_x, u_pderiv_x)
    u_rhs = momentum_euler_var_viscosity_generic_equation(rho, u, u_pderiv_x, u_pderiv_xx, eta, 
                                                          eta_pderiv_x, P_pderiv_x)
    s_rhs = entropy_generic_equation(rho, u, s_pderiv_x, u_pderiv_x, eta, kappa, kappa_pderiv_x, T, 
                                     T_pderiv_x, T_pderiv_xx)
    
    return np.stack((rho_rhs, u_rhs, s_rhs))