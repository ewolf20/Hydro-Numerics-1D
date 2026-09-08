import time

import numpy as np 

"""
General purpose equation solver for 1+1D partial differential equations. 

Given a set of equations wrapped_equation, evolve a given set of 1+1D partial differential equations
 using a user-specified method.

Parameters: 

    wrapped_equation: A function encoding the derivative system to be solved. Call signature must be 
        wrapped_equation(state_vars_stack, *fin_diff_funcs), with state_vars_stack a (k, l, N) array 
        as documented in hydrodynamic_equations.py, and with *fin_diff_funcs a set of functions with 
        signature fin_diff_func(state_var), where state_var is (l, N), also as documented. fin_diff_funcs 
        encode the finite difference scheme used to evalutate the function and its partial derivatives, 
        and must be in the order eval, pderiv_x, pderiv_xx, ... 

        If explicit_eq is True, call signature for wrapped_equation is wrapped_equation(t, x, state_vars_stack, 
            *fin_diff_funcs), where t is a scalar and x is an array of shape (N,)

    initial_state: A (k, N) array encoding the initial state of the system at t=0 for all x-points. 

    x_diff: The evolution assumes evenly spaced x points with spacing x_diff 

    t_diff: The time spacing used by the method. Some methods use higher-order steps in time - the 
        convention for t_diff is that t_diff * t_steps is equal to the total evolution time. 

    t_steps: The number of time steps used by the solver. 

    deriv_order: The order of partial derivatives appearing in wrapped-equation; e.g. if only pderiv_x 
    and eval appear, then deriv_order = 1

    explicit_eq: If True, it is assumed that the RHS of the derivative system contains an explicit dependence 
        on x and/or t

    t_init: (float) The initial time of the simulation. Default 0.0. Primarily used for explicit equations. 

    x_left: (float) The initial x-value of the leftmost simulation point. Default 0.0. Ditto above.

    boundary_func: If not None, impose a boundary condition on the solver by applying the function 
    boundary_func(state_vars) at the endpoints. The call signature of boundary_func should be to take in 
    and return a shape (k,) array, representing the values of the individual state vars at the endpoints. 
    The function uses the values of the state vars returned by the numerical scheme and imposes the boundary 
    condition on the returned state vars. For instance, the boundary condition u = 0 is imposed as: 
        f([rho, u, s]) = [rho, 0, s]

    boundary_func_left, '_right: If these are not None, they specify individual boundary functions that 
        are applied only to the left and right boundaries, respectively. If either of these is None,
        they default to boundary_func. 

    output_increment: The number of steps to increment between output states + times of the system. If 
    None, output_increment will be chosen so that approximately 100 steps are output. 

    print_progress: If true, the solver will issue print statements meant to estimate its total runtime

    guarantee_last: If True, guarantee that the last simulated time and state of the system are recorded 
    in times and states, even if t_steps % output_increment != 0.

    check_finite: If True, the solver will check at each stage whether the system state is finite 
    (i.e. not np.inf or np.nan); if this condition fails, the solver will abort and return the system evolution 
    up to the last finite state.

    incremental_output: If True, the solver will save its output values incrementally to a specified .npy file, 
    in addition to returning them at termination. Carries a small time cost, but good if interruption is expected. 

    resume_existing: If True and both time_numpy_path and state_var_numpy_path are not None, resume the simulation 
    using the state saved in those locations. 

        NOTE: This is not a completely clean resumption, instead acting as if one had fed the final state from state_vars_numpy_path 
        in as an initial state, then evolved the system for a time equal to the difference between the last time from 
        time_numpy_path and the imputed target runtime. 

    time_numpy_path, state_var_numpy_path: Default None. Used in conjunction with incremental_output and resume_existing, per above.

    Returns:

    If check_finite, a tuple (success, times, states). Success is a boolean representing whether the system 
    diverged at any point. Times is a length L 1D array containing evolution times, and states is a (k, L, N) 
    array of the system state at all the corresponding times. If check_finite is false, only (times, states) is 
    returned.
"""
def solve_equations(wrapped_equation, initial_state, x_diff, t_diff, t_steps, method = "forward_euler", 
                    deriv_order = 1, explicit_eq = False, t_init = 0.0, x_left = 0.0, boundary_func = None, 
                    boundary_func_left = None, boundary_func_right = None, explicit_boundary = False,
                    output_increment = None, print_progress = False, guarantee_last = False, check_finite = True, 
                    time_numpy_path = None, state_var_numpy_path = None, incremental_output = False, 
                    resume_existing = False):

    #Function for updating the output arrays - no returns as it mutates the underlying lists
    def update_output(t, state_update, output_time_list, output_state_list): 
        output_state_list.append(state_update) 
        output_time_list.append(t)
        if incremental_output and not (time_numpy_path is None or state_var_numpy_path is None):
            output_state_array = np.array(output_state_list) 
            #Reshape to standard form of state variable index first 
            output_state_array = np.moveaxis(output_state_array, 1, 0)
            np.save(state_var_numpy_path, output_state_array)
            output_time_array = np.array(output_time_list)
            np.save(time_numpy_path, output_time_array)


    method_time_order, method_fin_diff_funcs, stepper = _handle_method(method, x_diff, explicit_eq)
    equation_fin_diff_funcs = method_fin_diff_funcs[:deriv_order + 1]

    if boundary_func_left is None: 
        boundary_func_left = boundary_func 
    if boundary_func_right is None: 
        boundary_func_right = boundary_func

    if output_increment is None:
        output_increment = t_steps // 100

    output_state_list = []
    output_time_list = []

    #If resume_existing is true, hijack the initial state and inject the last state from the existing results...
    if resume_existing and not (time_numpy_path is None or state_var_numpy_path is None):
        #Load existing times 
        existing_times = np.load(time_numpy_path)
        existing_state_vars = np.load(state_var_numpy_path)

        #Fix convention of 
        existing_state_vars_rearranged = np.moveaxis(existing_state_vars, 1, 0)
        output_state_list = list(existing_state_vars_rearranged)
        output_time_list = list(existing_times)

        last_existing_time = existing_times[-1] 
        #Time axis is second, by convention
        last_existing_state_var = existing_state_vars[:, -1] 
        initial_state = last_existing_state_var 
        t = last_existing_time 
        last_existing_time_num_steps = int(np.round(last_existing_time / t_diff))
        t_steps = t_steps - last_existing_time_num_steps
    else:
        t = t_init 
        update_output(t_init, initial_state, output_time_list, output_state_list)

    #Massage initial_state into the form required by the stepper 
    #Initial state should be a 2D array; insert a time axis in position 1
    initial_state_dim_expanded = np.expand_dims(initial_state, axis = 1)
    initial_state_reshaped = np.repeat(initial_state_dim_expanded, method_time_order, axis = 1)

    current_state_vars_stack = initial_state_reshaped

    #Define an array of x-positions for an explicit RHS
    if explicit_eq:
        num_x_samps = initial_state.shape[-1]
        x_vals = x_left + x_diff * np.arange(num_x_samps)


    PRINT_PROGRESS_COMPLETION_FRACTION = 0.01
    print_progress_index = int(np.round(t_steps * PRINT_PROGRESS_COMPLETION_FRACTION))
    if print_progress:
        print("Running solver:")
        print("Time step: {0:.2e}".format(t_diff)) 
        print("Number steps: {0:.0f}".format(t_steps))
        tick = time.time()

    success = True
    for i in range(t_steps):
        t += t_diff
        if not explicit_eq:
            state_update = stepper(wrapped_equation, current_state_vars_stack, equation_fin_diff_funcs, t_diff)
        else:
            #Pass a time stack to accommodate higher-order temporal methods
            t_stack = t - t_diff * np.arange(method_time_order)
            #Reshape to broadcast against x correctly
            t_stack = np.expand_dims(t_stack, axis = -1)
            state_update = stepper(t_stack, x_vals, wrapped_equation, current_state_vars_stack, equation_fin_diff_funcs, t_diff)

        #Impose boundary functions at x endpoints
        if not boundary_func_left is None:
            if explicit_boundary:
                state_update[:, 0] = boundary_func_left(t, state_update[:, 0])
            else:
                state_update[:, 0] = boundary_func_left(state_update[:, 0])

        if not boundary_func_right is None:
            if explicit_boundary:
                state_update[:, -1] = boundary_func_right(t, state_update[:, -1])
            else:
                state_update[:, -1] = boundary_func_right(state_update[:, -1])

        if check_finite and not np.all(np.isfinite(state_update)):
            #If an infinity happened, return the last non-infinite step we have 
            last_valid_time = t - t_diff
            last_valid_state = current_state_vars_stack[:, 0]
            update_output(last_valid_time, last_valid_state, output_time_list, output_state_list)
            success = False 
            break

        if print_progress and i == print_progress_index:
            tock = time.time()
            elapsed = tock - tick
            estimated_time = elapsed / PRINT_PROGRESS_COMPLETION_FRACTION
            print("Estimated Completion Time: {0:.1f} s".format(estimated_time))

        #For correct 'fencepost' logic, we should use this...
        if (i + 1) % output_increment == 0:
            update_output(t, state_update, output_time_list, output_state_list)
            if print_progress:
                print("Completed: {0:.1f} %".format(100 * i / t_steps))

        current_state_vars_stack[:, 1:] = current_state_vars_stack[:, :-1] 
        current_state_vars_stack[:, 0] = state_update

    #If guarantee_last, ensure that the last state is included in the returns, and avoid duplicates 
    if guarantee_last and success and t_steps % output_increment != 0:
        #Use the fact that t and state_var are at their final values
        update_output(t, state_update, output_time_list, output_state_list)

    output_state_array = np.array(output_state_list)
    output_time_array = np.array(output_time_list)

    #Reshape to standard form of state variable index first 
    output_state_array = np.moveaxis(output_state_array, 1, 0) 
    
    if check_finite:
        return (success, output_time_array, output_state_array)
    else:
        return (output_time_array, output_state_array)


#Handle method
def _handle_method(method, x_diff, explicit_eq):
    if method == "forward_euler":
        time_order = 1 
        fin_diff_funcs = [_eval_fe, _pderiv_x_fe_factory(x_diff), 
                     _pderiv_xx_fe_factory(x_diff)] 
        if explicit_eq:
            stepper = _stepper_forward_euler_explicit
        else:
            stepper = _stepper_forward_euler
    elif method == "leapfrog":
        time_order = 2
        fin_diff_funcs = [_eval_leapfrog, _pderiv_x_leapfrog_factory(x_diff), 
                          _pderiv_xx_leapfrog_factory(x_diff)]
        if explicit_eq:
            stepper = _stepper_leapfrog_explicit
        else:
            stepper = _stepper_leapfrog
    else:
        raise ValueError("Allowed methods are: 'forward_euler', 'leapfrog'")
    
    return (time_order, fin_diff_funcs, stepper)

#Helper functions, so that we don't rewrite the same finite difference scheme 
#a million times 

#Implement pderiv_x with np.gradient
#Equivalent in the bulk to: 
#   partial_x u_j = 1.0 / (2 Delta x) * (u_{j + 1} - u_{j - 1})
def _pderiv_x_helper(current_var, x_diff):
    pderiv_x = np.gradient(current_var, x_diff, edge_order = 2)
    return pderiv_x

#Double invocation of np.gradient???
#Not very local, but maybe that's a good thing? Equivalent in the bulk to: 
#   partial_xx u_j = 1.0 / (4 Delta x^2) (u_{j + 2} - 2 u_{j} + u_{j - 2})
def _pderiv_xx_helper(current_var, x_diff):
        pderiv_x = np.gradient(current_var, x_diff, edge_order = 2) 
        pderiv_xx = np.gradient(pderiv_x, x_diff, edge_order = 2)
        return pderiv_xx

#Forward Euler methods
def _eval_fe(state_var):
    return state_var[0]

def _pderiv_x_fe_factory(x_diff): 
    def _pderiv_x_fe(state_var):
        current_var = state_var[0]
        return _pderiv_x_helper(current_var, x_diff)
    return _pderiv_x_fe


def _pderiv_xx_fe_factory(x_diff):
    def _pderiv_xx_fe(state_var):
        current_var = state_var[0]
        return _pderiv_xx_helper(current_var, x_diff)
    return _pderiv_xx_fe


def _stepper_forward_euler(wrapped_equation, state_vars_stack, num_funcs, t_diff):
    rhs = wrapped_equation(state_vars_stack, *num_funcs) 
    new_state = state_vars_stack[:, 0] + rhs * t_diff
    return new_state

def _stepper_forward_euler_explicit(t, x, wrapped_equation, state_vars_stack, num_funcs, t_diff):
    rhs = wrapped_equation(t, x, state_vars_stack, *num_funcs) 
    new_state = state_vars_stack[:, 0] + rhs * t_diff
    return new_state

#Leapfrog integration
#NOTE: We deliberately evaluate the zeroth and second derivatives at different time locations from the first.
#This attempts to treat the advective terms via leapfrog and the diffusive via (modified) forward euler
#Diffusive terms are unstable in the leapfrog method 

def _eval_leapfrog(state_var):
    return state_var[1]

def _pderiv_x_leapfrog_factory(x_diff):
    def _pderiv_x_leapfrog(state_var):
        current_var = state_var[0] 
        return _pderiv_x_helper(current_var, x_diff)
    return _pderiv_x_leapfrog

def _pderiv_xx_leapfrog_factory(x_diff):
    def _pderiv_xx_leapfrog(state_var):
        current_var = state_var[1] 
        return _pderiv_xx_helper(current_var, x_diff)
    return _pderiv_xx_leapfrog

def _stepper_leapfrog(wrapped_equation, state_vars_stack, num_funcs, t_diff):
    rhs = wrapped_equation(state_vars_stack, *num_funcs)
    new_state = state_vars_stack[:, 1] + rhs * 2 * t_diff
    return new_state

def _stepper_leapfrog_explicit(t, x, wrapped_equation, state_vars_stack, num_funcs, t_diff):
    rhs = wrapped_equation(t, x, state_vars_stack, *num_funcs)
    new_state = state_vars_stack[:, 1] + rhs * 2 * t_diff
    return new_state