import numpy as np 


def solve_equations(wrapped_equation, initial_state, x_diff, t_diff, t_steps, method = "forward_euler", 
                    deriv_order = 1, output_increment = 10):
    method_time_order, method_num_funcs, stepper = _handle_method(method, x_diff)
    equation_num_funcs = method_num_funcs[:deriv_order + 1]

    output_state_list = []
    output_time_list = []
    output_state_list.append(initial_state) 
    output_time_list.append(0.0)

    #Massage initial_state into the form required by the stepper 
    #Initial state should be a 2D array; insert a time axis in position 1
    target_shape = (initial_state.shape[0], method_time_order, initial_state.shape[1])
    initial_state_dim_expanded = np.expand_dims(initial_state, axis = 1)
    initial_state_reshaped = np.broadcast_to(initial_state_dim_expanded, target_shape)

    current_state_vars_stack = initial_state_reshaped 

    for i in range(t_steps): 
        state_update = stepper(wrapped_equation, current_state_vars_stack, equation_num_funcs, t_diff)

        if i % output_increment == 0:
            output_state_list.append(state_update)
            output_time_list.append(i * t_diff)
        
        current_state_vars_stack[:, 1:] = current_state_vars_stack[:, :-1] 
        current_state_vars_stack[:, 0] = state_update

    output_state_array = np.array(output_state_list) 
    output_time_array = np.array(output_time_list) 

    #Reshape to standard form of state variable index first 
    output_state_array = np.moveaxis(output_state_array, 1, 0) 
    
    return (output_time_array, output_state_array)


#Handle method
def _handle_method(method, x_diff):
    if method == "forward_euler":
        time_order = 1 
        num_funcs = [_eval_forward_euler, _pderiv_x_forward_euler_factory(x_diff), 
                     _pderiv_xx_forward_euler_factory(x_diff)] 
        stepper = _stepper_forward_euler
    else:
        raise ValueError("Allowed methods are: 'forward_euler'")
    
    return (time_order, num_funcs, stepper)


#Forward Euler integration
def _eval_forward_euler(state_var):
    return state_var[0] 

def _pderiv_x_forward_euler_factory(x_diff): 
    def _pderiv_x_forward_euler(state_var):
        current_var = state_var[0]
        pderiv_x = np.gradient(current_var, x_diff, edge_order = 2)
        return pderiv_x
    return _pderiv_x_forward_euler


def _pderiv_xx_forward_euler_factory(x_diff):
    def _pderiv_xx_forward_euler(state_var):
        current_var = state_var[0]
        pderiv_x = np.gradient(current_var, x_diff, edge_order = 2)
        pderiv_xx = np.gradient(pderiv_x, x_diff, edge_order = 2) 
        return pderiv_xx
    return _pderiv_xx_forward_euler


def _stepper_forward_euler(wrapped_equation, state_vars_stack, num_funcs, t_diff):
    rhs = wrapped_equation(state_vars_stack, *num_funcs) 
    new_state = state_vars_stack[:, 0] + rhs * t_diff
    return new_state
