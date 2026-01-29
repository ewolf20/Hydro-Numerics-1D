import numpy as np 

UNITARY_GAMMA = 5/3

#t, x are assumed in units such that x/t is in units of c_0; returned 
#density is in units of rho_0
def polytropic_riemann_expansion_density_profile(t, x, gamma = UNITARY_GAMMA): 
    xi = x / t
    return np.power(
        np.clip(
            1 - (gamma - 1)/(gamma + 1) * (1 + xi), 
            0, 
            1
        ), 2 / (gamma - 1)
    )

#As above, t, x assumed such that x/t is in units of c_0
#Returned velocity is in units of c_0
def polytropic_riemann_expansion_velocity_profile(t, x, gamma = UNITARY_GAMMA):
    xi = x / t 
    max_velocity = 2 / (gamma - 1) 
    min_position = -1 
    max_position = 2 / (gamma - 1)
    min_velocity = 0
    return np.where(
        x <= min_position, 
        min_velocity, 
        np.where(
            x >= max_position, 
            max_velocity, 
            2 / (gamma + 1) * (1 + xi)
        )
    )

#Rho is in units of rho_0; returned pressure is in units of rho_0 c_0^2
def normalized_pressure_isentropic_polytropic_eos(rho, gamma = UNITARY_GAMMA):
    return 1.0 / gamma * np.power(rho, gamma)

#Rho is in units of rho_0; returned speed is in units of c_0
def normalized_speed_of_sound_isentropic_polytropic_eos(rho, gamma = UNITARY_GAMMA):
    return np.power(rho, (gamma - 1) / 2.0)

#Rho is in units of rho_0; returned energy per mass is in units of c_0^2
def normalized_energy_per_mass_isentropic_polytropic_eos(rho, gamma = UNITARY_GAMMA):
    return 1.0 / (gamma * (gamma - 1)) * np.power(rho, gamma - 1)