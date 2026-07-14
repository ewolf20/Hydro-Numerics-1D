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
    min_velocity = 0
    return np.clip(
        2 / (gamma + 1) * (1 + xi), 
        min_velocity, 
        max_velocity
    )

"""
Analytic profile for an inviscid gas accelerated with acceleration -a into a stationary wall at x = 0, 
#where the background speeed of sound is c_0 and the equation of state is polytropic with exponent gamma.

X, t, and v are assumed normalized to the appropriate combination of c_0 and a. 

WARNING: The formulas below _implicitly_ assume that the shockwave has not yet formed; if this condition is violated, 
they may fail

"""

def _validate_piston_t(t, gamma):
    shock_formation_time = 2.0 / (gamma + 1) 
    if np.any(t > shock_formation_time):
        raise ValueError("Specified t is greater than shock formation time; analytic formulas are invalid")

def polytropic_piston_velocity_profile(t, x, gamma = UNITARY_GAMMA):
    _validate_piston_t(t, gamma)

    #Analytic formulas are simplest in the free-falling frame, where the piston is accelerating
    x_free_fall = x + 0.5 * np.square(t)
    return np.where(
        x_free_fall < t, 
        _v_left(t, x_free_fall, gamma), 
        0
    )

def polytropic_piston_density_profile(t, x, gamma = UNITARY_GAMMA): 
    _validate_piston_t(t, gamma)

    x_free_fall = x + 0.5 * np.square(t) 
    return np.where(
        x_free_fall < t, 
        _rho_left(t, x_free_fall, gamma), 
        1
    )

#Formula for the density at the piston face. Valid for some period of time after shock formation, until the "news"
#of shock formation reaches the piston face. 
#WARNING: no checks are made to see if this condition is fulfilled!
def polytropic_piston_face_density(t, gamma = UNITARY_GAMMA):
    piston_face_velocity = t 
    piston_face_c = _v_to_c_func(piston_face_velocity, gamma)
    piston_face_rho = _c_to_rho_func(piston_face_c, gamma) 
    return piston_face_rho


def _v_left(t, x, gamma):
    neg_b = (gamma + 1) / 2.0 * t - 1.0
    v = (1.0 / gamma) * (neg_b + np.sqrt(np.square(neg_b) - 4 * (gamma / (2)) * (x - t)))
    return v

def _v_to_c_func(v, gamma):
    return 1 + (gamma - 1) / 2.0 * v

def _c_left(t, x, gamma): 
    v = _v_left(t, x, gamma)
    c = _v_to_c_func(v, gamma)
    return c
    
def _c_to_rho_func(c, gamma):
    return np.power(c, 2 / (gamma - 1))

def _rho_left(t, x, gamma):
    c = _c_left(t, x, gamma)
    #We've also normalized rho to rho_0 - this is how it works out
    rho = _c_to_rho_func(c, gamma)
    return rho






#Rho is in units of rho_0; returned pressure is in units of rho_0 c_0^2
def normalized_pressure_isentropic_polytropic_eos(rho, gamma = UNITARY_GAMMA):
    return 1.0 / gamma * np.power(rho, gamma)

#Rho is in units of rho_0; returned speed is in units of c_0
def normalized_speed_of_sound_isentropic_polytropic_eos(rho, gamma = UNITARY_GAMMA):
    return np.power(rho, (gamma - 1) / 2.0)

#Rho is in units of rho_0; returned energy per mass is in units of c_0^2
def normalized_energy_per_mass_isentropic_polytropic_eos(rho, gamma = UNITARY_GAMMA):
    return 1.0 / (gamma * (gamma - 1)) * np.power(rho, gamma - 1)



#The time-evolution equation for a unity-area Gaussian, initially of width sigma_0, 
#evolving under purely diffusive dynamics with diffusivity D.
#Mostly for testing purposes.
def diffusive_gaussian(t, x, sigma_0, D):
    sigma_prime = np.sqrt(np.square(sigma_0) + 2 * D * t)
    return 1.0 / (sigma_prime * np.sqrt(2 * np.pi)) * np.exp(-np.square(x) / (2.0 * np.square(sigma_prime)))

