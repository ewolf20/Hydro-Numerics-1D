import os 
import sys

import numpy as np 


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo = path_to_file + "/../../"
sys.path.insert(0, path_to_repo)

from Hydro_Numerics_1D.code import analytic_functions 


def test_polytropic_riemann_expansion_density_profile(): 
    #Start by checking that the result is correct for unitarity 
    sample_t = 1.0 
    sample_x_range = np.linspace(-5, 5, 10000) 
    predicted_profile = np.clip(
        np.power((3 - sample_x_range) / 4, 3), 
        0, 
        1
    )
    scale_invariant_gamma = 5/3
    returned_profile = analytic_functions.polytropic_riemann_expansion_density_profile(sample_t, sample_x_range, 
                                                                                       gamma = scale_invariant_gamma)
    assert np.allclose(predicted_profile, returned_profile)

    #Then check that the result at x=0 is consistent with expectation 
    gamma_range = np.linspace(1.2, 2.0, 1000) 
    expected_zero_values = np.power(1 - (gamma_range - 1) / (gamma_range + 1), 2 / (gamma_range - 1))

    returned_zero_values = analytic_functions.polytropic_riemann_expansion_density_profile(sample_t, 0, gamma = gamma_range)  
    assert np.allclose(expected_zero_values, returned_zero_values)


def test_polytropic_riemann_expansion_velocity_profile():
    #As before, check that unitarity makes sense 
    sample_t = 1.0 
    sample_x_range = np.linspace(-5, 5, 10000) 
    scale_invariant_gamma = 5/3 
    predicted_max_velocity = 3 
    predicted_profile = np.clip(
        2.0 / (scale_invariant_gamma + 1) * (1 + sample_x_range), 
        0, 
        predicted_max_velocity
    )
    returned_profile = analytic_functions.polytropic_riemann_expansion_velocity_profile(sample_t, sample_x_range, 
                                                                                        gamma = scale_invariant_gamma)
    
    assert np.allclose(predicted_profile, returned_profile)

    #Now check the prediction at x=0... 

    gamma_range = np.linspace(1.2, 2.0, 1000) 
    predicted_zero_values = 2.0 / (gamma_range + 1) 
    returned_zero_values = analytic_functions.polytropic_riemann_expansion_velocity_profile(sample_t, 0, gamma = gamma_range)
    assert np.allclose(predicted_zero_values, returned_zero_values)

#Just verify that we've written it down properly...
def test_normalized_pressure_isentropic_polytropic_eos():
    sample_gamma = 1.5 
    sample_rho = 3
    expected_pressure = 1 / sample_gamma * np.power(sample_rho, sample_gamma)
    returned_pressure = analytic_functions.normalized_pressure_isentropic_polytropic_eos(sample_rho, gamma = sample_gamma)
    assert np.isclose(expected_pressure, returned_pressure)


#Likewise, verify correctly written only
def test_normalized_speed_of_sound_isentropic_polytropic_eos():
    sample_rho = 3.0 
    sample_gamma = 1.5 
    expected_sos = np.power(sample_rho, (sample_gamma - 1) / 2)
    returned_sos = analytic_functions.normalized_speed_of_sound_isentropic_polytropic_eos(sample_rho, gamma = sample_gamma)
    assert np.isclose(expected_sos, returned_sos)

def test_normalized_energy_per_mass_isentropic_polytropic_eos():
    sample_gamma = 1.5 
    sample_rho = 3 
    expected_energy_per_mass = 1.0 / (sample_gamma * (sample_gamma - 1)) * np.power(sample_rho, sample_gamma - 1)
    returned_energy_per_mass = analytic_functions.normalized_energy_per_mass_isentropic_polytropic_eos(sample_rho, gamma = sample_gamma)
    assert np.isclose(expected_energy_per_mass, returned_energy_per_mass)
    