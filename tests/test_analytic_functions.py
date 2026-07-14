import os 
import sys

import numpy as np 
import matplotlib.pyplot as plt


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo = path_to_file + "/../../"
sys.path.insert(0, path_to_repo)

from Hydro_Numerics_1D.code import analytic_functions 

SCALE_INVARIANT_GAMMA = 5/3


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


def test_polytropic_piston_velocity_profile():
    #First check that the velocity at the piston is correct at all times 
    long_time_range = np.linspace(0, 0.5, 1000) 
    piston_face_velocities_long_time = analytic_functions.polytropic_piston_velocity_profile(long_time_range, 0, gamma = SCALE_INVARIANT_GAMMA)
    expected_piston_face_velocities_long_time = long_time_range 

    assert np.allclose(piston_face_velocities_long_time, expected_piston_face_velocities_long_time, rtol = 1e-5, atol = 1e-7)

    #Then check that the velocities get launched properly along a sample characteristic 
    SAMPLE_LAUNCH_TIME = 0.1 
    piston_face_velocity_launch_time = SAMPLE_LAUNCH_TIME
    expected_velocity_along_characteristic = piston_face_velocity_launch_time
    piston_face_position_free_falling_sample_launch_time = 0.5 * np.square(SAMPLE_LAUNCH_TIME)
    CHARACTERISTIC_EVOLUTION_TIME = 0.1 

    expected_characteristic_speed = (SCALE_INVARIANT_GAMMA + 1) / 2.0 * piston_face_velocity_launch_time + 1.0 

    characteristic_evolution_time_range = np.linspace(0, CHARACTERISTIC_EVOLUTION_TIME, 100)
    characteristic_positions_free_falling = (piston_face_position_free_falling_sample_launch_time +
                                         expected_characteristic_speed * characteristic_evolution_time_range)
    
    characteristic_times = SAMPLE_LAUNCH_TIME + characteristic_evolution_time_range
    characteristic_positions_original_frame = characteristic_positions_free_falling - 0.5 * np.square(characteristic_times)

    velocities_along_characteristic = analytic_functions.polytropic_piston_velocity_profile(characteristic_times, characteristic_positions_original_frame)
    assert np.allclose(velocities_along_characteristic, expected_velocity_along_characteristic)


def test_polytropic_piston_density_profile():
    #Restrict focus to places before shock has formed
    x_range = np.linspace(0, 1, 100) 
    time_range = np.linspace(0, 0.5, 100)

    time_grid, x_grid = np.meshgrid(time_range, x_range, indexing = "ij")

    sample_rho_values = analytic_functions.polytropic_piston_density_profile(time_grid, x_grid, gamma = SCALE_INVARIANT_GAMMA)

    #We've already tested the velocities, so trust them
    sample_velocities = analytic_functions.polytropic_piston_velocity_profile(time_grid, x_grid, gamma = SCALE_INVARIANT_GAMMA)
    #Use the known result from riemann invariants 
    sample_c_values = 1.0 + (SCALE_INVARIANT_GAMMA - 1.0) / 2.0 * sample_velocities 
    #Convert the speeds of sound to densities
    expected_rho_values = np.power(sample_c_values, 2.0 / (SCALE_INVARIANT_GAMMA - 1.0))

    assert np.allclose(sample_rho_values, expected_rho_values)


def test_polytropic_piston_face_density():
    #Analytically calculated value
    SAMPLE_TIME = 0.75
    EXPECTED_PISTON_FACE_DENSITY = np.power(5/4, 3) 
    sample_piston_face_density = analytic_functions.polytropic_piston_face_density(SAMPLE_TIME, gamma = SCALE_INVARIANT_GAMMA)
    assert np.isclose(sample_piston_face_density, EXPECTED_PISTON_FACE_DENSITY)


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
    