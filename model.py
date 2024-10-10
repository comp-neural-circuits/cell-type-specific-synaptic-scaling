

import numpy as np
from numba import cuda, jit
import matplotlib.pyplot as plt
from util import *


@jit(nopython=True) # ensures that the function is compiled without using the Python interpreter ("nopython" mode). If Numba encounters any code that cannot be translated to machine code, it will raise an error.
def model(delta_t, sampling_rate, l_res_rates, l_res_weights, sim_duration, weights, g,
          g_stim, stim_times, taus, beta_K, rheobases,
          flags=(0, 0, 0, 0, 0, 0), flags_theta = (1,1)):
    
    ##### Initializing the setup
    (sampling_rate_stim, sampling_rate_sim) = sampling_rate
    (r_phase1, r_phase2, r_phase3, max_E) = l_res_rates
    (J_exc_phase1, J_phase2) = l_res_weights
    (w_EEii, w_EPii, w_ESii, w_PEii, w_PPii, w_PSii, w_SEii,
     w_EEij, w_EPij, w_ESij, w_PEij, w_PPij, w_PSij, w_SEij) = weights
    (g_E, g_P, g_S, g_top_down_to_S) = g
    g_S_total = g_S # total input to S is equal to g_S before the offset of the conditioning
    (g_stim_E, g_stim_P, g_stim_S) = g_stim
    (stim_start, stim_stop) = stim_times[0]
    (tau_E, tau_P, tau_S, tau_plas,
     tau_scaling_E, tau_scaling_P, tau_scaling_S,
     tau_theta, tau_beta) = taus
    (rheobase_E, rheobase_P, rheobase_S) = rheobases

    # Setting up initial conditions
    E01, E02, P01, P02, S01, S02 = 1,1,1,1,1,1 # The initial rates are arbitrarily set to 1
    EE110, EE120, EE210, EE220 = w_EEii, w_EEij, w_EEij, w_EEii
    EP110, EP120, EP210, EP220 = w_EPii, w_EPij, w_EPij, w_EPii
    ES110, ES120, ES210, ES220 = w_ESii, w_ESij, w_ESij, w_ESii
    E1, E2 = 0,0
    max_E[0] = 0
    stimulus_E1, stimulus_P1, stimulus_S1 = 0, 0, 0
    stimulus_E2, stimulus_P2, stimulus_S2 = 0, 0, 0

    learning_rate = 1
    r_baseline = 0
    theta1, theta2 = 1, 1
    beta1, beta2 = 1, 1

    # Flags of the plasticity mechanisms are initialized here
    hebbian_flag, three_factor_flag, adaptive_set_point_flag, E_scaling_flag, P_scaling_flag, S_scaling_flag = 0, 0, 0, 0, 0, 0
    flag_theta_shift, flag_theta_local = 0, 0

    # Counters and indices are initialized for different phases
    # (counter and i (index) to fill the arrays. np.mod doesn't work in numba,
    # thus we need counters to hold data at every "sampling_rate" step)
    phase1, phase3 = 0,0
    counter1, counter2 , counter3 = 0,0,0 # Counter to hold data with the respective sampling rate
    i_1, i_2, i_3 = 0,0,0 # Index to fill the data arrays

    stim_applied = 0  # The number of stimulation applied is held


    ##### The loop of the numerical iterations
    for step in range(sim_duration):


        ### If it is the start of the stimulation
        if step == int((stim_start + 2) * (1 / delta_t)):
            # If it is the first stimuli (conditioning)
            if stim_applied == 0:
                r_baseline = E1
                theta1, theta2 = r_baseline, r_baseline
                beta1, beta2 = r_baseline - beta_K, r_baseline - beta_K

                (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
                 E_scaling_flag, P_scaling_flag, S_scaling_flag) = flags
                (flag_theta_shift, flag_theta_local) = flags_theta

                # Hebbian learning is activated at conditioning onset
                if hebbian_flag:
                    learning_rate = 1

            if stim_applied == 1:  # If it is the second stimuli (testing)
                # Stop the data-holder counter by setting the counter2 to a high value
                counter2 = sampling_rate_sim + 5  # stop the data-holder counter

            # Stimulation of the selected cells for the respected stimuli is set
            stimulus_E1, stimulus_E2 = g_stim_E[stim_applied]
            stimulus_P1, stimulus_P2 = g_stim_P[stim_applied]
            stimulus_S1, stimulus_S2 = g_stim_S[stim_applied]

            # Increase the no stim applied
            stim_applied = stim_applied + 1


        ### If it is the end of the stimulation
        if step == int((stim_stop + 2)*(1/delta_t)):
            # The offset of the conditioning
            if stim_applied == 1:
                g_S_total = g_S + g_top_down_to_S # Add top-down input to S
                counter2 = sampling_rate_sim  # Start the data-holder counter

            # Hebbian learning is turned off due to the third factor
            if three_factor_flag:
                learning_rate = 0

            # All stimuli are turned off
            stimulus_E1, stimulus_E2 = 0, 0
            stimulus_P1, stimulus_P2 = 0, 0
            stimulus_S1, stimulus_S2 = 0, 0

            # Set the new timing for the next stim if exists
            if stim_times.shape[0] > stim_applied:
                (stim_start, stim_stop) = stim_times[stim_applied]

        # setting the counters for phase 1 and 3 with 5 seconds of
        if step == int(2*(1/delta_t)):
            counter1 = sampling_rate_stim  # Start the data-holder counter1
            phase1 = 1
        elif step == int((stim_times[0][1] + 5 + 2) * (1 / delta_t)):
            phase1 = 0

        elif step == int((stim_times[1][0] - 5 + 2) * (1 / delta_t)):
            counter3 = sampling_rate_stim  # Start the data-holder counter3
            phase3 = 1
        elif step == int((stim_times[1][1] + 5 + 2) * (1 / delta_t)):
            phase3 = 0


        ### Data is registered to the arrays
        if phase1 and counter1 == sampling_rate_stim:
            r_phase1[:,i_1] = [E01, E02, P01, P02, S01, S02]
            J_exc_phase1[:,i_1] = [EE110, EE120, EE210, EE220]

            i_1 = i_1 + 1
            counter1 = 0  # restart

        elif phase3 and counter3 == sampling_rate_stim:
            r_phase3[:,i_3] = [E01, E02, P01, P02, S01, S02]

            i_3 = i_3 + 1
            counter3 = 0  # restart

        if stim_applied == 1 and counter2 == sampling_rate_sim:
            r_phase2[:,i_2] = [E01, E02, P01, P02, S01, S02, theta1, theta2, beta1, beta2]
            J_phase2[:,i_2] = [EE110, EE120, EE210, EE220, EP110, EP120, EP210, EP220, ES110, ES120, ES210, ES220]

            i_2 = i_2 + 1
            counter2 = 0  # restart

        if E01 > max_E[0]:
            max_E[0] = E01

        # if the system explodes, stop the simulation
        if E01 > 1000:
            break

        if E01 == 0:
            break


        ### Calculating the firing rates at this timestep
        I1 = g_E - EP110 * P01 - EP120 * P02 - ES110 * S01 - ES120 * S02 + EE110 * E01 + EE120 * E02 + stimulus_E1
        I2 = g_E - EP210 * P01 - EP220 * P02 - ES210 * S01 - ES220 * S02 + EE210 * E01 + EE220 * E02 + stimulus_E2

        E1 = E01 + delta_t*(1/tau_E)*(-E01 + np.maximum(0,I1 - rheobase_E))
        E2 = E02 + delta_t*(1/tau_E)*(-E02 + np.maximum(0,I2 - rheobase_E))

        P1 = P01 + delta_t*(1/tau_P)*(-P01 + np.maximum(0, w_PEii * E01 + w_PEij * E02 - w_PSii * S01 - w_PSij * S02
                                                         -w_PPii * P01 - w_PPij * P02 + g_P - rheobase_P + stimulus_P1))
        P2 = P02 + delta_t*(1/tau_P)*(-P02 + np.maximum(0, w_PEij * E01 + w_PEii * E02 - w_PSij * S01 - w_PSii * S02
                                                         -w_PPij * P01 - w_PPii * P02 + g_P - rheobase_P + stimulus_P2))

        S1 = S01 + delta_t*(1/tau_S)*(-S01 + np.maximum(0, w_SEii * E01 + w_SEij * E02 + g_S - rheobase_S + stimulus_S1))
        S2 = S02 + delta_t*(1/tau_S)*(-S02 + np.maximum(0, w_SEij * E01 + w_SEii * E02 + g_S - rheobase_S + stimulus_S2))

        # Firing rates, set-points and set-point regulators cannot go below 0
        E1 = max(E1, 0); E2 = max(E2, 0)
        P1 = max(P1, 0); P2 = max(P2, 0)
        S1 = max(S1, 0); S2 = max(S2, 0)
        beta1=max(beta1,0); beta2=max(beta2, 0)
        theta1=max(theta1,1e-10); theta2=max(theta2, 1e-10) # Nonzero lower boundary to prevent zero division in scaling equation


        ### Calculating the plasticity for this timestep
        # Set point regulators for the E populations
        beta1 = beta1 + adaptive_set_point_flag*delta_t * (1 / tau_beta) * (E1 - beta1)
        beta2 = beta2 + adaptive_set_point_flag*delta_t * (1 / tau_beta) * (E2 - beta2)

        # Set points for the E populations
        theta1 = theta1 + adaptive_set_point_flag*delta_t * (1 / tau_theta) * \
                   (-flag_theta_shift*(theta1 - beta1) + flag_theta_local*(E1 - theta1))
        theta2 = theta2 + adaptive_set_point_flag*delta_t * (1 / tau_theta) * \
                   (-flag_theta_shift*(theta2 - beta2) + flag_theta_local*(E2 - theta2))

        # Ratios in the synaptic scaling equations are calculated
        ratio_E1 = E1 / theta1; ratio_E2 = E2 / theta2

        # Synaptic scaling terms are calculated and applied
        ss1_e = E_scaling_flag * delta_t * (1 / tau_scaling_E) * ((1 - ratio_E1))
        ss2_e = E_scaling_flag * delta_t * (1 / tau_scaling_E) * ((1 - ratio_E2))

        ss1_p = P_scaling_flag*delta_t * (1 / tau_scaling_P) * ((1 - ratio_E1))
        ss2_p = P_scaling_flag*delta_t * (1 / tau_scaling_P) * ((1 - ratio_E2))

        ss1_s = S_scaling_flag*delta_t * (1 / tau_scaling_S) * ((1 - ratio_E1))
        ss2_s = S_scaling_flag*delta_t * (1 / tau_scaling_S) * ((1 - ratio_E2))

        EE110 = EE110 + ss1_e*EE110
        EE120 = EE120 + ss1_e*EE120
        EE210 = EE210 + ss2_e*EE210
        EE220 = EE220 + ss2_e*EE220
        EP11  = EP110 - ss1_p*EP110
        EP12  = EP120 - ss1_p*EP120
        EP21  = EP210 - ss2_p*EP210
        EP22  = EP220 - ss2_p*EP220
        ES11  = ES110 + ss1_s*ES110
        ES12  = ES120 + ss1_s*ES120
        ES21  = ES210 + ss2_s*ES210
        ES22  = ES220 + ss2_s*ES220

        # Hebbian terms are calculated and applied
        heb_term11 = hebbian_flag * learning_rate * delta_t * (1 / tau_plas) * (E1 - r_baseline) * E1
        heb_term12 = hebbian_flag * learning_rate * delta_t * (1 / tau_plas) * (E1 - r_baseline) * E2
        heb_term21 = hebbian_flag * learning_rate * delta_t * (1 / tau_plas) * (E2 - r_baseline) * E1
        heb_term22 = hebbian_flag * learning_rate * delta_t * (1 / tau_plas) * (E2 - r_baseline) * E2

        EE11 = EE110 + heb_term11
        EE12 = EE120 + heb_term12
        EE21 = EE210 + heb_term21
        EE22 = EE220 + heb_term22

        # Lower bondary is applied to the weights
        EE11 = max(0,EE11);EE12 = max(0,EE12)
        EE21 = max(0,EE21);EE22 = max(0,EE22)
        EP11 = max(0,EP11);EP12 = max(0,EP12)
        EP21 = max(0,EP21);EP22 = max(0,EP22)
        ES11 = max(0,ES11);ES12 = max(0,ES12)
        ES21 = max(0,ES21);ES22 = max(0,ES22)

        # Placeholder parameters are freed
        E01 = E1; E02 = E2; P01 = P1; P02 = P2; S01 = S1; S02 = S2
        EE110=EE11; EE120=EE12; EE210=EE21; EE220=EE22
        EP110=EP11; EP120=EP12; EP210=EP21; EP220=EP22
        ES110=ES11; ES120=ES12; ES210=ES21; ES220=ES22

        # update the data-holder counters
        counter1 = counter1 + 1; counter2 = counter2 + 1; counter3 = counter3 + 1