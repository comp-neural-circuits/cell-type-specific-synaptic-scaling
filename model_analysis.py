import numpy as np
import matplotlib.pyplot as plt
from util import *
import sys
from model import *
from plotting_functions import *
import os
# from parameters import *
import pickle

def analyze_model(hour_sim, flags_list, flags_theta=(1,1), dir_data=r'\figures\data\\', dir_plot=r'\figures\\',
                  K=0.25, g_top_down_to_S=0, flag_only_S_on=False, run_simulation=True, save_results = False, plot_results=False):
    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param flags_list: contains a list of tuples. Each tuple is a collection of all the flags (e.g. synaptic scaling, hebbian learning, ...)
    :param flags_theta: used to study the behaviour of the model (no longer useful). Theta1 for population1 and Theta2 for population2
    :param dir_data
    :param dir_plot
    :param K: Tunes the steady state value of target activity and its regulator
    :param g_top_down_to_S: Represents the top-down signal to SST neurons triggered by the hyperexcitation. It reaches
    SST neurons at the offset of the conditioning
    :param run_simulation: True to run the numerical simulation, False to read the already saved data
    :param save_results: True to save the results
    :param plot_results: True to plot the results

    Multi-purpose function to analyze the model. Here we run (if run_simulation is True) our computational model to
    investigate the role of cell-type dependent synaptic scaling mechanisms in associative learning. We replicate the
    experimental procedure in [1] in model() in model.py. The model has two subnetworks, each consisted of a canonical
    circuit of excitatory pyramidal neurons (E), parvalbumin-positive interneurons (P), somatostatin-positive
    interneurons (S). The simulation procedure is divided into three phases:
        Phase 1 - Conditioning: The first subnetwork receives extra input representing the conditioned stimulus in [1].
        The parameters describing the stimulation (when and how much stimulation) is described in analyze_model()
        function. The onset response of the excitatory firing rate of the first subnetwork is defined as the aversion
        threshold of this network. Three-factor Hebbian learning is active during this period. Also, synaptic scaling
        mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 2: In the experiment [1], the novel stimulus is presented to the mice at 4h/24h/48h after conditioning.
        This phase corresponds to the waiting time after conditioning and before testing. During this phase, synaptic
        scaling mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 3 - Testing: In this phase, the second subnetwork receives extra input corresponds to the novel stimulus
        in [1]. The memory specificity/overgeneralization is determined whether the excitatory rate in the second
        subnetwork is below/above the aversion threshold, respectively.


    During simulation model() writes data to the data arrays. This data can be saved (if save_results is set to True)
    and the results can be plotted (if plot_results is set to True).

    [1] Wu, C. H., Ramos, R., Katz, D. B., & Turrigiano, G. G. (2021). Homeostatic synaptic scaling establishes the
    specificity of an associative memory. Current biology, 31(11), 2274-2285.
    """

    stim_duration = 15  # stimulation duration in seconds
    # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each, 2 extra seconds to reach steady state initially (thermalization)
    sim_duration = int((hour_sim) * 60 * 60 + (stim_duration + 10) * 2 + 2)
    delta_t = 0.0001  # time step in seconds (0.1 ms)
    sampling_rate_stim = 20  # register data at every 20 step during phase 1 and 3 (conditioning and testing)
    sampling_rate_sim = 200_000  # register data at every 2e5 time step (20 seconds) during phase 2 (in between conditioning and testing)
    sampling_rate = (sampling_rate_stim, sampling_rate_sim)

    # Total number of timepoints for stimulation and simulation
    n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
    n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1 # total no the rest

    l_time_points_stim = np.linspace(0, stim_duration + 10, n_time_points_stim) #time points for the first 15s
    l_time_points_phase2 = np.linspace(0, hour_sim, n_time_points_phase2) ##time points for the seoncd phase 4/24/48h

    # Timepoints of the onset (first column) and offset (second column) of the first (first row) and second (second) stimuli.
    stim_times = np.array([[5, 5 + stim_duration],
                           [int(hour_sim * 60 * 60) + 5, int(hour_sim * 60 * 60) + 5 + stim_duration]]).reshape(2, 2)

    # The stimuli are given as inputs to the populations.
    g_stim_E = np.array([(1, 0), (0, 1)])
    g_stim_P = np.array([(0.5, 0), (0, 0.5)])
    g_stim_S = np.array([(0, 0), (0, 0)])
    g_stim = (g_stim_E, g_stim_P, g_stim_S)

    # Time constants
    tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
    tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
    tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
    tau_hebb = 240  # time constant of three-factor Hebbian learning in seconds(2min)
    tau_theta = 24 * (60 * 60)  # time constant of target activity in seconds(24h)
    tau_beta = 28 * (60 * 60)  # time constant of target activity regulator in seconds(28h)
    tau_scaling_E = 8 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
    tau_scaling_P = 8 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
    tau_scaling_S = 8 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
    taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

    # Rheobases (minimum input needed for firing rates to be above zero)
    rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
    rheobases = (rheobase_E, rheobase_P, rheobase_S)

    # Background inputs
    g_E = 4.5
    g_P = 3.2
    g_S = 3
    back_inputs = (g_E, g_P, g_S, g_top_down_to_S)

    # Initial conditions for plastic weights
    # w_EP_within = 0.81; w_EP_cross = 0.41
    # w_ES_within = 0.81; w_ES_cross = 0.31
    # w_EE_within = 0.71; w_EE_cross = 0.41
    w_EP_within = 0.91; w_EP_cross = 0.41
    w_ES_within = 0.51; w_ES_cross = 0.31
    w_EE_within = 0.51; w_EE_cross = 0.51

    # # Initial conditions for plastic weights
    # w_EP_within = 0.7; w_EP_cross = w_EP_within*0.3
    # w_ES_within = 0.7; w_ES_cross = w_ES_within*0.3
    # w_EE_within = 0.5; w_EE_cross = 0.4

    # Weights
    w_PE_within = 0.3; w_PE_cross = 0.1
    w_PP_within = 0.2; w_PP_cross = 0.1
    w_PS_within = 0.3; w_PS_cross = 0.1
    w_SE_within = 0.4; w_SE_cross = 0.1

    weights = (w_EE_within, w_EP_within, w_ES_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
               w_EE_cross, w_EP_cross, w_ES_cross, w_PE_cross, w_PP_cross, w_PS_cross, w_SE_cross)

    # Arrays created to hold data
    r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    J_EE_phase1 = np.zeros((4, n_time_points_stim), dtype=np.float32)  # WEE11,WEE12,WEE21,WEE22
    r_phase2 = np.zeros((10, n_time_points_phase2),
                        dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2,theta1,theta2,beta1,beta2
    J_phase2 = np.zeros((12, n_time_points_phase2),
                        dtype=np.float32)  # WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
    r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    max_E = np.zeros(1, dtype=np.float32)

    # Lists to hold data arrays
    l_res_rates = (r_phase1, r_phase2, r_phase3, max_E)
    l_res_weights = (J_EE_phase1, J_phase2)

    for flags in flags_list:
        id, title = determine_name(flags)
        name = 'Case' + id + '_' + str(hour_sim) + 'h' + '_k' + str(K).replace(".","") + '_td' + str(g_top_down_to_S)

        print('*****', title, '*****')

        if run_simulation:

            print('Simulation started.')
            print('\n')

            #All flags = 0 and simulation is 30 seconds long. It is used to evaluate what happens when activating E1 what's the response of E2. Afterwards it is evaluating the av_threshold given the result
            model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(30 * (1 / delta_t)), weights,
                  back_inputs, g_stim, stim_times, taus, K, rheobases, flags=(0,0,0,0,0,0), flags_theta=flags_theta)

            idx_av_threshold = int(15 * (1 / delta_t) * (1 / sampling_rate_stim))
            # av_threshold = r_phase1[1][idx_av_threshold] * 1.15 #it is defined with an extra 15% for old reason. not required anymore
            av_threshold = r_phase1[1][idx_av_threshold]

            model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(sim_duration * (1 / delta_t)), weights,
                  back_inputs, g_stim, stim_times, taus, K, rheobases, flags=flags, flags_theta=flags_theta)

            if save_results:
                l_results = [l_time_points_stim, l_time_points_phase2, delta_t, sampling_rate, l_res_rates,
                             l_res_weights,
                             av_threshold, stim_times, stim_duration, sim_duration]

                # Open a file and save
                with open(dir_data + name + '.pkl', 'wb') as file:
                    # A new file will be created
                    pickle.dump(l_results, file)
                print('Data is saved.')

        else:
            # Open the file and read
            with open(dir_data + name + '.pkl', 'rb') as file:
                l_results = pickle.load(file)
            print('Data is read.')

            [l_time_points_stim, l_time_points_phase2, delta_t, sampling_rate, l_res_rates, l_res_weights,
             av_threshold, stim_times, stim_duration, sim_duration] = l_results

        if plot_results:
            print('Plotting the results.')
            time_plots([l_time_points_stim, l_time_points_phase2], l_res_rates, l_res_weights, av_threshold,
                       stim_times, dir_plot + name, hour_sim, flag_only_S_on=flag_only_S_on, format='.png')
            time_plots([l_time_points_stim, l_time_points_phase2], l_res_rates, l_res_weights, av_threshold,
                       stim_times, dir_plot + name, hour_sim, flag_only_S_on=flag_only_S_on, format='.pdf')


def plot_testing_at_regular_intervals(flags_list, flags_theta=(1,1), dir_data=r'\figures\data\\', dir_plot=r'\figures\\',
                                      K=0.25, g_top_down_to_S=0, flag_only_S_on=False, run_simulation=True,
                                      save_results = False, plot_results=False):

    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param flags_list
    :param flags_theta
    :param dir_data
    :param dir_plot
    :param K: Tunes the steady state value of target activity and its regulator
    :param g_top_down_to_S: Represents the top-down signal to SST neurons triggered by the hyperexcitation. It reaches
    SST neurons at the offset of the conditioning
    :param run_simulation: True to run the numerical simulation, False to read the already saved data
    :param save_results: True to save the results
    :param plot_results: True to plot the results

    Multi-purpose function to analyze the model. Here we run (if run_simulation is True) our computational model to
    investigate the role of cell-type dependent synaptic scaling mechanisms in associative learning. We replicate the
    experimental procedure in [1] in model() in model.py. The model has two subnetworks, each consisted of a canonical
    circuit of excitatory pyramidal neurons (E), parvalbumin-positive interneurons (P), somatostatin-positive
    interneurons (S). The simulation procedure is divided into three phases:
        Phase 1 - Conditioning: The first subnetwork receives extra input representing the conditioned stimulus in [1].
        The parameters describing the stimulation (when and how much stimulation) is described in analyze_model()
        function. The onset response of the excitatory firing rate of the first subnetwork is defined as the aversion
        threshold of this network. Three-factor Hebbian learning is active during this period. Also, synaptic scaling
        mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 2: In the experiment [1], the novel stimulus is presented to the mice at 4h/24h/48h after conditioning.
        This phase corresponds to the waiting time after conditioning and before testing. During this phase, synaptic
        scaling mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 3 - Testing: In this phase, the second subnetwork receives extra input corresponds to the novel stimulus
        in [1]. The memory specificity/overgeneralization is determined whether the excitatory rate in the second
        subnetwork is below/above the aversion threshold, respectively.


    During simulation model() writes data to the data arrays. This data can be saved (if save_results is set to True)
    and the results can be plotted (if plot_results is set to True).

    [1] Wu, C. H., Ramos, R., Katz, D. B., & Turrigiano, G. G. (2021). Homeostatic synaptic scaling establishes the
    specificity of an associative memory. Current biology, 31(11), 2274-2285.
    """

    hour_sims = np.arange(48) + 1

    for flags in flags_list:
        id, title = determine_name(flags)
        name = 'Case' + id + '_test_every_h' + '_k' + str(K).replace(".","") + '_td' + str(g_top_down_to_S)

        l_delta_rE1 = []

        print('*****', title, '*****')

        if run_simulation:
            print('Simulation started.')
            print('\n')
            for hour_sim in hour_sims:
                stim_duration = 15 # stimulation duration in seconds
                # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each, 2 extra seconds to reach steady state initially
                sim_duration = int((hour_sim) * 60 * 60 + (stim_duration + 5 + 5) * 2 + 2)
                delta_t = 0.0001 # time step in seconds (0.1 ms)
                sampling_rate_stim = 20 # register data at every 20 step during phase 1 and 3 (conditioning and testing)
                sampling_rate_sim = 200000 # register data at every 2e5 time step (20 seconds) during phase 2 (in between conditioning and testing)
                sampling_rate = (sampling_rate_stim, sampling_rate_sim)

                # Total number of timepoints for stimulation and simulation
                n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
                n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1 # total no the rest

                # l_time_points_stim = np.linspace(0, stim_duration + 10, n_time_points_stim)
                l_time_points_phase2 = np.linspace(0, hour_sim, n_time_points_phase2)

                # Timepoints of the onset (first column) and offset (second column) of the first (first row) and second (second) stimuli.
                stim_times = np.array([[5, 5 + stim_duration],
                                       [int(hour_sim * 60 * 60) + 5, int(hour_sim * 60 * 60) + 5 + stim_duration]]).reshape(2, 2)

                # The stimuli are given as inputs to the populations.
                g_stim_E = np.array([(1, 0), (0, 1)])
                g_stim_P = np.array([(0.5, 0), (0, 0.5)])
                g_stim_S = np.array([(0, 0), (0, 0)])
                g_stim = (g_stim_E, g_stim_P, g_stim_S)

                # Time constants
                tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
                tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
                tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
                tau_hebb = 240  # time constant of three-factor Hebbian learning in seconds(2min)
                tau_theta = 24 * (60 * 60)  # time constant of target activity in seconds(24h)
                tau_beta = 28 * (60 * 60)  # time constant of target activity regulator in seconds(28h)
                tau_scaling_E = 8 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
                tau_scaling_P = 8 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
                tau_scaling_S = 8 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
                taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

                # Rheobases (minimum input needed for firing rates to be above zero)
                rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
                rheobases = (rheobase_E, rheobase_P, rheobase_S)

                # Background inputs
                g_E = 4.5
                g_P = 3.2
                g_S = 3
                back_inputs = (g_E, g_P, g_S, g_top_down_to_S)

                # Initial conditions for plastic weights
                # w_EP_within = 0.81; w_EP_cross = 0.41
                # w_ES_within = 0.81; w_ES_cross = 0.31
                # w_EE_within = 0.71; w_EE_cross = 0.41
                w_EP_within = 0.91; w_EP_cross = 0.41
                w_ES_within = 0.51; w_ES_cross = 0.31
                w_EE_within = 0.51; w_EE_cross = 0.51

                # # Initial conditions for plastic weights
                # w_EP_within = 0.7; w_EP_cross = w_EP_within*0.3
                # w_ES_within = 0.7; w_ES_cross = w_ES_within*0.3
                # w_EE_within = 0.5; w_EE_cross = 0.4

                # Weights
                w_PE_within = 0.3; w_PE_cross = 0.1
                w_PP_within = 0.2; w_PP_cross = 0.1
                w_PS_within = 0.3; w_PS_cross = 0.1
                w_SE_within = 0.4; w_SE_cross = 0.1

                weights = (w_EE_within, w_EP_within, w_ES_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
                           w_EE_cross, w_EP_cross, w_ES_cross, w_PE_cross, w_PP_cross, w_PS_cross, w_SE_cross)

                # Arrays created to hold data
                r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2
                J_EE_phase1 = np.zeros((4, n_time_points_stim), dtype=np.float32) # WEE11,WEE12,WEE21,WEE22
                r_phase2 = np.zeros((10, n_time_points_phase2), dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2,theta1,theta2,beta1,beta2
                J_phase2 = np.zeros((12, n_time_points_phase2),dtype=np.float32) # WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
                r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2
                max_E = np.zeros(1, dtype=np.float32)

                # Lists to hold data arrays
                l_res_rates = (r_phase1, r_phase2, r_phase3, max_E)
                l_res_weights = (J_EE_phase1, J_phase2)

                model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(30 * (1 / delta_t)), weights,
                      back_inputs, g_stim, stim_times, taus, K, rheobases, flags=(0,0,0,0,0,0), flags_theta=flags_theta)

                idx_av_threshold = int(15 * (1 / delta_t) * (1 / sampling_rate_stim))
                # av_threshold = r_phase1[1][idx_av_threshold] * 1.15 #it is defined with an extra 15% for old reason. not required anymore
                av_threshold = r_phase1[1][idx_av_threshold]

                model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(sim_duration * (1 / delta_t)), weights,
                      back_inputs, g_stim, stim_times, taus, K, rheobases, flags=flags, flags_theta=flags_theta)

                l_delta_rE1.append(np.max(r_phase3[0][int(stim_times[0][0] * (1 / (delta_t * sampling_rate_stim))):
                                               int(stim_times[0][1] * (1 / (delta_t * sampling_rate_stim)))]).copy())

                print('Simulation of ' + str(hour_sim) + ' hours is completed')
            if save_results:
                l_results = [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim]

                # Open a file and save
                with open(dir_data + name + '.pkl', 'wb') as file:
                    # A new file will be created
                    pickle.dump(l_results, file)
                print('Data is saved.')

        else:
            # Open the file and read
            with open(dir_data + name + '.pkl', 'rb') as file:
                l_results = pickle.load(file)
            print('Data is read.')

            [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim] = l_results

        if plot_results:
            print('Plotting the results.')
            change_in_reactivation_every_h_vslides(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.png')
            change_in_reactivation_every_h_vslides(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.pdf')


def plot_all_cases_CIR(dir_data=r'\figures\data\\', dir_plot=r'\figures\\', K=0.25, g_top_down_to_S=0):

    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param flags_list
    :param flags_theta
    :param dir_data
    :param dir_plot
    :param K: Tunes the steady state value of target activity and its regulator
    :param g_top_down_to_S: Represents the top-down signal to SST neurons triggered by the hyperexcitation. It reaches
    SST neurons at the offset of the conditioning
    :param run_simulation: True to run the numerical simulation, False to read the already saved data
    :param save_results: True to save the results
    :param plot_results: True to plot the results

    Multi-purpose function to analyze the model. Here we run (if run_simulation is True) our computational model to
    investigate the role of cell-type dependent synaptic scaling mechanisms in associative learning. We replicate the
    experimental procedure in [1] in model() in model.py. The model has two subnetworks, each consisted of a canonical
    circuit of excitatory pyramidal neurons (E), parvalbumin-positive interneurons (P), somatostatin-positive
    interneurons (S). The simulation procedure is divided into three phases:
        Phase 1 - Conditioning: The first subnetwork receives extra input representing the conditioned stimulus in [1].
        The parameters describing the stimulation (when and how much stimulation) is described in analyze_model()
        function. The onset response of the excitatory firing rate of the first subnetwork is defined as the aversion
        threshold of this network. Three-factor Hebbian learning is active during this period. Also, synaptic scaling
        mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 2: In the experiment [1], the novel stimulus is presented to the mice at 4h/24h/48h after conditioning.
        This phase corresponds to the waiting time after conditioning and before testing. During this phase, synaptic
        scaling mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 3 - Testing: In this phase, the second subnetwork receives extra input corresponds to the novel stimulus
        in [1]. The memory specificity/overgeneralization is determined whether the excitatory rate in the second
        subnetwork is below/above the aversion threshold, respectively.


    During simulation model() writes data to the data arrays. This data can be saved (if save_results is set to True)
    and the results can be plotted (if plot_results is set to True).

    [1] Wu, C. H., Ramos, R., Katz, D. B., & Turrigiano, G. G. (2021). Homeostatic synaptic scaling establishes the
    specificity of an associative memory. Current biology, 31(11), 2274-2285.
    """
    # directory = os.getcwd()
    directory = ""

    hour_sims = np.arange(48) + 1
    l_all_delta_rE1 = []
    name_plot = 'CIR_all_cases'

    flags_list = [(1, 1, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 0, 1), (1, 1, 1, 1, 1, 0),
                  (1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 0), (1, 1, 1, 0, 0, 1), (1, 1, 1, 0, 0, 0)]

    for flags in flags_list:
        id, title = determine_name(flags)
        name_data = 'Case' + id + '_test_every_h' + '_k' + str(K).replace(".","") + '_td' + str(g_top_down_to_S)

        # Open the file and read
        with open(directory+dir_data + name_data + '.pkl', 'rb') as file:
            l_results = pickle.load(file)
        print('Data is read.')

        [_, l_time_points_phase2, _, l_delta_rE1, av_threshold, _, _] = l_results
        l_all_delta_rE1.append(l_delta_rE1)

    print('Plotting the results.')
    all_cases_CIR(l_time_points_phase2, hour_sims, l_all_delta_rE1, av_threshold, directory+dir_plot+name_plot, format='.png')
    all_cases_CIR(l_time_points_phase2, hour_sims, l_all_delta_rE1, av_threshold, directory+dir_plot+name_plot, format='.pdf')


def plot_testing_at_regular_intervals_weights(ww_weights,flags_list, flags_theta=(1,1), dir_data=r'\figures\data\\', dir_plot=r'\figures\\',
                                      K=0.25, g_top_down_to_S=0, flag_only_S_on=False, run_simulation=True,
                                      save_results = False, plot_results=False):

    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param flags_list
    :param flags_theta
    :param dir_data
    :param dir_plot
    :param K: Tunes the steady state value of target activity and its regulator
    :param g_top_down_to_S: Represents the top-down signal to SST neurons triggered by the hyperexcitation. It reaches
    SST neurons at the offset of the conditioning
    :param run_simulation: True to run the numerical simulation, False to read the already saved data
    :param save_results: True to save the results
    :param plot_results: True to plot the results

    Multi-purpose function to analyze the model. Here we run (if run_simulation is True) our computational model to
    investigate the role of cell-type dependent synaptic scaling mechanisms in associative learning. We replicate the
    experimental procedure in [1] in model() in model.py. The model has two subnetworks, each consisted of a canonical
    circuit of excitatory pyramidal neurons (E), parvalbumin-positive interneurons (P), somatostatin-positive
    interneurons (S). The simulation procedure is divided into three phases:
        Phase 1 - Conditioning: The first subnetwork receives extra input representing the conditioned stimulus in [1].
        The parameters describing the stimulation (when and how much stimulation) is described in analyze_model()
        function. The onset response of the excitatory firing rate of the first subnetwork is defined as the aversion
        threshold of this network. Three-factor Hebbian learning is active during this period. Also, synaptic scaling
        mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 2: In the experiment [1], the novel stimulus is presented to the mice at 4h/24h/48h after conditioning.
        This phase corresponds to the waiting time after conditioning and before testing. During this phase, synaptic
        scaling mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 3 - Testing: In this phase, the second subnetwork receives extra input corresponds to the novel stimulus
        in [1]. The memory specificity/overgeneralization is determined whether the excitatory rate in the second
        subnetwork is below/above the aversion threshold, respectively.


    During simulation model() writes data to the data arrays. This data can be saved (if save_results is set to True)
    and the results can be plotted (if plot_results is set to True).

    [1] Wu, C. H., Ramos, R., Katz, D. B., & Turrigiano, G. G. (2021). Homeostatic synaptic scaling establishes the
    specificity of an associative memory. Current biology, 31(11), 2274-2285.
    """

    hour_sims = np.arange(48) + 1

    # print('Simulation started.')
    for flags in flags_list:
        id, title = determine_name(flags)
        name = 'Case' + id + '_test_every_h' + '_k' + str(K).replace(".","") + '_td' + str(g_top_down_to_S) + "_"
        name += '_'.join(str(weight).replace(".","") for weight in ww_weights)

        l_delta_rE1 = []
        # print('*****', title, '*****')

        if run_simulation:
            # print('Simulation started.')
            # print('\n')
            for hour_sim in hour_sims:
                stim_duration = 15 # stimulation duration in seconds
                # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each, 2 extra seconds to reach steady state initially
                sim_duration = int((hour_sim) * 60 * 60 + (stim_duration + 5 + 5) * 2 + 2)
                delta_t = 0.0001 # time step in seconds (0.1 ms)
                sampling_rate_stim = 20 # register data at every 20 step during phase 1 and 3 (conditioning and testing)
                sampling_rate_sim = 200000 # register data at every 2e5 time step (20 seconds) during phase 2 (in between conditioning and testing)
                sampling_rate = (sampling_rate_stim, sampling_rate_sim)

                # Total number of timepoints for stimulation and simulation
                n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
                n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1 # total no the rest

                # l_time_points_stim = np.linspace(0, stim_duration + 10, n_time_points_stim)
                l_time_points_phase2 = np.linspace(0, hour_sim, n_time_points_phase2)

                # Timepoints of the onset (first column) and offset (second column) of the first (first row) and second (second) stimuli.
                stim_times = np.array([[5, 5 + stim_duration],
                                       [int(hour_sim * 60 * 60) + 5, int(hour_sim * 60 * 60) + 5 + stim_duration]]).reshape(2, 2)

                # The stimuli are given as inputs to the populations.
                g_stim_E = np.array([(1, 0), (0, 1)])
                g_stim_P = np.array([(0.5, 0), (0, 0.5)])
                g_stim_S = np.array([(0, 0), (0, 0)])
                g_stim = (g_stim_E, g_stim_P, g_stim_S)

                # Time constants
                tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
                tau_P = 0.005 # time constant of P population firing rate in seconds(5ms)
                tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
                tau_hebb = 240 # time constant of three-factor Hebbian learning in seconds(2min)
                tau_theta = 24 * (60 * 60) # time constant of target activity in seconds(24h)
                tau_beta = 28 * (60 * 60) # time constant of target activity regulator in seconds(28h)
                tau_scaling_E = 8 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
                tau_scaling_P = 8 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
                tau_scaling_S = 8 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
                taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

                # Rheobases (minimum input needed for firing rates to be above zero)
                rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
                rheobases = (rheobase_E, rheobase_P, rheobase_S)

                # Background inputs
                g_E = 4.5
                g_P = 3.2
                g_S = 3
                back_inputs = (g_E, g_P, g_S, g_top_down_to_S)

                # Initial conditions for plastic weights
                (w_EP_within, w_EP_cross, w_ES_within, w_ES_cross, w_EE_within, w_EE_cross) = ww_weights

                # Weights
                w_PE_within = 0.3; w_PE_cross = 0.1
                w_PP_within = 0.2; w_PP_cross = 0.1
                w_PS_within = 0.3; w_PS_cross = 0.1
                w_SE_within = 0.4; w_SE_cross = 0.1

                weights = (w_EE_within, w_EP_within, w_ES_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
                           w_EE_cross, w_EP_cross, w_ES_cross, w_PE_cross, w_PP_cross, w_PS_cross, w_SE_cross)

                # Arrays created to hold data
                # r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2
                # J_EE_phase1 = np.zeros((4, n_time_points_stim), dtype=np.float32) # WEE11,WEE12,WEE21,WEE22
                # r_phase2 = np.zeros((10, n_time_points_phase2), dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2,theta1,theta2,beta1,beta2
                # J_phase2 = np.zeros((12, n_time_points_phase2),dtype=np.float32) # WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
                # r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2
                max_E = np.zeros(1, dtype=np.float32)

                r_phase1 = np.full((6, n_time_points_stim), np.nan, dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2
                J_EE_phase1 = np.full((4, n_time_points_stim), np.nan, dtype=np.float32) # WEE11,WEE12,WEE21,WEE22
                r_phase2 = np.full((10, n_time_points_phase2), np.nan, dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2,theta1,theta2,beta1,beta2
                J_phase2 = np.full((12, n_time_points_phase2), np.nan, dtype=np.float32) # WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
                r_phase3 = np.full((6, n_time_points_stim), np.nan, dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2

                # Lists to hold data arrays
                l_res_rates = (r_phase1, r_phase2, r_phase3, max_E)
                l_res_weights = (J_EE_phase1, J_phase2)

                model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(30 * (1 / delta_t)), weights,
                      back_inputs, g_stim, stim_times, taus, K, rheobases, flags=(0,0,0,0,0,0), flags_theta=flags_theta)

                idx_av_threshold = int(15 * (1 / delta_t) * (1 / sampling_rate_stim))
                av_threshold = r_phase1[1][idx_av_threshold] * 1.15

                model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(sim_duration * (1 / delta_t)), weights,
                      back_inputs, g_stim, stim_times, taus, K, rheobases, flags=flags, flags_theta=flags_theta)

                l_delta_rE1.append(np.max(r_phase3[0][int(stim_times[0][0] * (1 / (delta_t * sampling_rate_stim))):
                                               int(stim_times[0][1] * (1 / (delta_t * sampling_rate_stim)))]).copy())

                # print('Simulation of ' + str(hour_sim) + ' hours is completed')
            if save_results:
                l_results = [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim]

                # Open a file and save
                with open(dir_data + name + '.pkl', 'wb') as file:
                    # A new file will be created
                    pickle.dump(l_results, file)

        else:
            # Open the file and read
            with open(dir_data + name + '.pkl', 'rb') as file:
                l_results = pickle.load(file)
            print('Data is read.')

            [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim] = l_results

        if plot_results:
            print('Plotting the results.')
            change_in_reactivation_every_h(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.png')
            change_in_reactivation_every_h(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.pdf')
    print("Data for", '_'.join(str(weight).replace(".", "") for weight in ww_weights), "is saved\n")
