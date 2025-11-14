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
                  K=0.25, flag_only_S_on=False, run_simulation=True, save_results = False, plot_results=False,modulation_SST=0):
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

    os.makedirs(dir_data, exist_ok=True)
    os.makedirs(dir_plot, exist_ok=True)
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
    if modulation_SST == 0:
        g_stim_S = np.array([(0, 0), (0, 0)])
    elif modulation_SST > 0:
        g_stim_S = np.array([(0.5, 0), (0, 0.5)])
    elif modulation_SST < 0:
        g_stim_S = np.array([(-0.5, 0), (0, -0.5)])
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
    back_inputs = (g_E, g_P, g_S)

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
        name = 'Case' + id + '_' + str(hour_sim) + 'h' + '_k' + str(K).replace(".","")

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
                       stim_times, dir_plot + name, hour_sim,modulation_SST, flag_only_S_on=flag_only_S_on, format='.png')
            time_plots([l_time_points_stim, l_time_points_phase2], l_res_rates, l_res_weights, av_threshold,
                       stim_times, dir_plot + name, hour_sim,modulation_SST, flag_only_S_on=flag_only_S_on, format='.pdf')


#this function is a generalized version of the one above. this one, with the right flags, is the only one necessary. For clarity, they are separated
def analyze_model_timescales(hour_sim, flags_list, flags_theta=(1,1), dir_data=r'\figures\data\\', dir_plot=r'\figures\\',
                  K=0.25, flag_only_S_on=False, run_simulation=True, save_results = False, plot_results=False,modulation_SST=0,timescales_exploration=False):
    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param flags_list: contains a list of tuples. Each tuple is a collection of all the flags (e.g. synaptic scaling, hebbian learning, ...)
    :param flags_theta: used to study the behaviour of the model (no longer useful). Theta1 for population1 and Theta2 for population2
    :param dir_data
    :param dir_plot
    :param K: Tunes the steady state value of target activity and its regulator
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
    os.makedirs(dir_data, exist_ok=True)
    os.makedirs(dir_plot, exist_ok=True)
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
    if modulation_SST == 0:
        g_stim_S = np.array([(0, 0), (0, 0)])
    elif modulation_SST > 0:
        g_stim_S = np.array([(0.5, 0), (0, 0.5)])
    elif modulation_SST < 0:
        g_stim_S = np.array([(-0.5, 0), (0, -0.5)])
    g_stim = (g_stim_E, g_stim_P, g_stim_S)

    # Time constants
    tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
    tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
    tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
    tau_hebb = 240  # time constant of three-factor Hebbian learning in seconds(2min)
    tau_theta_list = [12*3600]
    tau_beta_list  = [0.01*3600]
    # tau_theta_list = (np.arange(24-12, 24+12) * 3600).tolist()
    # tau_beta_list = (np.arange(28-12, 28+12) * 3600).tolist()
    # tau_theta_list = [6*3600, 26*3600, 260*3600]
    # tau_beta_list  = [1e-20*3600, 1e20*3600]
    tau_scaling_E = 8 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
    tau_scaling_P = 8 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
    tau_scaling_S = 8 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)

    # Rheobases (minimum input needed for firing rates to be above zero)
    rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
    rheobases = (rheobase_E, rheobase_P, rheobase_S)

    # Background inputs
    g_E = 4.5
    g_P = 3.2
    g_S = 3
    back_inputs = (g_E, g_P, g_S)

    for tau_beta in tau_beta_list:
        for tau_theta in tau_theta_list:
            taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)
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
                name = 'Case' + id + '_' + str(hour_sim) + 'h' + '_k' + str(K).replace(".","") + '_theta_' + str(int(tau_theta/3600)).replace(".","") + '_beta_' + str(int(tau_beta/3600)).replace(".","")

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
                    os.makedirs(dir_plot + 'theta_' + str(int(tau_theta/3600)).replace(".","") + '_beta_' + str(int(tau_beta/3600)).replace(".","") + "/", exist_ok=True)
                    time_plots([l_time_points_stim, l_time_points_phase2], l_res_rates, l_res_weights, av_threshold,
                            stim_times, dir_plot + 'theta_' + str(int(tau_theta/3600)).replace(".","") + '_beta_' + str(int(tau_beta/3600)).replace(".","") + "/" + name, hour_sim,modulation_SST, flag_only_S_on=flag_only_S_on, format='.png')
                    time_plots([l_time_points_stim, l_time_points_phase2], l_res_rates, l_res_weights, av_threshold,
                            stim_times, dir_plot + 'theta_' + str(int(tau_theta/3600)).replace(".","") + '_beta_' + str(int(tau_beta/3600)).replace(".","")+ "/" + name, hour_sim,modulation_SST, flag_only_S_on=flag_only_S_on, format='.pdf')



def analyze_model_3_compartmental_v3(hour_sim, flags_list, dir_data=r'\figures\data\\', dir_plot=r'\figures\\', modulation_SST=0, run_simulation=True, save_results=False, plot_results=False):
    """
    :param hour_sim: Defines how many hours does the simulation lasts
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
    os.makedirs(dir_data, exist_ok=True)
    os.makedirs(dir_plot, exist_ok=True)
    delta_t = 0.0001  # time step in seconds (0.1 ms)
    stim_duration = 15  # stimulation duration in seconds
    # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each, 2 extra seconds to reach steady state initially
    sim_duration = int(((hour_sim) * 60 * 60 + (stim_duration + 10) * 2 + 2)*(1/delta_t))
    sampling_rate_stim = 20  # register data at every 20 step during phase 1 and 3 (conditioning and testing)
    sampling_rate_sim = 200_000  # register data at every 2e5 time step (20 seconds) during phase 2 (in between conditioning and testing)
    sampling_rate = (sampling_rate_stim, sampling_rate_sim)

    # Total number of timepoints for stimulation and simulation
    n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
    n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1  # total no the rest

    l_time_points_stim = np.linspace(0, stim_duration + 10, n_time_points_stim)
    l_time_points_phase2 = np.linspace(0, hour_sim, n_time_points_phase2)

    # Timepoints of the onset (first column) and offset (second column) of the first (first row) and second (second) stimuli.
    stim_times = np.array([[5, 5 + stim_duration],
                           [int(hour_sim * 60 * 60) + 5, int(hour_sim * 60 * 60) + 5 + stim_duration]]).reshape(2, 2)

    # Time constants
    tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
    tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
    tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
    tau_dend = tau_E  # time constant of E population firing rate in seconds(20ms)
    tau_hebb = 120  # time constant of three-factor Hebbian learning in seconds(2min)
    tau_theta = 24 * (60 * 60)  # time constant of target activity in seconds(24h)
    tau_beta = 28 * (60 * 60)  # time constant of target activity regulator in seconds(28h)
    tau_scaling_E = 2.5 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
    tau_scaling_P = 6.5 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
    tau_scaling_S = 2.5 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
    taus = (tau_E, tau_P, tau_S, tau_dend, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

    # g_stim, rheobases, g, K, lambdas = get_model_parameters(coupling='strong', excitation='high')

    # The stimuli are given as inputs to the populations.
    g_stim_E = np.array([(2.5, 0), (0, 2.5)])
    g_stim_P = np.array([(0.5, 0), (0, 0.5)])
    if modulation_SST == 0:
        g_stim_S = np.array([(0, 0), (0, 0)])
    elif modulation_SST > 0:
        g_stim_S = np.array([(1.25, 0), (0, 1.25)])
    elif modulation_SST < 0:
        g_stim_S = np.array([(-1.25, 0), (0, -1.25)])
    g_stim = (g_stim_E, g_stim_P, g_stim_S)

    # Rheobases (minimum input needed for firing rates to be above zero)
    rheobase_E, rheobase_P, rheobase_S, rheobase_A,rheobase_B  = 1, 1.5, 1.5, 3, 9
    rheobases = (rheobase_E, rheobase_P, rheobase_S, rheobase_A,rheobase_B)

    # Background inputs
    g_AD = 4
    g_BD = 6
    g_E = 0
    g_P = 4
    g_S = 3.2
    g = (g_AD, g_BD, g_E, g_P, g_S)

    # K parameter in target regulator equation, it tunes the steady state value of target activity and its regulator
    K = 0.3

    # Constant that define the contribution of each current
    lambda_AD = 0.4 # in stronger lambda config it is 0.5
    lambda_BD = 0.3 # in stronger lambda config it is 0.3
    lambdas = (lambda_AD, lambda_BD)


    # Initial conditions for plastic weights
    w_EP_within = 0.6; w_EP_cross = 0.18 #PV -> soma
    w_DS_within = 0.4; w_DS_cross = 0.18# SST -> apioal
    w_DE_within = 0.5; w_DE_cross = 0.3 #E -> apical
    w_EE_within = 0.5; w_EE_cross = 0.3 #E -> basal

    # Weights
    w_PE_within = 0.35; w_PE_cross = 0.10 #E -> PV
    w_PP_within = 0.20; w_PP_cross = 0.10 #PV -> PV
    w_PS_within = 0.30; w_PS_cross = 0.10 #SST -> PV
    w_SE_within = 0.15; w_SE_cross = 0.10 #E -> SST

    weights = (w_DE_within, w_EE_within, w_EP_within, w_DS_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
               w_DE_cross,  w_EE_cross,  w_EP_cross,  w_DS_cross,  w_PE_cross,  w_PP_cross,  w_PS_cross,  w_SE_cross)

    # Arrays created to hold data
    r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    I_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # IAD1, IAD2, IBD1, IBD2, IE1, IE2
    J_exc_phase1 = np.zeros((8, n_time_points_stim), dtype=np.float32)  # WDE11,WDE12,WDE21,WDE22,WEE11,WEE12,WEE21,WEE22
    r_phase2 = np.zeros((6, n_time_points_phase2), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    set_phase2 = np.zeros((12, n_time_points_phase2), dtype=np.float32) # thetaDD1,thetaDD2,thetaBD1,thetaBD2,thetaE1,thetaE2,betaAD1,betaAD2,betaBD1,betaBD2,betaE1,betaE2
    I_phase2 = np.zeros((6, n_time_points_phase2), dtype=np.float32) # IAD1, IAD2, IBD1, IBD2, IE1, IE2
    J_phase2 = np.zeros((20, n_time_points_phase2),
                        dtype=np.float32)  # WDE11,WDE12,WDE21,WDE22,WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
    r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    max_E = np.zeros(1, dtype=np.float32)

    # Lists to hold data arrays
    l_res_rates = (r_phase1, I_phase1, r_phase2, I_phase2, set_phase2, r_phase3, max_E)
    l_res_weights = (J_exc_phase1, J_phase2)

    # The flags for activating the following plasticity mechanisms in the given order: Hebbian learning, three-factor Hebbian learning,
    # adaptive set point, E-to-E scaling, P-to-E scaling, S-to-E scaling

    flags_theta = (1, 1)

    for flags in flags_list:
        id, title = determine_name(flags)
        name = 'Case' + id + '_' + str(hour_sim) + 'h' # + '_k' + str(K).replace(".","") + '_td' + str(g_top_down_to_S)
        print('*****', title, '*****')

        if run_simulation:

            print('Simulation started.')
            print('\n')
            model_3_compartmental_v3(delta_t, sampling_rate, l_res_rates, l_res_weights, sim_duration, weights, g, g_stim,
                                  stim_times, taus, K, rheobases, lambdas, flags=flags,flags_theta=flags_theta)

            idx_av_threshold = int(15 * (1 / delta_t) * (1 / sampling_rate_stim))
            # av_threshold = r_phase1[1][idx_av_threshold] * 1.15 #it is defined with an extra 15% for old reason. not required anymore
            av_threshold = r_phase1[1][idx_av_threshold]

            if save_results:
                l_results = [l_time_points_stim, l_time_points_phase2, delta_t, sampling_rate, l_res_rates,
                             l_res_weights, av_threshold, stim_times, stim_duration, sim_duration]

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
            plot_all_3_compartmental([l_time_points_stim, l_time_points_phase2], l_res_rates, l_res_weights,
                                      av_threshold, stim_times,modulation_SST, dir_plot + name, hour_sim, format='.png', scale_y=False)
            plot_all_3_compartmental([l_time_points_stim, l_time_points_phase2], l_res_rates, l_res_weights,
                                      av_threshold, stim_times, modulation_SST, dir_plot + name, hour_sim, format='.pdf')



def plot_testing_at_regular_intervals(flags_list, flags_theta=(1,1), dir_data=r'\figures\data\\', dir_plot=r'\figures\\',
                                      K=0.25, flag_only_S_on=False, run_simulation=True,
                                      save_results = False, plot_results=False,modulation_SST=0):

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
    os.makedirs(dir_data, exist_ok=True)
    os.makedirs(dir_plot, exist_ok=True)
    hour_sims = np.arange(48) + 1

    for flags in flags_list:
        id, title = determine_name(flags)
        name = 'Case' + id + '_test_every_h' + '_k' + str(K).replace(".","")

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
                if modulation_SST == 0:
                    g_stim_S = np.array([(0, 0), (0, 0)])
                elif modulation_SST > 0:
                    g_stim_S = np.array([(0.5, 0), (0, 0.5)])
                elif modulation_SST < 0:
                    g_stim_S = np.array([(-0.5, 0), (0, -0.5)])
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
                back_inputs = (g_E, g_P, g_S)

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
                l_results = [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim,l_res_weights] #added weights for analysis with Kris

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

            [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim,l_res_weights] = l_results

        if plot_results:
            print('Plotting the results.')
            change_in_reactivation_every_h_vslides(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.png')
            change_in_reactivation_every_h_vslides(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.pdf')


#this function is a generalized version of the one above. this one, with the right flags, is the only one necessary. For clarity, they are separated
def plot_testing_at_regular_intervals_timescales(flags_list, flags_theta=(1,1), dir_data=r'\figures\data\\', dir_plot=r'\figures\\',
                                      K=0.25, flag_only_S_on=False, run_simulation=True,
                                      save_results = False, plot_results=False,modulation_SST=0):

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
    os.makedirs(dir_data, exist_ok=True)
    os.makedirs(dir_plot, exist_ok=True)
    hour_sims = np.arange(48) + 1
    tau_theta_list = (np.arange(22, 31,2) * 3600).tolist()
    # tau_beta_list = (np.arange(16, 37) * 3600).tolist()
    tau_beta_list  = [30*3600]
    # tau_theta_list = [12*3600]
    # tau_beta_list  = [0.01*3600]
    # tau_theta_list = (np.arange(24-12, 24+12) * 3600).tolist()
    # tau_beta_list = (np.arange(28-12, 28+12) * 3600).tolist()
    # tau_theta_list = [6*3600, 26*3600, 260*3600]
    # tau_beta_list  = [1e-20*3600, 1e20*3600]
    # tau_theta_list = [6*3600, 26*3600, 260*3600]
    # tau_beta_list  = [1e-20*3600, 1e20*3600]
    for tau_beta in tau_beta_list:
        for tau_theta in tau_theta_list:
            for flags in flags_list:
                id, title = determine_name(flags)
                name = 'Case' + id + '_test_every_h' + '_k' + str(K).replace(".","") + '_theta_' + str(int(tau_theta/3600)).replace(".","") + '_beta_' + str(int(tau_beta/3600)).replace(".","") 

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
                        if modulation_SST == 0:
                            g_stim_S = np.array([(0, 0), (0, 0)])
                        elif modulation_SST > 0:
                            g_stim_S = np.array([(0.5, 0), (0, 0.5)])
                        elif modulation_SST < 0:
                            g_stim_S = np.array([(-0.5, 0), (0, -0.5)])
                        g_stim = (g_stim_E, g_stim_P, g_stim_S)

                        # Time constants
                        tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
                        tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
                        tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
                        tau_hebb = 240  # time constant of three-factor Hebbian learning in seconds(2min)
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
                        back_inputs = (g_E, g_P, g_S)

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
                        l_results = [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim,l_res_weights] #added weights for analysis with Kris

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

                    [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim,l_res_weights] = l_results

                if plot_results:
                    print('Plotting the results.')
                     
                    os.makedirs(dir_plot + 'theta_' + str(int(tau_theta/3600)).replace(".","") + '_beta_' + str(int(tau_beta/3600)).replace(".","") + "/", exist_ok=True)
                    change_in_reactivation_every_h_vslides(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                                dir_plot + 'theta_' + str(int(tau_theta/3600)).replace(".","") + '_beta_' + str(int(tau_beta/3600)).replace(".","") + "/" + name, flag_only_S_on=flag_only_S_on, format='.png')
                    change_in_reactivation_every_h_vslides(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                                dir_plot + 'theta_' + str(int(tau_theta/3600)).replace(".","") + '_beta_' + str(int(tau_beta/3600)).replace(".","") + "/" + name, flag_only_S_on=flag_only_S_on, format='.pdf')



def plot_testing_at_regular_intervals_dendrites_v3(flags_list, flags_theta=(1,1), dir_data=r'\figures\data\\', dir_plot=r'\figures\\', flag_only_S_on=False, modulation_SST=0, run_simulation=True,
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
    os.makedirs(dir_data, exist_ok=True)
    os.makedirs(dir_plot, exist_ok=True)
    hour_sims = np.arange(48) + 1
    # hour_sims = np.array([1, 4, 24, 48], dtype=int)
    # K parameter in target regulator equation, it tunes the steady state value of target activity and its regulator
    K = 0.25

    for flags in flags_list:
        id, title = determine_name(flags)
        name = 'Case' + id + '_test_every_h' + '_k' + str(K).replace(".","") + '_td'

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
                g_stim_E = np.array([(2.5, 0), (0, 2.5)])
                g_stim_P = np.array([(0.5, 0), (0, 0.5)])
                if modulation_SST == 0:
                    g_stim_S = np.array([(0, 0), (0, 0)])
                elif modulation_SST > 0:
                    g_stim_S = np.array([(1.25, 0), (0, 1.25)])
                elif modulation_SST < 0:
                    g_stim_S = np.array([(-1.25, 0), (0, -1.25)])
                g_stim = (g_stim_E, g_stim_P, g_stim_S)

                # Time constants
                tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
                tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
                tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
                tau_dend = tau_E
                tau_hebb = 120  # time constant of three-factor Hebbian learning in seconds(2min)
                tau_theta = 24 * (60 * 60)  # time constant of target activity in seconds(24h)
                tau_beta = 28 * (60 * 60)  # time constant of target activity regulator in seconds(28h)
                tau_scaling_E = 2.5 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
                tau_scaling_P = 6.5 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
                tau_scaling_S = 2.5 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
                taus = (tau_E, tau_P, tau_S, tau_dend, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

                # Rheobases (minimum input needed for firing rates to be above zero)
                rheobase_E, rheobase_P, rheobase_S, rheobase_A,rheobase_B  = 1, 1.5, 1.5, 3, 9
                rheobases = (rheobase_E, rheobase_P, rheobase_S, rheobase_A,rheobase_B)

                # Background inputs
                g_AD = 4
                g_BD = 6
                g_E = 0
                g_P = 4
                g_S = 3.2
                back_inputs = (g_AD, g_BD, g_E, g_P, g_S)

                # Constant that define the contribution of each current
                lambda_AD = 0.4 # in stronger lambda config it is 0.5
                lambda_BD = 0.3 # in stronger lambda config it is 0.3
                lambdas = (lambda_AD, lambda_BD)

                # Initial conditions for plastic weights
                w_EP_within = 0.6; w_EP_cross = 0.18
                w_DS_within = 0.4; w_DS_cross = 0.18
                w_DE_within = 0.5; w_DE_cross = 0.3
                w_EE_within = 0.5; w_EE_cross = 0.3

                # Weights
                w_PE_within = 0.35; w_PE_cross = 0.10
                w_PP_within = 0.20; w_PP_cross = 0.10
                w_PS_within = 0.30; w_PS_cross = 0.10
                w_SE_within = 0.15; w_SE_cross = 0.10

                weights = (w_DE_within, w_EE_within, w_EP_within, w_DS_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
                        w_DE_cross,  w_EE_cross,  w_EP_cross,  w_DS_cross,  w_PE_cross,  w_PP_cross,  w_PS_cross,  w_SE_cross)

                # Arrays created to hold data
                r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                I_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # IAD1, IAD2, IBD1, IBD2, IE1, IE2
                J_exc_phase1 = np.zeros((8, n_time_points_stim), dtype=np.float32)  # WDE11,WDE12,WDE21,WDE22,WEE11,WEE12,WEE21,WEE22
                r_phase2 = np.zeros((6, n_time_points_phase2), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                set_phase2 = np.zeros((12, n_time_points_phase2), dtype=np.float32) # thetaDD1,thetaDD2,thetaBD1,thetaBD2,thetaE1,thetaE2,betaAD1,betaAD2,betaBD1,betaBD2,betaE1,betaE2
                I_phase2 = np.zeros((6, n_time_points_phase2), dtype=np.float32) # IAD1, IAD2, IBD1, IBD2, IE1, IE2
                J_phase2 = np.zeros((20, n_time_points_phase2),
                                    dtype=np.float32)  # WDE11,WDE12,WDE21,WDE22,WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
                r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                max_E = np.zeros(1, dtype=np.float32)

                # Lists to hold data arrays
                l_res_rates = (r_phase1, I_phase1, r_phase2, I_phase2, set_phase2, r_phase3, max_E)
                l_res_weights = (J_exc_phase1, J_phase2)

                model_3_compartmental_v3(delta_t, sampling_rate, l_res_rates, l_res_weights, int(30 * (1 / delta_t)), weights, back_inputs, g_stim,
                        stim_times, taus, K, rheobases, lambdas, flags=(0,0,0,0,0,0),flags_theta=flags_theta)

                idx_av_threshold = int(15 * (1 / delta_t) * (1 / sampling_rate_stim))
                # av_threshold = r_phase1[1][idx_av_threshold] * 1.15 #it is defined with an extra 15% for old reason. not required anymore
                av_threshold = r_phase1[1][idx_av_threshold]

                model_3_compartmental_v3(delta_t, sampling_rate, l_res_rates, l_res_weights, int(sim_duration * (1 / delta_t)), weights, back_inputs, g_stim,
                                  stim_times, taus, K, rheobases, lambdas, flags=flags,flags_theta=flags_theta)
                # print(r_phase3)
                l_delta_rE1.append(np.max(r_phase3[0][int(stim_times[0][0] * (1 / (delta_t * sampling_rate_stim))):
                                               int(stim_times[0][1] * (1 / (delta_t * sampling_rate_stim)))]).copy())

                print('Simulation of ' + str(hour_sim) + ' hours is completed')
            if save_results:
                l_results = [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim,l_res_weights] #added weights for analysis with Kris

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

            [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim,l_res_weights] = l_results

        if plot_results:
            print('Plotting the results.')
            change_in_reactivation_every_h_vslides(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.png')
            change_in_reactivation_every_h_vslides(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.pdf')


def plot_testing_at_regular_intervals_weights(ww_weights,flags_list, plastic_flag, flags_theta=(1,1), dir_data=r'\figures\data\\', dir_plot=r'\figures\\',
                                      K=0.25, flag_only_S_on=False, run_simulation=True,
                                      save_results = False, plot_results=False,modulation_SST=0):

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
    os.makedirs(dir_data, exist_ok=True)
    os.makedirs(dir_plot, exist_ok=True)
    # print('Simulation started.')
    for flags in flags_list:
        id, title = determine_name(flags)
        name = 'Case' + id + '_test_every_h' + '_k' + str(K).replace(".","") + "_"
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
                if modulation_SST == 0:
                    g_stim_S = np.array([(0, 0), (0, 0)])
                elif modulation_SST > 0:
                    g_stim_S = np.array([(0.5, 0), (0, 0.5)])
                elif modulation_SST < 0:
                    g_stim_S = np.array([(-0.5, 0), (0, -0.5)])
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
                back_inputs = (g_E, g_P, g_S)

                if plastic_flag == True: #I want to modify plastic weights
                    # Initial conditions for plastic weights
                    (w_EP_within, w_EP_cross, w_ES_within, w_ES_cross, w_EE_within, w_EE_cross) = ww_weights

                    # Weights
                    w_PE_within = 0.3; w_PE_cross = 0.1
                    w_PP_within = 0.2; w_PP_cross = 0.1
                    # w_PS_within = 0.3; w_PS_cross = 0.1
                    # w_SE_within = 0.4; w_SE_cross = 0.1
                    #strong_connection version
                    w_PS_within = 0.95; w_PS_cross = 0.1
                    w_SE_within = 0.1; w_SE_cross = 0.1

                    weights = (w_EE_within, w_EP_within, w_ES_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
                            w_EE_cross, w_EP_cross, w_ES_cross, w_PE_cross, w_PP_cross, w_PS_cross, w_SE_cross)
                else:
                    # Initial conditions for plastic weights
                    (w_PE_within,w_PP_within,w_PS_within, w_SE_within) = ww_weights

                    w_PE_cross = 0.1
                    w_PP_cross = 0.1
                    w_PS_cross = 0.1
                    w_SE_cross = 0.1

                    w_EP_within = 0.91; w_EP_cross = 0.41
                    w_ES_within = 0.51; w_ES_cross = 0.31
                    w_EE_within = 0.51; w_EE_cross = 0.51

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
                l_results = [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim,l_res_weights]

                # Open a file and save
                with open(dir_data + name + '.pkl', 'wb') as file:
                    # A new file will be created
                    pickle.dump(l_results, file)

        else:
            # Open the file and read
            with open(dir_data + name + '.pkl', 'rb') as file:
                l_results = pickle.load(file)
            print('Data is read.')

            [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim,l_res_weights] = l_results

        #it doesn't go through here
        if plot_results:
            print('Plotting the results.')
            change_in_reactivation_every_h(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.png')
            change_in_reactivation_every_h(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.pdf')
    print("Data for", '_'.join(str(weight).replace(".", "") for weight in ww_weights), "is saved\n")
