import os
# Current directory is defined
# directory = os.getcwd() + "/"
directory = "/home/ge74coy/mnt/nasgroup/labmembers/fabioveneto/synaptic_scaling/strong_connection/"
os.chdir(directory)

from model_analysis import *
from plotting_functions import *

import sys 

def weights_run_all_together_v1(w_EP_within,w_EP_cross, w_ES_within,w_ES_cross, w_EE_within, w_EE_cross):
    plastic_flag = True

    modulation_SST = 0 # if 0 doesn't run SST modulation, if != 0 runs positive and negative modulation

    hebbian_flag, three_factor_flag, adaptive_set_point_flag= 1, 1, 1
    ##### Plotting the figures
    # w_ES_cross = w_ES_within*0.3
    # w_EP_cross = w_EP_within*0.3
    ww_weights = (w_EP_within, w_EP_cross, w_ES_within, w_ES_cross, w_EE_within, w_EE_cross)
    ### Plotting Figure 2
    # Initialize the settings of the simulation, all plasticity mechanisms are active for the full model
    E_scaling_flag = 1
    P_scaling_flag = 1
    S_scaling_flag = 1
    flags_full = (hebbian_flag, three_factor_flag, adaptive_set_point_flag, E_scaling_flag, P_scaling_flag, S_scaling_flag)

    if modulation_SST == 0:
        dir_data = os.getcwd() + "/data/"
        dir_plot = dir_data + "figures/"
        ### Plotting Figure 5
        # Turning off the flag of E-to-E scaling
        E_scaling_flag = 0
        P_scaling_flag = 1
        S_scaling_flag = 1
        flags_E_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
        E_scaling_flag, P_scaling_flag, S_scaling_flag)

        # Turning off the flag of PV-to-E scaling
        E_scaling_flag = 1
        P_scaling_flag = 0
        S_scaling_flag = 1
        flags_P_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
        E_scaling_flag, P_scaling_flag, S_scaling_flag)

        # Turning off the flag of SST-to-E scaling
        E_scaling_flag = 1
        P_scaling_flag = 1
        S_scaling_flag = 0
        flags_S_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
        E_scaling_flag, P_scaling_flag, S_scaling_flag)

        flags_list = [flags_full,flags_E_off, flags_P_off, flags_S_off]

        plot_testing_at_regular_intervals_weights(ww_weights,flags_list,plastic_flag, dir_data = dir_data, dir_plot = dir_plot,
                                        run_simulation=1, save_results =1, plot_results=0,modulation_SST=0)
    else:
        dir_data = os.getcwd() + "/SST_modulation/"
        dir_plot = dir_data + "figures/"
        flags_list = [flags_full]

        dir_data_pos = dir_data + "positive/"
        dir_data_neg = dir_data + "negative/"
        plot_testing_at_regular_intervals_weights(ww_weights,flags_list,plastic_flag, dir_data = dir_data_pos, dir_plot = dir_plot,
                                        run_simulation=1, save_results =1, plot_results=0,modulation_SST=1)
        plot_testing_at_regular_intervals_weights(ww_weights,flags_list,plastic_flag, dir_data = dir_data_neg, dir_plot = dir_plot,
                                run_simulation=1, save_results =1, plot_results=0,modulation_SST=-1)

    



    #eactly same function, checking arguments
def weights_run_all_together_v2(w_PE_within,w_PP_within,w_PS_within, w_SE_within):
    plastic_flag = False

    modulation_SST = 0

    dir_data = os.getcwd() + "/data/"
    dir_plot = dir_data + "figures/"

    hebbian_flag, three_factor_flag, adaptive_set_point_flag= 1, 1, 1
    ##### Plotting the figures
    # w_ES_cross = w_ES_within*0.3
    # w_EP_cross = w_EP_within*0.3
    ww_weights = (w_PE_within,w_PP_within,w_PS_within, w_SE_within)
    ### Plotting Figure 2
    # Initialize the settings of the simulation, all plasticity mechanisms are active for the full model
    E_scaling_flag = 1
    P_scaling_flag = 1
    S_scaling_flag = 1
    flags_full = (hebbian_flag, three_factor_flag, adaptive_set_point_flag, E_scaling_flag, P_scaling_flag, S_scaling_flag)

    if modulation_SST == 0:
        ### Plotting Figure 5
        # Turning off the flag of E-to-E scaling
        E_scaling_flag = 0
        P_scaling_flag = 1
        S_scaling_flag = 1
        flags_E_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
        E_scaling_flag, P_scaling_flag, S_scaling_flag)

        # Turning off the flag of PV-to-E scaling
        E_scaling_flag = 1
        P_scaling_flag = 0
        S_scaling_flag = 1
        flags_P_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
        E_scaling_flag, P_scaling_flag, S_scaling_flag)

        # Turning off the flag of SST-to-E scaling
        E_scaling_flag = 1
        P_scaling_flag = 1
        S_scaling_flag = 0
        flags_S_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
        E_scaling_flag, P_scaling_flag, S_scaling_flag)

        flags_list = [flags_full,flags_E_off, flags_P_off, flags_S_off]

        plot_testing_at_regular_intervals_weights(ww_weights,flags_list,plastic_flag, dir_data = dir_data, dir_plot = dir_plot,
                                        run_simulation=1, save_results =1, plot_results=0,modulation_SST=0)
    else:
        dir_data = os.getcwd() + "/SST_modulation/"
        dir_plot = dir_data + "figures/"
        flags_list = [flags_full]

        dir_data_pos = dir_data + "positive/"
        dir_data_neg = dir_data + "negative/"
        plot_testing_at_regular_intervals_weights(ww_weights,flags_list,plastic_flag, dir_data = dir_data_pos, dir_plot = dir_plot,
                                        run_simulation=1, save_results =1, plot_results=0,modulation_SST=1)
        plot_testing_at_regular_intervals_weights(ww_weights,flags_list,plastic_flag, dir_data = dir_data_neg, dir_plot = dir_plot,
                                run_simulation=1, save_results =1, plot_results=0,modulation_SST=-1)

def parse_parameter(parameter):
    parsed_arguments = []
    parameter_values = parameter.split(',')

    for parsed_parameter in parameter_values:
        parsed_parameter = parsed_parameter.strip()

        if parsed_parameter.lower() == 'true':
            parsed_value = True
        elif parsed_parameter.lower() == 'false':
            parsed_value = False
        else:
            try:
                parsed_value = int(parsed_parameter)
            except ValueError:
                try:
                    parsed_value = float(parsed_parameter)
                except ValueError:
                    raise ValueError(f"Invalid parameter value: {parsed_parameter}")

        parsed_arguments.append(parsed_value)

    print("Parameter values:", parsed_arguments)
    # Call the sim function with the parsed parameters
    try:
        weights_run_all_together_v1(*parsed_arguments)
    except:
        weights_run_all_together_v2(*parsed_arguments)


# Example usage of parameter values
parameter = sys.argv[1]
parse_parameter(parameter)