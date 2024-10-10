#   python3 /home/ge74coy/mnt/naspersonal/Code/synaptic_scaling/main_for_paper.py 

import os
# Current directory is defined
directory = os.getcwd() + "/"
os.chdir(directory)

from model_analysis import *
from plotting_functions import *

# Directories for plotting are defined
dir_data = directory + "data/"
dir_plot = dir_data + "figures/"

run_flag_cont=0 #flag to either run or load the simulutation to get the CIR plots
run_flag_discrete=0 #flag to either run or load the simulutation to get the 4/24/48h plots

plot_flag = 1 #flag if you want your results to be printed and saved



##### Plotting the figures

### Hebbian learning, the third factor in the three-factor Hebbian learning, and adaptive set-point are active in all figures
hebbian_flag, three_factor_flag, adaptive_set_point_flag= 1, 1, 1



### Plotting Figure 2
# Initialize the settings of the simulation, all plasticity mechanisms are active for the full model
E_scaling_flag = 1
P_scaling_flag = 1
S_scaling_flag = 1
flags_full = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

flags_list = [flags_full] #this is a list of tuples. You can add multiple tuples and run multiple conditions

analyze_model(4,  flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure2/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(24, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure2/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure2/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure2/",
                                  run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)



### Plotting Figure 3
# K parameter is set to different values for the full model

analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure3/", K=0,
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure3/", K=0.5,
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)

plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure3/", K=0,
                                  run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure3/", K=0.5,
                                  run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)



### Plotting Figure 4
# All synatic scaling mechanisms are blocked
E_scaling_flag = 0
P_scaling_flag = 0
S_scaling_flag = 0
flags_no_scaling = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

flags_list = [flags_no_scaling]

analyze_model(4, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure4/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure4/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure4/",
                                  run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)



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

flags_list = [flags_E_off, flags_P_off, flags_S_off]

analyze_model(4, flags_list[0:1], dir_data = dir_data, dir_plot = dir_plot + "figure5/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(24, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure5/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure5/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure5/",
                                  run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)



### Plotting Figure 6
# Turning on only the flag of E-to-E scaling
E_scaling_flag = 1
P_scaling_flag = 0
S_scaling_flag = 0
flags_only_E_on = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

# Turning on only the flag of PV-to-E scaling
E_scaling_flag = 0
P_scaling_flag = 1
S_scaling_flag = 0
flags_only_P_on = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

# Turning on only the flag of SST-to-E scaling
E_scaling_flag = 0
P_scaling_flag = 0
S_scaling_flag = 1
flags_only_S_on = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

flags_list = [flags_only_E_on, flags_only_P_on, flags_only_S_on]

analyze_model(4, flags_list[0:2], dir_data = dir_data, dir_plot = dir_plot + "figure6/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(24, flags_list[0:2], dir_data = dir_data, dir_plot = dir_plot + "figure6/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
plot_testing_at_regular_intervals(flags_list[0:2], dir_data = dir_data, dir_plot = dir_plot + "figure6/",
                                  run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)

# Plotting margins are different for only SST-to-E scaling on case, thus the related flag is set to True
analyze_model(4, flags_list[2:], dir_data = dir_data, dir_plot = dir_plot + "figure6/",
              flag_only_S_on=True, run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(24, flags_list[2:], dir_data = dir_data, dir_plot = dir_plot + "figure6/",
              flag_only_S_on=True, run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
plot_testing_at_regular_intervals(flags_list[2:], dir_data = dir_data, dir_plot = dir_plot + "figure6/",
                                  flag_only_S_on=True, run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)

# analyze_model(48, flags_list[0:2], dir_data = dir_data, dir_plot = dir_plot + "figure6/",
#               run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
# analyze_model(48, flags_list[2:], dir_data = dir_data, dir_plot = dir_plot + "figure6/",
#               flag_only_S_on=True, run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)

### Plotting the summary figure of CIR for all cases (figure number is not determined yet)
# The simulation of testing the model at every hour is already run in the previous lines. This function reads the
# data for each case and plots the summary graph.
plot_all_cases_CIR(dir_data=dir_data, dir_plot=dir_plot)