# conda activate
# python3 /home/ge74coy/mnt/naspersonal/Code/synaptic_scaling/main_for_paper.py 
# python3 /home/ge74coy/Desktop/synaptic_scaling/final_before_onlyfiles/main_for_paper.py


import os
from model_analysis import *
from plotting_functions import *

def get_data_directory(modulation_SST: float) -> str:
    """
    Return the appropriate data directory for a given SST modulation
    and ensure that the directory exists.
    """
    base = ""

    if modulation_SST == 0:
        directory = base
    elif modulation_SST > 0:
        directory = os.path.join(base, "SST_modulation", "positive")
    else:
        directory = os.path.join(base, "SST_modulation", "negative")

    os.makedirs(directory, exist_ok=True)
    return directory


modulation_SST = 0
directory = get_data_directory(modulation_SST)
os.chdir(directory)

# Directories for plotting are defined
dir_data = directory + "data/"
dir_plot = dir_data + "figures/"

run_flag_cont=1
run_flag_discrete=1

plot_flag = 1


##### Plotting the figures

### Hebbian learning, the third factor in the three-factor Hebbian learning, and adaptive set-point are active in all figures
hebbian_flag, three_factor_flag, adaptive_set_point_flag= 1, 1, 1

### Plotting Figure 2 - 3 (includes also Fig. S2)
# Initialize the settings of the simulation, all plasticity mechanisms are active for the full model
E_scaling_flag = 1
P_scaling_flag = 1
S_scaling_flag = 1
flags_full = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

flags_list = [flags_full] #this is a list of tuples. You can add multiple tuples and run multiple conditions

analyze_model(4,  flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure2_3/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(24, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure2_3/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure2_3/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)

### Plotting Figure 4b
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure4/",
                                  run_simulation=run_flag_cont, save_results=1, plot_results=plot_flag)


### Plotting Figure 5
# K parameter is set to different values for the full model

analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure5/", K=0,
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure5/", K=0.5,
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)

plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure5/", K=0,
                                  run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure5/", K=0.5,
                                  run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)


### Plotting Figure 6 - SST modulation
# Initialize the settings of the simulation, all plasticity mechanisms are active for the full model
E_scaling_flag = 1
P_scaling_flag = 1
S_scaling_flag = 1
flags_full = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

flags_list = [flags_full] #this is a list of tuples. You can add multiple tuples and run multiple conditions

modulation_SST = 1
directory = get_data_directory(modulation_SST)
os.chdir(directory)


analyze_model(4,  flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure6/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag,modulation_SST=modulation_SST)
analyze_model(24, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure6/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag,modulation_SST=modulation_SST)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure6/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag,modulation_SST=modulation_SST)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure6/",
                                  run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag,modulation_SST=modulation_SST)

modulation_SST = -1
directory = get_data_directory(modulation_SST)
os.chdir(directory)


analyze_model(4,  flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure6/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag,modulation_SST=modulation_SST)
analyze_model(24, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure6/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag,modulation_SST=modulation_SST)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure6/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag,modulation_SST=modulation_SST)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure6/",
                                  run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag,modulation_SST=modulation_SST)



### Plotting Figure 7
# All synatic scaling mechanisms are blocked
E_scaling_flag = 0
P_scaling_flag = 0
S_scaling_flag = 0
flags_no_scaling = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

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

flags_list = [flags_no_scaling, flags_E_off,flags_P_off,flags_S_off]


modulation_SST = 0
directory = get_data_directory(modulation_SST)
os.chdir(directory)

analyze_model(4, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure7/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure7/",
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
plot_testing_at_regular_intervals(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure7/",
                                  run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)



### Plotting Figure 8 (and S7)
# Initialize the settings of the simulation, all plasticity mechanisms are active for the full model
E_scaling_flag = 1
P_scaling_flag = 1
S_scaling_flag = 1
flags_full = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
 E_scaling_flag, P_scaling_flag, S_scaling_flag)

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

flags_list = [flags_full,flags_E_off,flags_P_off,flags_S_off] #this is a list of tuples. You can add multiple tuples and run multiple conditions
# flags_list = [flags_E_off] #this is a list of tuples. You can add multiple tuples and run multiple conditions


analyze_model_3_compartmental_v3(4,  flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure8/", modulation_SST=modulation_SST,
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model_3_compartmental_v3(24, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure8/", modulation_SST=modulation_SST,
              run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
analyze_model_3_compartmental_v3(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure8/",modulation_SST=modulation_SST,
             run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
plot_testing_at_regular_intervals_dendrites_v3(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure8/",modulation_SST=modulation_SST,
                                   run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)



### SUPPLEMENTARY MATERIAL

#Figure S1, S3 and S6 in notebooks (analytics.ipynb, robustness.ipynb)








### extra for revisions
### Plotting Figure - timescales analysis
# # Initialize the settings of the simulation, all plasticity mechanisms are active for the full model
# E_scaling_flag = 1
# P_scaling_flag = 1
# S_scaling_flag = 1
# flags_full = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
#  E_scaling_flag, P_scaling_flag, S_scaling_flag)

# E_scaling_flag = 0
# P_scaling_flag = 1
# S_scaling_flag = 1
# flags_E_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
#  E_scaling_flag, P_scaling_flag, S_scaling_flag)

# # Turning off the flag of PV-to-E scaling
# E_scaling_flag = 1
# P_scaling_flag = 0
# S_scaling_flag = 1
# flags_P_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
#  E_scaling_flag, P_scaling_flag, S_scaling_flag)

# # Turning off the flag of SST-to-E scaling
# E_scaling_flag = 1
# P_scaling_flag = 1
# S_scaling_flag = 0
# flags_S_off = (hebbian_flag, three_factor_flag, adaptive_set_point_flag,
#  E_scaling_flag, P_scaling_flag, S_scaling_flag)

# flags_list = [flags_E_off,flags_P_off,flags_S_off] #this is a list of tuples. You can add multiple tuples and run multiple conditions
# # flags_list = [flags_full] #this is a list of tuples. You can add multiple tuples and run multiple conditions

# # analyze_model_timescales(4,  flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure_supp2/",
# #               run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
# # analyze_model_timescales(24, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure_supp2/",
# #               run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
# # analyze_model_timescales(48, flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure_supp2/",
# #               run_simulation=run_flag_discrete, save_results=1, plot_results=plot_flag)
# plot_testing_at_regular_intervals_timescales(flags_list, dir_data = dir_data, dir_plot = dir_plot + "figure_supp4_5/",
#                                   run_simulation=run_flag_cont, save_results =1, plot_results=plot_flag)
