import numpy as np
from matplotlib import pyplot as plt
# from matplotlib import colormaps as colmaps
# from matplotlib import cm as colmaps
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerTuple

import seaborn as sns



def time_plots(t, res_rates, res_weights, av_threshold, stim_times, name, hour_sim, modulation_SST,flag_only_S_on=0, format='.pdf'):

    (l_time_points_stim, l_time_points_phase2) = t
    (r_phase1, r_phase2, r_phase3, max_E) = res_rates
    (J_EE_phase1, J_phase2) = res_weights

    # Plotting configuration
    ratio = 1.5
    figure_len, figure_width = 13 * ratio, 15.7 * ratio
    font_size_1, font_size_2 = 80 * ratio, 65 * ratio
    font_size_label = 80 * ratio
    legend_size = 50 * ratio
    line_width, tick_len = 9 * ratio, 20 * ratio
    plot_line_width = 9 * ratio
    hfont = {'fontname': 'Arial'}
    sns.set(style='ticks')

    # Line styles for dashed and dotted lines
    line_style_rb = (0, (0.05, 2.5))
    line_style_r_at = (0, (5, 5))
    
    # The colors for given variables
    color_list = ['#3276b3', '#91bce0', # rE1 and WEE11, rE2 and WEE22
                  '#C10000', '#EFABAB', # rP1 and WEP11, rP2 and WEP22
                  '#007100', '#87CB87', # rS1 and WES11, rS2 and WES22
                #   '#6600cc'] # timepoints in long simulation
                '#80808026']

    stim_applied = 1

    rE_y_labels_beta_theta = [0.5,1.0,1.5]
    for i in stim_times:
        (stim_start, stim_stop) = i

        if stim_applied == 1:
            rE1 = r_phase1[0]; rE2 = r_phase1[1]
            rE_y_labels = [0.5, 1, 1.5, 2, 2.5]  # , 3.5] #[0,5,10,15]
            rE_ymin = 0.5
            rE_ymax = 2.5
        elif stim_applied == 2:
            rE1 = r_phase3[0]; rE2 = r_phase3[1]

            # Only S on case has different margins for excitatory firing rates
            if flag_only_S_on:
                if t[-1][-1] < 5:
                    rE_ymin = 3
                    rE_ymax = 5
                    rE_y_labels = [1, 2, 3, 4, 5]  # , 3.5] #[0,5,10,15]
                else:
                    rE_ymin = 20
                    rE_ymax = 40
                    rE_y_labels = [25, 30, 35, 40]  # , 3.5] #[0,5,10,15]
            else:
                rE_ymin = 0.5
                rE_ymax = 2.5
                rE_y_labels = [0.5, 1, 1.5, 2, 2.5]  # , 3.5] #[0,5,10,15]
        if modulation_SST != 0:
            rE_ymin = 0.5
            rE_ymax = 4.5
            rE_y_labels = [0.5, 1.5, 2.5,3.5,4.5]  # , 3.5] #[0,5,10,15]
        

        ### Excitatory firing rates during stimulation ###
        xmin = 0
        xmax = stim_times[0][1] + 5

        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

        e1, = ax.plot(l_time_points_stim, rE1, color=color_list[0], linewidth=plot_line_width, label=r'$r_{E1}$')
        e2, = ax.plot(l_time_points_stim, rE2, color=color_list[1], linewidth=plot_line_width, label=r'$r_{E2}$')
        # r_at, = plt.plot(l_time_points_stim, av_threshold * np.ones_like(l_time_points_stim), dash_capstyle='round',
        #                  linestyle=line_style_r_at, color='black', linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_stim, r_phase1[0][0] * np.ones_like(l_time_points_stim), dash_capstyle='round',
                       linestyle=line_style_r_at, color='grey', linewidth=plot_line_width*1.3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([rE_ymin, rE_ymax])
        plt.yticks(rE_y_labels, fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)

        ax.legend([(e1, e2), rb], [r'$r_{E1}$, $r_{E2}$', '$r_{bs}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)
        plt.tight_layout()

        plt.savefig(name + "_stim" + str(stim_applied) + '_activity' + format)
        plt.close()

        stim_applied = stim_applied + 1



    ### Excitatory weights ###
    xmin = 0
    xmax = stim_times[0][1] + 5
    if modulation_SST != 0:
        ymin = 0.4
        ymax = 0.8
    else:
        ymin = 0.4
        ymax = 0.7
    plt.figure(figsize=(figure_width, figure_len))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

    wee11, = ax.plot(l_time_points_stim, J_EE_phase1[0], color=color_list[0], linewidth=plot_line_width)
    wee12, = ax.plot(l_time_points_stim, J_EE_phase1[1], '--', color=color_list[0], linewidth=plot_line_width)
    wee21, = ax.plot(l_time_points_stim, J_EE_phase1[2], '--',color=color_list[1], linewidth=plot_line_width)
    wee22, = ax.plot(l_time_points_stim, J_EE_phase1[3], color=color_list[1], linewidth=plot_line_width)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    if modulation_SST != 0:
        plt.yticks([0.4, 0.6,0.8], fontsize=font_size_1, **hfont)
    else:
        plt.yticks([0.4, 0.5, 0.6,0.7], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
    plt.ylabel('Weights', fontsize=font_size_label, **hfont)

    ax.legend([(wee11, wee12), (wee21, wee22)], [r'$w_{E_{1}E_{1}}$, $w_{E_{1}E_{2}}$', '$w_{E_{2}E_{1}}$, $w_{E_{2}E_{2}}$'],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              fontsize=legend_size, loc='upper left', handlelength=5)
    plt.tight_layout()

    plt.savefig(name + 'conditioning_wEE' + format)
    plt.close()



    # Plot the long term behaviour only at 48 hours
    if hour_sim >= 48:
        # Assign data to variables
        rE1 = r_phase2[0]; rE2 = r_phase2[1]
        rP1 = r_phase2[2]; rP2 = r_phase2[3]
        rS1 = r_phase2[4]; rS2 = r_phase2[5]

        J_EE11 = J_phase2[0]; J_EE22 = J_phase2[3]
        J_EP11 = J_phase2[4]; J_EP22 = J_phase2[7]
        J_DS11 = J_phase2[8]; J_DS22 = J_phase2[11]

        ### Excitatory rates in the long term ###
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        e1, = ax.plot(l_time_points_phase2, rE1, color=color_list[0], linewidth=plot_line_width)
        e2, = ax.plot(l_time_points_phase2, rE2, color=color_list[1], linewidth=plot_line_width)
        # r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
        #                  linestyle=line_style_r_at, color='black', linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_r_at, color='grey', linewidth=plot_line_width*1.3)

        ymin = rE_ymin
        ymax = rE_ymax
        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48-0.1, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([rE_ymin, rE_ymax])
        plt.yticks(rE_y_labels, fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(e1, e2), rb], [r'$r_{E1}$, $r_{E2}$', '$r_{bs}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)
        plt.tight_layout()

        plt.savefig(name + '_long_E_rates' + format)
        plt.close()



        ### PV rates in the long term ###
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 1
        ymax = 1.2
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        p1, = ax.plot(l_time_points_phase2, rP1, color=color_list[2], linewidth=plot_line_width)
        p2, = ax.plot(l_time_points_phase2, rP2, color=color_list[3], linewidth=plot_line_width)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48-0.1, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks([1, 1.1, 1.2], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(p1), (p2), ],
                  ['$r_{P1}$', '$r_{P2}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right',
                  handlelength=3)
        plt.tight_layout()

        plt.savefig(name + '_long_P_rates' + format)
        plt.close()



        ### SST rates in the long term ###
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 2
        ymax = 2.6
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        s1, = ax.plot(l_time_points_phase2, rS1, color=color_list[4], linewidth=plot_line_width)
        s2, = ax.plot(l_time_points_phase2, rS2, color=color_list[5], linewidth=plot_line_width)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48-0.1, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks([2, 2.2, 2.4, 2.6], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(s1), (s2)], ['$r_{S1}$', '$r_{S2}$'], handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper right', handlelength=3)
        plt.tight_layout()

        plt.savefig(name + '_long_S_rates' + format)
        plt.close()



        ### Set-point in the long term ###
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 0.5
        ymax=1.5
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        theta1, = ax.plot(l_time_points_phase2, r_phase2[6], color=color_list[0], linewidth=plot_line_width)
        theta2, = ax.plot(l_time_points_phase2, r_phase2[7], color=color_list[1], linewidth=plot_line_width)
        # r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
        #                  linestyle=line_style_r_at, color='black', linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_r_at, color='grey', linewidth=plot_line_width*1.3)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48-0.1, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks(rE_y_labels_beta_theta, fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point $\theta$', fontsize=font_size_label, **hfont)
        ax.legend([(theta1, theta2), rb], [r'$\theta_{E1}$, $\theta_{E2}$', '$r_{bs}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)

        plt.tight_layout()

        plt.savefig(name + '_long_thetas' + format)
        plt.close()



        ### Set-point regulator in the long term ###
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        beta1, = ax.plot(l_time_points_phase2, r_phase2[8], color=color_list[0], linewidth=plot_line_width)
        beta2, = ax.plot(l_time_points_phase2, r_phase2[9], color=color_list[1], linewidth=plot_line_width)
        # r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
        #                  linestyle=line_style_r_at, color='black', linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_r_at, color='grey', linewidth=plot_line_width*1.3)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48-0.1, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks(rE_y_labels_beta_theta, fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point regulator $\beta$', fontsize=font_size_label, **hfont)
        ax.legend([(beta1, beta2), rb], [r'$\beta_{E1}$, $\beta_{E2}$', '$r_{bs}$'],
                  handler_map={tuple:HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper left',handlelength=5)
        plt.tight_layout()

        plt.savefig(name + '_long_betas' + format)
        plt.close()



        ### All plastic within-population weights in the long term ###
        ymin = 0
        ymax = 1.8
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        mag_of_y = ax.yaxis.get_offset_text()
        mag_of_y.set_size(font_size_1)

        plt.vlines(4, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48-0.1, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        wEE1, = ax.plot(l_time_points_phase2, J_EE11, linewidth=plot_line_width, color=color_list[0])
        wEE2, = ax.plot(l_time_points_phase2, J_EE22, linewidth=plot_line_width, color=color_list[1])
        wEP1, = ax.plot(l_time_points_phase2, J_EP11, linewidth=plot_line_width, color=color_list[2])
        wEP2, = ax.plot(l_time_points_phase2, J_EP22, linewidth=plot_line_width, color=color_list[3])
        wES1, = ax.plot(l_time_points_phase2, J_DS11, linewidth=plot_line_width, color=color_list[4])
        wES2, = ax.plot(l_time_points_phase2, J_DS22, linewidth=plot_line_width, color=color_list[5])
        ax.legend([(wEE1, wEE2), (wEP1, wEP2), (wES1, wES2)],
                  [r'$w_{E_{1}E_{1}}$, $w_{E_{2}E_{2}}$', r'$w_{E_{1}P_{1}}$, $w_{E_{2}P_{2}}$',
                   r'$w_{E_{1}S_{1}}$, $w_{E_{2}S_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks([0, 0.9, 1.8], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel(r'Weights', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_weights' + format)
        plt.close()



def plot_all_3_compartmental(t, res_rates, res_weights, av_threshold, stim_times,modulation_SST, name, hour_sim, format='.svg', scale_y=True):

    (l_time_points_stim, l_time_points_phase2) = t
    (r_phase1, I_phase1, r_phase2, I_phase2, set_phase2, r_phase3, max_E) = res_rates
    (J_EE_phase1, J_phase2) = res_weights

    # plotting configuration
    ratio = 1.5
    figure_len, figure_width = 13 * ratio, 12.75 * ratio
    figure_len1, figure_width1 = 13 * ratio, 13.7 * ratio
    figure_len2, figure_width2 = 13 * ratio, 14.35 * ratio
    font_size_1, font_size_2 = 80 * ratio, 65 * ratio
    font_size_label = 80 * ratio
    legend_size = 50 * ratio
    legend_size2 = 65 * ratio
    line_width, tick_len = 9 * ratio, 20 * ratio
    marker_size = 15 * ratio
    marker_edge_width = 3 * ratio
    plot_line_width = 9 * ratio
    hfont = {'fontname': 'Arial'}
    sns.set(style='ticks')

    x_label_text = 'Time (h)'

    line_style_rb = (0, (0.05, 2.5))
    line_style_r_at = (0, (5, 5))
    # defining the colors for
    color_list = ['#3276b3', '#91bce0', # rE1 and WEE11, rE2 and WEE22
                  '#C10000', '#EFABAB', # rP1 and WEP11, rP2 and WEP22
                  '#007100', '#87CB87', # rS1 and WES11, rS2 and WES22
                  '#6600cc', # timepoints in long simulation
                  'indigo', 'mediumpurple', # I_AD1 and WDE11, I_AD2 and WDE22
                  'olive', 'darkkhaki']     # I_BD1, I_BD2

    rE_y_labels = [0.5, 1, 1.5, 2, 2.5] #, 3.5] #[0,5,10,15]
    rE_ymax = 2.5
    rE_ymin = 0.5

    stim_applied = 1

    for i in stim_times:
        (stim_start, stim_stop) = i

        if stim_applied == 1:
            rE1 = r_phase1[0]; rE2 = r_phase1[1]
            rP1 = r_phase1[2]; rP2 = r_phase1[3]
            rS1 = r_phase1[4]; rS2 = r_phase1[5]
        elif stim_applied == 2:
            rE1 = r_phase3[0]; rE2 = r_phase3[1]
            rP1 = r_phase3[2]; rP2 = r_phase3[3]
            rS1 = r_phase3[4]; rS2 = r_phase3[5]

        ######### rates ###########
        xmin = 0
        xmax = stim_times[0][1] + 5

        plt.figure(figsize=(figure_width1, figure_len1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

        # p1, = ax.plot(l_time_points_stim, rP1, color=color_list[2], linewidth=plot_line_width)
        # p2, = ax.plot(l_time_points_stim, rP2, color=color_list[3], linewidth=plot_line_width)
        # s1, = ax.plot(l_time_points_stim, rS1, color=color_list[4], linewidth=plot_line_width)
        # s2, = ax.plot(l_time_points_stim, rS2, color=color_list[5], linewidth=plot_line_width)
        e1, = ax.plot(l_time_points_stim, rE1, color=color_list[0], linewidth=plot_line_width, label=r'$r_{E1}$')
        e2, = ax.plot(l_time_points_stim, rE2, color=color_list[1], linewidth=plot_line_width, label=r'$r_{E2}$')
        # r_at, = plt.plot(l_time_points_stim, av_threshold * np.ones_like(l_time_points_stim), dash_capstyle='round',
        #                  linestyle=line_style_r_at, color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_stim, r_phase1[0][0] * np.ones_like(l_time_points_stim), dash_capstyle='round',
                       linestyle=line_style_rb, color=color_list[0], linewidth=plot_line_width*1.3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([rE_ymin, rE_ymax])
            plt.yticks(rE_y_labels, fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)

        # ax.legend([(e1, e2), (rb, r_at)], ['$r_{E1}$, $r_{E2}$', '$r_b$, $r_{at}$'],
        ax.legend([(e1, e2), (rb)], ['$r_{E1}$, $r_{E2}$', '$r_b$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)
        plt.tight_layout()

        plt.savefig(name + "_stim" + str(stim_applied) + '_activity' + format)
        plt.close()

        stim_applied = stim_applied + 1

    # ######### wAB during conditioning ###########
    # xmin = 0
    # xmax = stim_times[0][1] + 5
    # ymin = 0.3
    # ymax = 0.8
    # plt.figure(figsize=(figure_width1, figure_len1))
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(True)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     ax.spines[axis].set_linewidth(line_width)
    # plt.tick_params(width=line_width, length=tick_len)
    # plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

    # wde11, = ax.plot(l_time_points_stim, J_EE_phase1[0], color=color_list[0], linewidth=plot_line_width)
    # wde12, = ax.plot(l_time_points_stim, J_EE_phase1[1], '--', color=color_list[0], linewidth=plot_line_width)
    # wde21, = ax.plot(l_time_points_stim, J_EE_phase1[2], color=color_list[1], linewidth=plot_line_width)
    # wde22, = ax.plot(l_time_points_stim, J_EE_phase1[3], '--', color=color_list[1], linewidth=plot_line_width)

    # plt.xticks(fontsize=font_size_1, **hfont)
    # plt.yticks(fontsize=font_size_1, **hfont)

    # if scale_y:
    #     plt.ylim([ymin, ymax])
    #     plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, ], fontsize=font_size_1, **hfont)
    # plt.xlim([xmin, xmax])
    # plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
    # plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
    # plt.ylabel('Weights', fontsize=font_size_label, **hfont)

    # ax.legend([(wde11, wde12), (wde21, wde22)],
    #           [r'$w_{A_{1}B_{1}}$, $w_{A_{1}B_{2}}$', '$w_{A_{2}B_{1}}$', r'$w_{A_{2}B_{2}}$'],
    #           handler_map={tuple: HandlerTuple(ndivide=None)},
    #           fontsize=legend_size, loc='upper left', handlelength=5)
    # plt.tight_layout()

    # plt.savefig(name + 'conditioning_wAB' + format)
    # plt.close()

    # #########  wBB during conditioning  ###########
    # xmin = 0
    # xmax = stim_times[0][1] + 5
    # ymin = 0.3
    # ymax = 0.8
    # plt.figure(figsize=(figure_width1, figure_len1))
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(True)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     ax.spines[axis].set_linewidth(line_width)
    # plt.tick_params(width=line_width, length=tick_len)
    # plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

    # wee11, = ax.plot(l_time_points_stim, J_EE_phase1[4], color=color_list[0], linewidth=plot_line_width)
    # wee12, = ax.plot(l_time_points_stim, J_EE_phase1[5], '--', color=color_list[0], linewidth=plot_line_width)
    # wee21, = ax.plot(l_time_points_stim, J_EE_phase1[6], color=color_list[1], linewidth=plot_line_width)
    # wee22, = ax.plot(l_time_points_stim, J_EE_phase1[7], '--', color=color_list[1], linewidth=plot_line_width)

    # plt.xticks(fontsize=font_size_1, **hfont)
    # plt.yticks(fontsize=font_size_1, **hfont)

    # if scale_y:
    #     plt.ylim([ymin, ymax])
    #     plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, ], fontsize=font_size_1, **hfont)
    # plt.xlim([xmin, xmax])
    # plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
    # plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
    # plt.ylabel('Weights', fontsize=font_size_label, **hfont)

    # ax.legend([(wee11, wee12), (wee21, wee22)],
    #           [r'$w_{B_{1}B_{1}}$, $w_{B_{1}B_{2}}$', '$w_{B_{2}B_{1}}$', r'$w_{B_{2}B_{2}}$'],
    #           handler_map={tuple: HandlerTuple(ndivide=None)},
    #           fontsize=legend_size, loc='upper left', handlelength=5)
    # plt.tight_layout()

    # plt.savefig(name + 'conditioning_wBB' + format)
    # plt.close()

    ######### wAB & wBB during conditioning in one plot ###########
    xmin = 0
    xmax = stim_times[0][1] + 5
    ymin = 0.2
    ymax = 0.8

    plt.figure(figsize=(figure_width1, figure_len1))
    ax = plt.gca()

    # Axes styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    ax.tick_params(width=line_width, length=tick_len)
    ax.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

    # wAE curves
    wae11, = ax.plot(l_time_points_stim, J_EE_phase1[0], color=color_list[7], linewidth=plot_line_width)
    wae12, = ax.plot(l_time_points_stim, J_EE_phase1[1], '--', color=color_list[7], linewidth=plot_line_width)
    wae21, = ax.plot(l_time_points_stim, J_EE_phase1[2], '--', color=color_list[8], linewidth=plot_line_width)
    wae22, = ax.plot(l_time_points_stim, J_EE_phase1[3], color=color_list[8], linewidth=plot_line_width)

    # wBE curves
    wbe11, = ax.plot(l_time_points_stim, J_EE_phase1[4], color=color_list[9], linewidth=plot_line_width)
    wbe12, = ax.plot(l_time_points_stim, J_EE_phase1[5], '--', color=color_list[9], linewidth=plot_line_width)
    wbe21, = ax.plot(l_time_points_stim, J_EE_phase1[6], '--', color=color_list[10], linewidth=plot_line_width)
    wbe22, = ax.plot(l_time_points_stim, J_EE_phase1[7], color=color_list[10], linewidth=plot_line_width)

    # Limits and labels
    if scale_y:
        ax.set_ylim([ymin, ymax])
        ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlim([xmin, xmax])
    ax.set_xticks([5, 10, 15, 20, 25])
    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    ax.set_xlabel('Time (s)', fontsize=font_size_label, **hfont)
    ax.set_ylabel('Weights', fontsize=font_size_label, **hfont)

    # Combined legend
    ax.legend(
        [wae11, wae12, wae21, wae22, wbe11, wbe12, wbe21, wbe22],
        [r'$w_{A_{1}E_{1}}$', r'$w_{A_{1}E_{2}}$',
        r'$w_{A_{2}E_{1}}$', r'$w_{A_{2}E_{2}}$',
        r'$w_{B_{1}E_{1}}$', r'$w_{B_{1}E_{2}}$',
        r'$w_{B_{2}E_{1}}$', r'$w_{B_{2}E_{2}}$'],
        fontsize=legend_size, loc='upper left', handlelength=5
    )

    plt.tight_layout()
    plt.savefig(name + 'conditioning_wAB_wBB_combined' + format)
    plt.close()

    ######### conditioning currents ###########
    xmin = 0
    xmax = stim_times[0][1] + 5
    ymin = 0
    ymax = 12
    plt.figure(figsize=(figure_width, figure_len))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

    I_A1, = ax.plot(l_time_points_stim, I_phase1[0], color=color_list[7], linewidth=plot_line_width)
    I_A2, = ax.plot(l_time_points_stim, I_phase1[1], color=color_list[8], linewidth=plot_line_width)
    I_B1, = ax.plot(l_time_points_stim, I_phase1[2], color=color_list[9], linewidth=plot_line_width)
    I_B2, = ax.plot(l_time_points_stim, I_phase1[3], color=color_list[10],linewidth=plot_line_width)
    I_E1, = ax.plot(l_time_points_stim, I_phase1[4], color=color_list[0], linewidth=plot_line_width)
    I_E2, = ax.plot(l_time_points_stim, I_phase1[5], color=color_list[1], linewidth=plot_line_width)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    if scale_y:
        plt.ylim([ymin, ymax])
        plt.yticks([0,6,12], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
    plt.ylabel('Currents', fontsize=font_size_label, **hfont)
    # ax.legend([(I_A1, I_A2), (I_B1, I_B2), (I_E1, I_E2), rb, r_at],
    ax.legend([(I_A1, I_A2), (I_B1, I_B2), (I_E1, I_E2), rb],
              [r'$I_{A1}$, $I_{A2}$', r'$I_{B1}$, $I_{BD2}$', r'$I_{E1}$, $I_{E2}$'],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              fontsize=legend_size, loc='upper left', handlelength=5)
    plt.tight_layout()
    plt.savefig(name + 'conditioning_currents' + format)
    plt.close()

    # plot the long term behaviour only at 48 hours
    if hour_sim >= 4.8:
        l_rE1 = r_phase2[0]; l_rE2 = r_phase2[1]
        l_rP1 = r_phase2[2]; l_rP2 = r_phase2[3]
        l_rS1 = r_phase2[4]; l_rS2 = r_phase2[5]

        l_thetaA1, l_thetaA2, l_thetaB1, l_thetaB2, l_thetaE1, l_thetaE2 = set_phase2[0], set_phase2[1], set_phase2[2], set_phase2[3], set_phase2[4], set_phase2[5]
        l_betaA1, l_betaA2, l_betaB1, l_betaB2, l_betaE1, l_betaE2 = set_phase2[6], set_phase2[7], set_phase2[8], set_phase2[9], set_phase2[10], set_phase2[11]
        l_IA1, l_IA2, l_IB1, l_IB2, l_IE1, l_IE2 = I_phase2[0], I_phase2[1], I_phase2[2], I_phase2[3], I_phase2[4], I_phase2[5]


        l_J_AB11, l_J_AB12, l_J_AB21, l_J_AB22 = J_phase2[0], J_phase2[1], J_phase2[2], J_phase2[3]
        l_J_BB11, l_J_BB12, l_J_BB21, l_J_BB22 = J_phase2[4], J_phase2[5], J_phase2[6], J_phase2[7]
        l_J_EP11, l_J_EP12, l_J_EP21, l_J_EP22 = J_phase2[8], J_phase2[9], J_phase2[10],J_phase2[11]
        l_J_AS11, l_J_AS12, l_J_AS21, l_J_AS22 = J_phase2[12],J_phase2[13],J_phase2[14],J_phase2[15]
        l_J_BS11, l_J_BS12, l_J_BS21, l_J_BS22 = J_phase2[16],J_phase2[17],J_phase2[18],J_phase2[19]

        # rates ALL
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 0.5
        ymax = 2.5
        plt.figure(figsize=(figure_width1, figure_len1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        # p1, = ax.plot(l_time_points_phase2, l_rP1, color=color_list[2], linewidth=plot_line_width)
        # p2, = ax.plot(l_time_points_phase2, l_rP2, color=color_list[3], linewidth=plot_line_width)
        # s1, = ax.plot(l_time_points_phase2, l_rS1, color=color_list[4], linewidth=plot_line_width)
        # s2, = ax.plot(l_time_points_phase2, l_rS2, color=color_list[5], linewidth=plot_line_width)
        e1, = ax.plot(l_time_points_phase2, l_rE1, color=color_list[0], linewidth=plot_line_width)
        e2, = ax.plot(l_time_points_phase2, l_rE2, color=color_list[1], linewidth=plot_line_width)
        # r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
        #                  linestyle=line_style_r_at, color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_rb, color=color_list[0], linewidth=plot_line_width*1.3)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0.5, 1, 1.5, 2, 2.5], fontsize=font_size_1, **hfont)
        # if modulation_SST != 0:
        #     rE_ymin = 0.5
        #     rE_ymax = 4.5
        #     rE_y_labels = [0.5, 1.5, 2.5,3.5,4.5]  # , 3.5] #[0,5,10,15]
        #     plt.ylim([rE_ymin, rE_ymax])
        #     plt.yticks(rE_y_labels, fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        # ax.legend([(e1, e2), (p1, p2), (s1, s2), rb, r_at],
        #           [r'$r_{E1}$, $r_{E2}$', '$r_{P1}$, $r_{P2}$', '$r_{S1}$, $r_{S2}$', '$r_{b}$', '$r_{at}$'],
        ax.legend([(e1, e2), rb],
                  [r'$r_{E1}$, $r_{E2}$', '$r_{b}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right',
                  handlelength=3)

        plt.tight_layout()
        plt.savefig(name + '_long_ALL_rates' + format)
        plt.close()



        # currents
        ymin = 1
        ymax = 5
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width1, figure_len1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        I_A1, = ax.plot(l_time_points_phase2, l_IA1, color=color_list[7], linewidth=plot_line_width)
        I_A2, = ax.plot(l_time_points_phase2, l_IA2, color=color_list[8], linewidth=plot_line_width)
        I_B1, = ax.plot(l_time_points_phase2, l_IB1, color=color_list[9], linewidth=plot_line_width)
        I_B2, = ax.plot(l_time_points_phase2, l_IB2, color=color_list[10],linewidth=plot_line_width)
        I_E1, = ax.plot(l_time_points_phase2, l_IE1, color=color_list[0], linewidth=plot_line_width)
        I_E2, = ax.plot(l_time_points_phase2, l_IE2, color=color_list[1], linewidth=plot_line_width)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([1, 2, 3, 4, 5], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Currents', fontsize=font_size_label, **hfont)
        ax.legend([(I_A1, I_A2), (I_B1, I_B2), (I_E1, I_E2), rb],
                  [r'$I_{A1}$, $I_{A2}$', r'$I_{B1}$, $I_{B2}$', r'$I_{E1}$, $I_{E2}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)
        plt.tight_layout()
        plt.savefig(name + '_long_currents' + format)
        plt.close()


        # thetas
        ymin = 0
        ymax = 8
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        thetaA1, = ax.plot(l_time_points_phase2, l_thetaA1, color=color_list[7], linewidth=plot_line_width)
        thetaA2, = ax.plot(l_time_points_phase2, l_thetaA2, color=color_list[8], linewidth=plot_line_width)
        thetaB1, = ax.plot(l_time_points_phase2, l_thetaB1, color=color_list[9], linewidth=plot_line_width)
        thetaB2, = ax.plot(l_time_points_phase2, l_thetaB2, color=color_list[10],linewidth=plot_line_width)
        thetaE1, = ax.plot(l_time_points_phase2, l_thetaE1, color=color_list[0], linewidth=plot_line_width)
        thetaE2, = ax.plot(l_time_points_phase2, l_thetaE2, color=color_list[1], linewidth=plot_line_width)
        # r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
        #                  linestyle=line_style_r_at, color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_rb, color=color_list[0], linewidth=plot_line_width*1.3)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0, 4,8], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point $\theta$', fontsize=font_size_label, **hfont)
        ax.legend([(thetaA1, thetaA2), (thetaB1, thetaB2), (thetaE1, thetaE2), rb],
                  [r'$\theta_{A1}$, $\theta_{A2}$', r'$\theta_{B1}$, $\theta_{B2}$', r'$\theta_{E1}$, $\theta_{E2}$', '$r_{bs}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)

        plt.tight_layout()

        plt.savefig(name + '_long_thetas' + format)
        plt.close()


        # betas
        ymin = 0
        ymax = 8
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        betaA1, = ax.plot(l_time_points_phase2, l_betaA1, color=color_list[7], linewidth=plot_line_width)
        betaA2, = ax.plot(l_time_points_phase2, l_betaA2, color=color_list[8], linewidth=plot_line_width)
        betaB1, = ax.plot(l_time_points_phase2, l_betaB1, color=color_list[9], linewidth=plot_line_width)
        betaB2, = ax.plot(l_time_points_phase2, l_betaB2, color=color_list[10],linewidth=plot_line_width)
        betaE1, = ax.plot(l_time_points_phase2, l_betaE1, color=color_list[0], linewidth=plot_line_width)
        betaE2, = ax.plot(l_time_points_phase2, l_betaE2, color=color_list[1], linewidth=plot_line_width)
        # r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
        #                  linestyle=line_style_r_at, color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_rb, color=color_list[0], linewidth=plot_line_width*1.3)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0, 4,8], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point regulator $\beta$', fontsize=font_size_label, **hfont)
        ax.legend([(betaA1, betaA2), (betaB1, betaB2), (betaE1, betaE2), rb],
                  [r'$\beta_{A1}$, $\beta_{A2}$', r'$\beta_{B1}$, $\beta_{B2}$', r'$\beta_{E1}$, $\beta_{E2}$', '$r_{bs}$'],
                  handler_map={tuple:HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper left',handlelength=5)
        plt.tight_layout()

        plt.savefig(name + '_long_betas' + format)
        plt.close()



        ######### All plastic weights during scaling ##########

        # wAE
        ymin = 0
        ymax = 0.9
        plt.figure(figsize=(figure_width1, figure_len1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        mag_of_y = ax.yaxis.get_offset_text()
        mag_of_y.set_size(font_size_1)

        wAB11, = ax.plot(l_time_points_phase2, l_J_AB11, linewidth=plot_line_width, color=color_list[7])
        wAB12, = ax.plot(l_time_points_phase2, l_J_AB12, '--', linewidth=plot_line_width, color=color_list[7])
        wAB21, = ax.plot(l_time_points_phase2, l_J_AB21, '--', linewidth=plot_line_width, color=color_list[8])
        wAB22, = ax.plot(l_time_points_phase2, l_J_AB22, linewidth=plot_line_width, color=color_list[8])
        ax.legend([(wAB11, wAB12), (wAB21, wAB22)],
                  [r'$w_{A_{1}E_{1}}$, $w_{A_{1}E_{2}}$', r'$w_{A_{2}E_{1}}$, $w_{A_{2}E_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)
        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0.1, 0.5,0.9], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'$W_{AE}$ connections', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_W_AB' + format)
        plt.close()

        # wBE
        ymin = 0.0
        ymax = 0.9
        plt.figure(figsize=(figure_width1, figure_len1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        mag_of_y = ax.yaxis.get_offset_text()
        mag_of_y.set_size(font_size_1)

        wBB11, = ax.plot(l_time_points_phase2, l_J_BB11, linewidth=plot_line_width, color=color_list[0])
        wBB12, = ax.plot(l_time_points_phase2, l_J_BB12, '--', linewidth=plot_line_width, color=color_list[0])
        wBB21, = ax.plot(l_time_points_phase2, l_J_BB21, '--', linewidth=plot_line_width, color=color_list[1])
        wBB22, = ax.plot(l_time_points_phase2, l_J_BB22, linewidth=plot_line_width, color=color_list[1])
        ax.legend([(wBB11, wBB12), (wBB21, wBB22)],
                  [r'$w_{B_{1}E_{1}}$, $w_{B_{1}E_{2}}$', r'$w_{B_{2}E_{1}}$, $w_{B_{2}E_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0.1, 0.5,0.9], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'$W_{BE}$ connections', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_W_BB' + format)
        plt.close()

        # wEP
        ymin = 0.0
        ymax = 0.9
        plt.figure(figsize=(figure_width2, figure_len2))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        mag_of_y = ax.yaxis.get_offset_text()
        mag_of_y.set_size(font_size_1)

        wEP11, = ax.plot(l_time_points_phase2, l_J_EP11, linewidth=plot_line_width, color=color_list[2])
        wEP12, = ax.plot(l_time_points_phase2, l_J_EP12, '--', linewidth=plot_line_width, color=color_list[2])
        wEP21, = ax.plot(l_time_points_phase2, l_J_EP21, '--', linewidth=plot_line_width, color=color_list[3])
        wEP22, = ax.plot(l_time_points_phase2, l_J_EP22, linewidth=plot_line_width, color=color_list[3])
        ax.legend([(wEP11, wEP12), (wEP21, wEP22)],
                  [r'$w_{E_{1}P_{1}}$, $w_{E_{1}P_{2}}$', r'$w_{E_{2}P_{1}}$, $w_{E_{2}P_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0.1, 0.5,0.9], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'$W_{EP}$ connections', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_W_EP' + format)
        plt.close()

        # wAS
        ymin = 0
        ymax = 0.9
        plt.figure(figsize=(figure_width2, figure_len2))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        mag_of_y = ax.yaxis.get_offset_text()
        mag_of_y.set_size(font_size_1)

        wAS11, = ax.plot(l_time_points_phase2, l_J_AS11, linewidth=plot_line_width, color=color_list[4])
        wAS12, = ax.plot(l_time_points_phase2, l_J_AS12, '--', linewidth=plot_line_width, color=color_list[4])
        wAS21, = ax.plot(l_time_points_phase2, l_J_AS21, '--', linewidth=plot_line_width, color=color_list[5])
        wAS22, = ax.plot(l_time_points_phase2, l_J_AS22, linewidth=plot_line_width, color=color_list[5])
        ax.legend([(wAS11, wAS12), (wAS21, wAS22)],
                  [r'$w_{A_{1}S_{1}}$, $w_{A_{1}S_{2}}$', r'$w_{A_{2}S_{1}}$, $w_{A_{2}S_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0.1, 0.5,0.9], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'$W_{AS}$ connections', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_W_AS' + format)
        plt.close()

        ymin = 0
        ymax = 0.9
        plt.figure(figsize=(figure_width2, figure_len2))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        mag_of_y = ax.yaxis.get_offset_text()
        mag_of_y.set_size(font_size_1)

        wAS11, = ax.plot(l_time_points_phase2, l_J_BS11, linewidth=plot_line_width, color=color_list[4])
        wAS12, = ax.plot(l_time_points_phase2, l_J_BS12, '--', linewidth=plot_line_width, color=color_list[4])
        wAS21, = ax.plot(l_time_points_phase2, l_J_BS21, '--', linewidth=plot_line_width, color=color_list[5])
        wAS22, = ax.plot(l_time_points_phase2, l_J_BS22, linewidth=plot_line_width, color=color_list[5])
        ax.legend([(wAS11, wAS12), (wAS21, wAS22)],
                  [r'$w_{B_{1}S_{1}}$, $w_{B_{1}S_{2}}$', r'$w_{B_{2}S_{1}}$, $w_{B_{2}S_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0.1, 0.5,0.9], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'$W_{BS}$ connections', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_W_BS' + format)
        plt.close()

        #ALL weights together
        ymin_all, ymax_all = 0.0, 0.9

        plt.figure(figsize=(figure_width2, figure_len2))
        ax = plt.gca()

        # spines & ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        ax.yaxis.get_offset_text().set_size(font_size_1)

        # --- plot all curves on the SAME axes ---
        # AE (colors 7,8)
        h_AE1, = ax.plot(l_time_points_phase2, l_J_AB11,  linewidth=plot_line_width, color=color_list[7])
        h_AE2, = ax.plot(l_time_points_phase2, l_J_AB21, '--',  linewidth=plot_line_width, color=color_list[8])
        ax.plot(l_time_points_phase2, l_J_AB12,  '--', linewidth=plot_line_width, color=color_list[7])
        ax.plot(l_time_points_phase2, l_J_AB22, linewidth=plot_line_width, color=color_list[8])

        # BE (colors 0,1)
        h_BE1, = ax.plot(l_time_points_phase2, l_J_BB11,  linewidth=plot_line_width, color=color_list[0])
        h_BE2, = ax.plot(l_time_points_phase2, l_J_BB21, '--',  linewidth=plot_line_width, color=color_list[1])
        ax.plot(l_time_points_phase2, l_J_BB12,  '--', linewidth=plot_line_width, color=color_list[0])
        ax.plot(l_time_points_phase2, l_J_BB22, linewidth=plot_line_width, color=color_list[1])

        # EP (colors 2,3)
        h_EP1, = ax.plot(l_time_points_phase2, l_J_EP11,  linewidth=plot_line_width, color=color_list[2])
        h_EP2, = ax.plot(l_time_points_phase2, l_J_EP21, '--', linewidth=plot_line_width, color=color_list[3])
        ax.plot(l_time_points_phase2, l_J_EP12,  '--', linewidth=plot_line_width, color=color_list[2])
        ax.plot(l_time_points_phase2, l_J_EP22, linewidth=plot_line_width, color=color_list[3])

        # AS (colors 4,5)
        h_AS1, = ax.plot(l_time_points_phase2, l_J_AS11,  linewidth=plot_line_width, color=color_list[4])
        h_AS2, = ax.plot(l_time_points_phase2, l_J_AS21,'--',  linewidth=plot_line_width, color=color_list[5])
        ax.plot(l_time_points_phase2, l_J_AS12,  '--', linewidth=plot_line_width, color=color_list[4])
        ax.plot(l_time_points_phase2, l_J_AS22, linewidth=plot_line_width, color=color_list[5])

        # --- legends ---
        # Legend 1: which presynaptic pair (color)
        leg1 = ax.legend(
            handles=[h_AE1, h_AE2, h_BE1, h_BE2, h_EP1, h_EP2, h_AS1, h_AS2],
            labels=[
                r'$A_{1}\!\to\!E$', r'$A_{2}\!\to\!E$',
                r'$B_{1}\!\to\!E$', r'$B_{2}\!\to\!E$',
                r'$E_{1}\!\to\!P$', r'$E_{2}\!\to\!P$',
                r'$A_{1}\!\to\!S$', r'$A_{2}\!\to\!S$'
            ],
            fontsize=legend_size, loc='upper right', handlelength=3, frameon=False
        )
        ax.add_artist(leg1)

        # Legend 2: line style meaning (solid = target 1, dashed = target 2)
        h_style_solid, = ax.plot([], [], '-',  linewidth=plot_line_width, color='k')
        h_style_dash,  = ax.plot([], [], '--', linewidth=plot_line_width, color='k')
        ax.legend(
            handles=[h_style_solid, h_style_dash],
            labels=[r'$w_{\cdot\cdot 1}$', r'$w_{\cdot\cdot 2}$'],
            fontsize=legend_size, loc='upper left', frameon=False, handlelength=3
        )

        # axes labels, ticks, limits
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)
        if scale_y:
            plt.ylim([ymin_all, ymax_all])
            plt.yticks([0.1, 0.5, 0.9], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])

        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'All connections: $W_{AE},\,W_{BE},\,W_{EP},\,W_{AS}$', fontsize=font_size_label, **hfont)

        plt.tight_layout()
        plt.savefig(name + '_long_ALL' + format)
        plt.close()

        #Excitatory to dendrites
        ymin_all, ymax_all = 0.0, 0.9

        plt.figure(figsize=(figure_width2, figure_len2))
        ax = plt.gca()

        # spines & ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        ax.yaxis.get_offset_text().set_size(font_size_1)

        # AE (colors 7,8)
        h_AE1, = ax.plot(l_time_points_phase2, l_J_AB11,  linewidth=plot_line_width, color=color_list[7])
        ax.plot(l_time_points_phase2, l_J_AB12,  '--', linewidth=plot_line_width, color=color_list[7])
        h_AE2, = ax.plot(l_time_points_phase2, l_J_AB21,  linewidth=plot_line_width, color=color_list[8])
        ax.plot(l_time_points_phase2, l_J_AB22,  '--', linewidth=plot_line_width, color=color_list[8])

        # BE (colors 0,1)
        h_BE1, = ax.plot(l_time_points_phase2, l_J_BB11,  linewidth=plot_line_width, color=color_list[0])
        ax.plot(l_time_points_phase2, l_J_BB12,  '--', linewidth=plot_line_width, color=color_list[0])
        h_BE2, = ax.plot(l_time_points_phase2, l_J_BB21,  linewidth=plot_line_width, color=color_list[1])
        ax.plot(l_time_points_phase2, l_J_BB22,  '--', linewidth=plot_line_width, color=color_list[1])

        # grouped legend
        ax.legend(
            [(h_AE1, ax.lines[1]), (h_AE2, ax.lines[3]),
            (h_BE1, ax.lines[5]), (h_BE2, ax.lines[7])],
            [r'$E_{1}\!\to\!A_{1},\,E_{1}\!\to\!A_{2}$',
            r'$E_{2}\!\to\!A_{1},\,E_{2}\!\to\!A_{2}$',
            r'$E_{1}\!\to\!B_{1},\,E_{1}\!\to\!B_{2}$',
            r'$E_{2}\!\to\!B_{1},\,E_{2}\!\to\!B_{2}$'],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            fontsize=legend_size, loc='upper right', handlelength=3
        )

        # axis limits, labels, ticks
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)
        if scale_y:
            plt.ylim([ymin_all, ymax_all])
            plt.yticks([0.1, 0.5, 0.9], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])

        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'$E\!\to\!A,\;E\!\to\!B$ connections', fontsize=font_size_label, **hfont)

        plt.tight_layout()
        plt.savefig(name + '_long_W_AE_BE' + format)
        plt.close()


'''
def change_in_reactivation_every_h(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                    name, flag_only_S_on=0, format='.pdf'):

    # plotting configuration
    ratio = 1.5
    figure_len2, figure_width2 = 14.5 * ratio, 13.7 * ratio
    font_size_1, font_size_2 = 80 * ratio, 65 * ratio
    font_size_label = 80 * ratio
    legend_size = 50 * ratio
    legend_size2 = 65 * ratio
    line_width, tick_len = 9 * ratio, 20 * ratio
    marker_size = 15 * ratio
    marker_edge_width = 3 * ratio
    plot_line_width = 9 * ratio
    hfont = {'fontname': 'Arial'}
    sns.set(style='ticks')

    x_label_text = 'Time (h)'

    line_style_rb = (0, (0.05, 2.5))
    line_style_r_at = (0, (5, 5))
    # defining the colors for
    color_list = ['#3276b3', '#91bce0', # rE1 and WEE11, rE2 and WEE22
                  '#C10000', '#EFABAB', # rP1 and WEP11, rP2 and WEP22
                  '#007100', '#87CB87', # rS1 and WES11, rS2 and WES22
                  '#6600cc'] # timepoints in long simulation

    ### % Change in reactivation of E1
    xmin = 0
    xmax = l_time_points_phase2[-1]

    # Only SST-to-E scaling on case has different margins
    if flag_only_S_on:
        ymin = 1
        ymax = 10000
    else:
        ymin = 0
        ymax = 100

    plt.figure(figsize=(figure_width2, figure_len2))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    baseline_reactivation = av_threshold / 1.15
    change_in_reactivation = 100*(np.array(l_delta_rE1) - baseline_reactivation) / baseline_reactivation

    # Define the colors for the custom colormap
    cmap_colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # White, Red
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)

    # Define the TwoSlopeNorm with vcenter at 15, vmin at -10, and vmax at 100
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=15, vmax=100)

    # Normalize the change in reactivation
    values_range = norm(change_in_reactivation)  # Normalize using TwoSlopeNorm

    # Plotting and using the values to pick colors
    for i in range(len(hour_sims) - 1):
        color = cmap(values_range[i])
        ax.fill_between(hour_sims[i:i + 2], 0, change_in_reactivation[i:i + 2], color=color,
                        edgecolor=(0, 0, 0, 0))

    plt.plot(hour_sims, change_in_reactivation, color=color_list[0], linewidth=plot_line_width, label='% change')
    plt.plot(hour_sims, 15 * np.ones_like(hour_sims), dash_capstyle='round',
                     linestyle=line_style_r_at, color='black', linewidth=plot_line_width, label='Tolerance')
    plt.plot(hour_sims, np.zeros_like(hour_sims), dash_capstyle='round',
                     linestyle=line_style_rb, color='black', linewidth=plot_line_width, label='Baseline reactivation')

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.tick_params(width=line_width, length=tick_len, labelsize=font_size_1)
    # cbar.set_ticks([0, 15, 100], fontsize=font_size_1, **hfont)
    
    cbar.set_ticks([0, 15, 100])
    cbar.ax.tick_params(labelsize=font_size_1)

    cbar.outline.set_linewidth(line_width)  # Set colorbar border thickness


    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)

    if flag_only_S_on:
        plt.yscale('log')  # Set the y-axis to logarithmic scale
        plt.ylabel('% change in $r_{E1,re}$ (log)', fontsize=font_size_label, **hfont)
        ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='lower center',
                  bbox_to_anchor=(0.5, 1.05), handlelength=3)
    else:
        plt.yticks([-20, 0, 15, 50, 100], fontsize=font_size_1, **hfont)
        plt.ylabel('% change in $r_{E1,re}$', fontsize=font_size_label, **hfont)
        ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,loc='lower center', bbox_to_anchor=(0.5, 1.05), handlelength=3)

    plt.tight_layout()
    plt.savefig(name + '_delta_reactivation_color' + format)
    plt.show()
    plt.close()

'''
#variation (hopefully correct of original plot function)
def change_in_reactivation_every_h_vslides(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                    name, flag_only_S_on=0, format='.pdf'):

    # plotting configuration
    ratio = 1.5
    figure_len2, figure_width2 = 14.5 * ratio, 13.7 * ratio
    font_size_1, font_size_2 = 80 * ratio, 65 * ratio
    font_size_label = 80 * ratio
    legend_size = 50 * ratio
    legend_size2 = 65 * ratio
    line_width, tick_len = 9 * ratio, 20 * ratio
    marker_size = 15 * ratio
    marker_edge_width = 3 * ratio
    plot_line_width = 9 * ratio
    hfont = {'fontname': 'Arial'}
    sns.set(style='ticks')

    x_label_text = 'Time (h)'

    center_threshold = 0
    
    line_style_rb = (0, (0.05, 2.5))
    line_style_r_at = (0, (5, 5))
    # defining the colors for
    color_list = ['#3276b3', '#91bce0', # rE1 and WEE11, rE2 and WEE22
                  '#C10000', '#EFABAB', # rP1 and WEP11, rP2 and WEP22
                  '#007100', '#87CB87', # rS1 and WES11, rS2 and WES22
                #   '#6600cc'] # timepoints in long simulation
                '#80808026']

    ### % Change in reactivation of E1
    xmin = 0
    xmax = l_time_points_phase2[-1]

    # Only SST-to-E scaling on case has different margins
    ymin = -0.3
    ymax = 0.5

    plt.figure(figsize=(figure_width2, figure_len2))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    baseline_reactivation = av_threshold
    change_in_reactivation = (np.array(l_delta_rE1) - baseline_reactivation) / baseline_reactivation

    # Define the colors for the custom colormap
    cmap_colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # White, Red
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)

    # Define the TwoSlopeNorm with vcenter at 15, vmin at -10, and vmax at 100
    norm = mcolors.TwoSlopeNorm(vmin=ymin, vcenter=center_threshold, vmax=ymax)

    # Normalize the change in reactivation
    values_range = norm(change_in_reactivation)  # Normalize using TwoSlopeNorm

    # Plotting and using the values to pick colors
    for i in range(len(hour_sims) - 1):
        color = cmap(values_range[i])
        ax.fill_between(hour_sims[i:i + 2], center_threshold, change_in_reactivation[i:i + 2], color=color,
                        edgecolor=(0, 0, 0, 0))

    plt.plot(hour_sims, change_in_reactivation, color=color_list[0], linewidth=plot_line_width, label='')
    if center_threshold != 0.0:
        plt.plot(hour_sims, center_threshold * np.ones_like(hour_sims), dash_capstyle='round',
                         linestyle=line_style_r_at, color='black', linewidth=plot_line_width, label='Aversive Threshold')
    else:
        plt.plot(hour_sims, np.zeros_like(hour_sims), dash_capstyle='round',
                     linestyle=line_style_r_at, color='black', linewidth=plot_line_width, label='$r_{bs,re}$')

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.tick_params(width=line_width, length=tick_len, labelsize=font_size_1)
    # cbar.set_ticks([0, 15, 100], fontsize=font_size_1, **hfont)
    
    cbar.set_ticks([center_threshold])
    cbar.ax.tick_params(labelsize=font_size_1)

    cbar.outline.set_linewidth(line_width)  # Set colorbar border thickness


    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)

    plt.yticks([ymin, center_threshold, ymax], fontsize=font_size_1, **hfont)
    #         plt.yticks([-30, center_threshold, 320], fontsize=font_size_1, **hfont)
    #         plt.ylabel('CIR [%]', fontsize=font_size_label, **hfont)
    plt.ylabel('Generalization Index', fontsize=font_size_label, **hfont)
    ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,loc='lower center', bbox_to_anchor=(0.5, 1.05), handlelength=3)

    plt.tight_layout()
    plt.savefig(name + '_delta_reactivation_color' + format)
    plt.show()
    plt.close()





def all_cases_CIR(l_time_points_phase2, hour_sims, l_all_delta_rE1, av_threshold,
                                    name, format='.pdf'):


    # plotting configuration
    ratio = 1.5
    figure_len, figure_width = 13 * ratio, 19 * ratio
    font_size_1, font_size_2 = 80 * ratio, 65 * ratio
    font_size_label = 80 * ratio
    legend_size = 50 * ratio
    legend_size2 = 65 * ratio
    line_width, tick_len = 9 * ratio, 20 * ratio
    marker_size = 15 * ratio
    marker_edge_width = 3 * ratio
    plot_line_width = 9 * ratio
    hfont = {'fontname': 'Arial'}
    sns.set(style='ticks')

    x_label_text = 'Time (h)'

    line_style_rb = (0, (0.05, 2.5))
    line_style_r_at = (0, (5, 5))
    # defining the colors for
    color_list = ['#3276b3', '#91bce0', # rE1 and WEE11, rE2 and WEE22
                  '#C10000', '#EFABAB', # rP1 and WEP11, rP2 and WEP22
                  '#007100', '#87CB87', # rS1 and WES11, rS2 and WES22
                #   '#6600cc'] # timepoints in long simulation
                '#80808026']

    ### % Change in reactivation of E1
    xmin = 0
    xmax = l_time_points_phase2[-1]
    ymin = 0
    ymax = 6.8

    plt.figure(figsize=(figure_width, figure_len))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    baseline_reactivation = av_threshold /1.15
    change_in_reactivation = [100 * (np.array(l_all_delta_rE1)[i] - baseline_reactivation) / baseline_reactivation for i in range(len(l_all_delta_rE1))]
    # print(change_in_reactivation)

    # Define the colors for the custom colormap
    # cmap_colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # White, Red
    cmap_colors = ['#66e066', '#ffffff', '#e06666']  # Green, White, Red

    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)

    # Define the TwoSlopeNorm with vcenter at 15, vmin at -10, and vmax at 100
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=15, vmax=100)
    # norm = mcolors.TwoSlopeNorm(vmin=np.min(change_in_reactivation[0]), vcenter=0, vmax=np.max(change_in_reactivation[0]))

    # Normalize using TwoSlopeNorm
    values_range = [norm(change_in_reactivation[i]) for i in range(len(change_in_reactivation))]

    # Find and print values where CIR is zero
    for condition_idx, condition_changes in enumerate(change_in_reactivation):
        zero_indices = np.where(np.isclose(condition_changes, 15, atol=0.9))[0]  # Find indices where CIR ~ 0
        if len(zero_indices) > 0:
            zero_times = [hour_sims[idx] for idx in zero_indices]  # Convert indices to time units
            zero_times_str = ', '.join([f"{time:.2f} h" for time in zero_times])  # Format as strings with 2 decimals
            print(f"Condition {condition_idx + 1}: CIR is zero at time points: {zero_times_str}")
        else:
            print(f"Condition {condition_idx + 1}: CIR is never zero.")


    # print(values_range)

    # Plotting and using the values to pick colors
    for i in range(len(hour_sims) - 1):
        # ax.fill_between(hour_sims[i:i + 2], 7, 7.8, color=cmap(values_range[0][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 6, 6.8, color=cmap(values_range[0][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 5, 5.8, color=cmap(values_range[1][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 4, 4.8, color=cmap(values_range[2][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 3, 3.8, color=cmap(values_range[3][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 2, 2.8, color=cmap(values_range[4][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 1, 1.8, color=cmap(values_range[5][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 0, 0.8, color=cmap(values_range[6][i]), edgecolor=(0, 0, 0, 0))


    # Create a colorbar
    '''sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)  # Adjust the 'pad' for spacing
    cbar.ax.tick_params(width=line_width, length=tick_len, labelsize=font_size_1)
    cbar.set_ticks([0, 15, 100])  # Set the ticks for vmin, vcenter, and vmax
    cbar.set_ticklabels(['0%', '15%', '100%'])  # Custom labels for the ticks
    cbar.outline.set_linewidth(line_width)  # Set colorbar border thickness
    cbar.set_label('% Change in Reactivation', fontsize=font_size_label, **hfont)'''


    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
    plt.yticks([0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4],
               ['Only SST-to-E', 'Only PV-to-E', 'Only E-to-E', 'S-to-E off', 'P-to-E off', 'E-to-E off', 'Full model'],
               fontsize=font_size_1, **hfont)
    plt.title('Generalization Index', fontsize=font_size_label, **hfont)

    plt.tight_layout()
    plt.savefig(name + format)
    plt.show()
    plt.close()

def all_cases_CIR_diff(l_time_points_phase2, hour_sims, l_all_delta_rE1, av_threshold,
                                    name, format='.pdf'):


    # plotting configuration
    ratio = 1.5
    figure_len, figure_width = 13 * ratio, 19 * ratio
    font_size_1, font_size_2 = 80 * ratio, 65 * ratio
    font_size_label = 80 * ratio
    legend_size = 50 * ratio
    legend_size2 = 65 * ratio
    line_width, tick_len = 9 * ratio, 20 * ratio
    marker_size = 15 * ratio
    marker_edge_width = 3 * ratio
    plot_line_width = 9 * ratio
    hfont = {'fontname': 'Arial'}
    sns.set(style='ticks')

    x_label_text = 'Time (h)'

    line_style_rb = (0, (0.05, 2.5))
    line_style_r_at = (0, (5, 5))
    # defining the colors for
    color_list = ['#3276b3', '#91bce0', # rE1 and WEE11, rE2 and WEE22
                  '#C10000', '#EFABAB', # rP1 and WEP11, rP2 and WEP22
                  '#007100', '#87CB87', # rS1 and WES11, rS2 and WES22
                #   '#6600cc'] # timepoints in long simulation
                '#80808026']

    ### % Change in reactivation of E1
    xmin = 0
    xmax = l_time_points_phase2[-1]
    ymin = 0
    ymax = 6.8

    plt.figure(figsize=(figure_width, figure_len))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    baseline_reactivation = av_threshold / 1.15
    change_in_reactivation = [100 * (np.array(l_all_delta_rE1)[i] - baseline_reactivation) / baseline_reactivation for i in range(len(l_all_delta_rE1))]

    # Define the colors for the custom colormap
    cmap_colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # Green, White, Red
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)

    # Define the TwoSlopeNorm with vcenter at 15, vmin at -10, and vmax at 100
    # norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=15, vmax=100)

    # Normalize using TwoSlopeNorm
    # values_range = [norm(change_in_reactivation[i]) for i in range(len(change_in_reactivation))]
    values_range = change_in_reactivation

    # Plotting and using the values to pick colors
    for i in range(len(hour_sims) - 1):
        # ax.fill_between(hour_sims[i:i + 2], 7, 7.8, color=cmap(values_range[0][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 6, 6.8, color=cmap(values_range[0][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 5, 5.8, color=cmap(values_range[1][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 4, 4.8, color=cmap(values_range[2][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 3, 3.8, color=cmap(values_range[3][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 2, 2.8, color=cmap(values_range[4][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 1, 1.8, color=cmap(values_range[5][i]), edgecolor=(0, 0, 0, 0))
        ax.fill_between(hour_sims[i:i + 2], 0, 0.8, color=cmap(values_range[6][i]), edgecolor=(0, 0, 0, 0))


    # Create a colorbar
    """sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.tick_params(width=line_width, length=tick_len, labelsize=font_size_1)
    cbar.set_ticks([0, 15, 100], fontsize=font_size_1, **hfont)
    cbar.outline.set_linewidth(line_width)  # Set colorbar border thickness"""

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
    plt.yticks([0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4],
               ['Only SST-to-E', 'Only PV-to-E', 'Only E-to-E', 'S-to-E off', 'P-to-E off', 'E-to-E off', 'Full model'],
               fontsize=font_size_1, **hfont)
    plt.title('% change in $r_{E1,re}$', fontsize=font_size_label, **hfont)

    plt.tight_layout()
    plt.savefig(name + format)
    plt.show()
    plt.close()

