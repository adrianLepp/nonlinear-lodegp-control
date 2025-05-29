import os.path as path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List
from matplotlib.legend_handler import HandlerTuple

#https://jwalton.info/Embed-Publication-Matplotlib-Latex/   



#---------------------------------------


INIT = False

fileFolder = '../data/'
imgFolder = '../img/'
fileFormat = 'pdf'

def init():
    global textWidth, Session, Base, INIT
    if not INIT:
        # in LaTex show the textWidth with '\the\textwidth'
        textWidth= 469.4704
        plt.rcParams['text.usetex'] = True
        try:
            print(plt.style.available)
            print("Your style sheets are located at: {}".format(path.join(mpl.__path__[0], 'mpl-data', 'stylelib')))
            plt.style.use('seaborn-v0_8-paper')
            plt.style.use('tex')
            INIT = True
        except:
            print('style not found')


def set_size(width_pt, fraction=1, subplots=(1, 1), heightScale=1):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1]) * heightScale

    return (fig_width_in, fig_height_in)


CM_to_PT = 28.3465

line_style = [ '-', '--' ,  '.', '-.', ':',  '.', '-.', ':']
dflt_data_names = ['LODEGP', 'linear model', 'Training']
dflt_xLabel = 'Time ($\mathrm{s})$'
dflt_yLabel = ['Fill level ($\mathrm{m}$)', 'Flow rate ($\mathrm{m^3/s}$)']
dflt_headers = ['$x_1$', '$x_2$', '$u_1$']


def plot_loss(loss:dict):
    init()
    rows = 1
    cols = 1
    frac = 1
    h_scale = 0.8

    fig, ax1 = plt.subplots(rows, cols, figsize=set_size(2*textWidth/3, frac, (rows, cols), heightScale=h_scale))

    i = 0
    length = 0
    for key, values in loss.items():
        length = len(values)
        color = f'C{3+i}'
        ax1.plot(values, label=key, color=color)
        i += 1

    # ax1.plot(loss, label='Training Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Training Loss')
    ax1.set_ylim(-12,3)
    ax1.set_xlim(0, length-1)
    
    ax1.legend()
    ax1.grid(True)
    fig.tight_layout()
    return fig

def plot_error(gp_error, de_error=None, header:List[str]=None, xLabel='Time ($\mathrm{s})$', yLabel='Absolute error ($\mathrm{m}$)', uncertainty=None):
    init()
    rows = 2
    cols = 1
    frac = 1
    h_scale = 0.6

    alpha_train = 1
    alpha_gp = 0.8
    alpha_de = 0.8

    line_style_de = '--'
    line_style_gp = '-'

    lw_constr = 1
    t_range = [gp_error['time'][0], gp_error['time'][-1]]

    fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=set_size(textWidth, frac, (rows, cols), heightScale=h_scale))

    lines_1_gp = []
    lines_1_de = []
    lines_2_gp = []
    lines_2_de = []

    for i in range(len(header)):
        color = f'C{i}'
        if 'u' in header[i]:
            lines_2_gp.append(ax2.plot(gp_error['time'], gp_error[f'f{i+1}'], line_style_gp, color=color, alpha=alpha_gp)[0])
            if de_error is not None:
                lines_2_de.append(ax2.plot(de_error['time'], de_error[f'f{i+1}'], line_style_de, color=color, alpha=alpha_de)[0])
            if uncertainty is not None:
                zero_vector = np.zeros_like(gp_error['time'])
                ax2.fill_between(gp_error['time'], zero_vector , uncertainty[:,i], alpha=alpha_gp, color=color)
        else:
            lines_1_gp.append(ax1.plot(gp_error['time'], gp_error[f'f{i+1}'], line_style_gp, color=color, alpha=alpha_gp)[0])
            if de_error is not None:
                lines_1_de.append(ax1.plot(de_error['time'], de_error[f'f{i+1}'], line_style_de, color=color, alpha=alpha_de)[0])
            if uncertainty is not None:
                zero_vector = np.zeros_like(gp_error['time'])
                ax1.fill_between(gp_error['time'], zero_vector , uncertainty[:,i], alpha=alpha_gp, color=color)


    labels_1 = ['LODEGP $x_1$', 'LODEGP $x_2$']
    labels_2 = ['LODEGP $u_1$']
    handles_1 = [lines_1_gp[0], lines_1_gp[1]]
    handles_2 = [lines_2_gp[0]]

    if de_error is not None:
        labels_1.append('linear model $x_1$')
        labels_1.append('linear model $x_2$')
        labels_2.append('linear model $u_1$')
        handles_1.append(lines_1_de[0])
        handles_1.append(lines_1_de[1]) # lines_1_de[0], lines_1_de[1],
        handles_2.append(lines_2_de[0]) # lines_2_de[0], lines_2_de[1], lines_2_de[2]
    
    # lines_1_ref = (lines_1_ref[0], lines_1_ref[1])

    
    columnspacing = 0.5
    handletextpad = 0.5
    handlelength = 1.5
    ax1.legend(handles_1, labels_1, handler_map={tuple: HandlerTuple(ndivide=None)}, fancybox=True, framealpha=0.5,ncol=2, columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength)#, loc='upper right',
    ax2.legend(handles_2, labels_2, handler_map={tuple: HandlerTuple(ndivide=None)}, fancybox=True, framealpha=0.5,ncol=2, columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength)#, loc='upper right',

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)#, wspace=-0.8 
    # ax1.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=3)#, ncol=stateN, columnspacing=columnspacing, handletextpad=handletextpad
    # ax2.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=2)
    ax2.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel)
    ax2.set_ylabel('Absolute error ($\mathrm{m^3/s}$)')
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xticklabels([])
    ax1.set_xlim(t_range[0], t_range[1])
    ax2.set_xlim(t_range[0], t_range[1])
    # ax1.set_ylim(0.05-0.005, 0.255)
    # ax2.set_ylim(0-0.00001, 0.00015+0.00001)


    #ax2.yaxis.get_offset_text().set_position((-0.1, -0.5))
    #ax1.set_yticklabels([0.05,0.1,0.15,0.2,0.25])

    plt.plot()
    return fig


def plot_states(data, data_names=dflt_data_names, header=dflt_headers, xLabel=dflt_xLabel, yLabel=dflt_yLabel, uncertainty=None, title=None):
    init()
    rows = 2
    cols = 1
    frac = 1
    h_scale = 0.6

    alpha = [0.8, 0.8, 0.8, 0.8, 0.8,0.8, 0.8, 0.8, 0.8, 0.8] #FIXME
    line_style = [ '-', '--' ,  '.', '-.', ':',  '.', '-.', ':']

    alpha_uncertainty = 0.1

    lw_constr = 1
    t_range = [data[0].time[0], data[0].time[-1]] #TODO

    fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=set_size(textWidth, frac, (rows, cols), heightScale=h_scale))

    lines_1 = [[] for j in range(len(data))]
    lines_2 = [[] for j in range(len(data))]

    for i in range(len(header)):
        color = f'C{i}'
        for j in range(len(data)):
            if 'u' in header[i]:
                lines_2[j].append(ax2.plot(data[j].time, data[j].y[:,i], line_style[j], color=color, alpha=alpha[j])[0])
                if data[j].uncertainty is not None:
                    ax2.fill_between(data[j].time, data[j].uncertainty['lower'][:,i], data[j].uncertainty['upper'][:,i], alpha=alpha_uncertainty, color=color)
            else:
                lines_1[j].append(ax1.plot(data[j].time, data[j].y[:,i], line_style[j], color=color, alpha=alpha[j])[0])
                if data[j].uncertainty is not None:
                    ax1.fill_between(data[j].time, data[j].uncertainty['lower'][:,i], data[j].uncertainty['upper'][:,i], alpha=alpha_uncertainty, color=color)

    labels_2 = [f'{data_names[j]} {header[-1]}' for j in range(len(data))]
    labels_1 = []
    handles_1 = []
    handles_2 = []
    for j in range(len(data)):
        for i in range(len(header)-1):
            labels_1.append(f'{data_names[j]} {header[i]}')
            handles_1.append(lines_1[j][i])
        handles_2.append(lines_2[j][0])

    columnspacing = 0.5
    handletextpad = 0.5
    handlelength = 1.5
    ax1.legend(handles_1, labels_1, handler_map={tuple: HandlerTuple(ndivide=None)},  fancybox=True, framealpha=0.5,ncol=len(data_names), columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength) #loc='upper right',
    ax2.legend(handles_2, labels_2, handler_map={tuple: HandlerTuple(ndivide=None)}, fancybox=True, framealpha=0.5,ncol=len(data_names), columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength) #loc='upper right',

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)#, wspace=-0.8 
    if title is not None:
        ax1.set_title(title)
    # ax1.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=3)#, ncol=stateN, columnspacing=columnspacing, handletextpad=handletextpad
    # ax2.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=2)
    ax2.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel[0])
    ax2.set_ylabel(yLabel[1])
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xticklabels([])
    ax1.set_xlim(t_range[0], t_range[1])
    ax2.set_xlim(t_range[0], t_range[1])
    # ax1.set_ylim(0.05-0.005, 0.255)
    # ax2.set_ylim(0-0.00001, 0.00015+0.00001)

    #ax2.yaxis.get_offset_text().set_position((-0.1, -0.5))
    #ax1.se
    return fig

def plot_single_states(data, data_names=dflt_data_names, header=dflt_headers, xLabel=dflt_xLabel, yLabel=dflt_yLabel, line_styles= line_style, colors=None):
    init()
    rows = 1
    cols = 3
    # cols = data[0].state_dim + data[0].control_dim
    frac = 1
    h_scale = 1

    alpha = [0.75, 0.75, 0.75, 0.8, 0.8,0.8, 0.8, 0.8, 0.8, 0.8] #FIXME
    

    alpha_uncertainty = 0.1

    lw_constr = 1
    t_range = [data[0].time[0], data[0].time[-1]] #TODO

    size = set_size(textWidth, frac, (rows, cols), heightScale=h_scale)
    fig, axes = plt.subplots(rows, cols, figsize=(size[0]* 2,size[1]*2))

    for i in range(len(header)):
        for j in range(len(data)):
            if colors is not None:
                color = f'C{colors[j]}'
            else:
                color = f'C{j}'
            axes[i].plot(data[j].time, data[j].y[:,i],line_styles[j], color=color, alpha=alpha[j], label=data_names[j])
            if data[j].uncertainty is not None:
                axes[i].fill_between(data[j].time, data[j].uncertainty['lower'][:,i], data[j].uncertainty['upper'][:,i], alpha=alpha_uncertainty, color=color)

    for i in range(len(header)):
        axes[i].legend(fancybox=True, framealpha=0.5, loc='upper right')
        axes[i].set_xlabel(xLabel) 
        axes[i].set_ylabel(yLabel[i]) 
        axes[i].grid(True)
        axes[i].set_xlim(t_range[0], t_range[1])

    # axes[-1].set_ylim(-6, 4)
        

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    
    return fig

def surface_plot(x1:np.ndarray, x2:np.ndarray, y:np.ndarray, x1_ref:np.ndarray, x2_ref:np.ndarray, y_ref:np.ndarray, axisNames = [r'$x_1$', r'$x_2$', r'$y$'], labels = [r'$\hat{\alpha(\boldmath{x})}$', r'${\alpha(\boldmath{x})}$']):
    init()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15, azim=-120)
    ax.plot_surface(x1, x2, y, cmap='viridis', alpha=0.7, label=labels[0])

    ax.plot(x1_ref, x2_ref , y_ref, 'r.', label=labels[1])
    ax.set_xlabel(axisNames[0])
    ax.set_ylabel(axisNames[1])  
    ax.set_zlabel(axisNames[2])
    ax.legend(fancybox=True, framealpha=0.5)
    # ax.set_title(r'$\alpha$')

    if 'beta' in labels[0]:
        ax.set_zlim(0, 2)

    return fig

def plot_trajectory(test_data, points:dict, ax_labels=['Fill level tank 1 (m)', 'Fill level tank 2 (m)'], labels=['GP']):
    init()

    rows = 1
    cols = 1
    frac = 1
    h_scale = 1

    (fig_width_in, fig_height_in) = set_size(textWidth, frac, (rows, cols), heightScale=h_scale)
    fig, ax1 = plt.subplots(rows, cols, figsize=(4/5*fig_width_in, 4/5*fig_height_in))
    ax1.set_aspect('equal', adjustable='box')
    # plt.figure()
    if isinstance(test_data, list):
        for i in range(len(test_data)):
            ax1.plot(test_data[i].y[:,0],test_data[i].y[:,1], label=labels[i])
    else:
        ax1.plot(test_data.y[:,0],test_data.y[:,1], label='predicted trajectory')

    for key in points:
        if points[key] is not None:
            ax1.plot(points[key][0],points[key][1], 'o', label=key)


    # ax1.plot(centers[0],centers[1], 'o', color='C7', label='model centers')
    ax1.set_xlabel(ax_labels[0])
    ax1.set_ylabel(ax_labels[1])
    ax1.legend(fancybox=True, framealpha=0.5)
    ax1.grid(True)
    # ax1.set_xlim(0, 0.6)
    # ax1.set_ylim(0, 0.6)


def plot_weights(x, weights, x_label=dflt_xLabel):
    init()
    y_label = 'model weight'
    rows = 1
    cols = 1
    frac = 1
    h_scale = 0.5

    columnspacing = 0.5
    handletextpad = 0.5
    handlelength = 1.5
    (fig_width_in, fig_height_in) = set_size(textWidth, frac, (rows, cols), heightScale=h_scale)
    fig, ax1 = plt.subplots(rows, cols, figsize=(6/5*fig_width_in, 6/5*fig_height_in))

    for i, weight in enumerate(weights):
        ax1.plot(x, weight, label=f'model {i+1}')
        
        #plt.plot(x, sum(weights), label='Sum')
    ax1.legend(ncol=len(weights), fancybox=True, framealpha=0.5,columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength, loc='upper center')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_xlim(x[0], x[-1])
    ax1.grid(True)
    # ax1.set_ylim(-0.05, 1.15)
    return fig

def create_mpc_plot(training_data, sim_data, header:List[str], xLabel='Time ($\mathrm{s})$', yLabel='Water Level ($\mathrm{m}$)', reference_data=None, x_e=None,  close_constraint=False):
    width = '8.4 cm'

    textWidth_cm = 8.4

    init()
    rows = 2
    cols = 1
    frac = 1.2
    h_scale = 0.6

    alpha_train = 1
    alpha_est = 1
    alpha_ref = 0.7

    lw_constr = 1
    t_width = textWidth_cm * CM_to_PT  

    fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=set_size(t_width, frac, (rows, cols), heightScale=h_scale))
    

    if x_e is not None:
        equilibrium = np.array([x_e, x_e])
        t_range = [reference_data['time'][0], reference_data['time'][-1]]

        lower = [0, 0, 0]
        upper = [0.6, 0.6,2e-4]
        lower_constraint = np.array([lower, lower])
        upper_constraint = np.array([upper, upper] )

        ref_line = ':'
        constraint_line = '--'

    lines_1_x = []
    lines_1_ref = []
    lines_2_x = []
    lines_2_ref = []

    for i in range(len(header)):
        color = f'C{i}'
        gray = 'C7'
        if 'u' in header[i]:
            lines_2_x.append(ax2.plot(reference_data['time'], reference_data[f'f{i+1}'], color=color, alpha=alpha_est)[0])
            lines_2_ref.append(ax2.plot(t_range, equilibrium[:,i], ref_line,color=color, alpha=alpha_ref)[0])
            
            if close_constraint:
                lines_2_constr = ax2.plot(t_range, equilibrium[:,i]*0.9, constraint_line, color=gray, alpha=alpha_ref, lw=lw_constr)[0]
                ax2.plot(t_range, equilibrium[:,i]*1.1, constraint_line,  color=gray, alpha=alpha_ref, lw=lw_constr)
            else:
                lines_2_constr = ax2.plot(t_range, lower_constraint[:,i], constraint_line, label='Constraints', color=gray, alpha=alpha_ref, lw=lw_constr)[0]
                ax2.plot(t_range, upper_constraint[:,i], '--',  color=gray, alpha=alpha_ref)
        else:
            lines_1_x.append(ax1.plot(reference_data['time'], reference_data[f'f{i+1}'], color=color, alpha=alpha_est)[0]) 
            lines_1_ref.append(ax1.plot(t_range, equilibrium[:,i], ref_line, color=color, alpha=alpha_ref)[0])

            if close_constraint:
                lines_1_constr = ax1.plot(t_range, equilibrium[:,i]*0.9, constraint_line, color=gray, alpha=alpha_ref, lw=lw_constr)[0]
                ax1.plot(t_range, equilibrium[:,i]*1.1, constraint_line,  color=gray, alpha=alpha_ref, lw=lw_constr)

    labels_1 = ['$x_1$', '$x_2$', 'Reference']
    # labels_1 = ['Reference']
    labels_2 = ['$u_1$','Reference', 'Constraints']
    lines_1_ref = (lines_1_ref[0], lines_1_ref[1])

    handles_1 = [lines_1_x[0], lines_1_x[1], lines_1_ref ]
    handles_2 = [lines_2_x[0], lines_2_ref[0], lines_2_constr]

    if close_constraint:
        labels_1.append('Constraints')
        handles_1.append(lines_1_constr)

    columnspacing = 0.5
    handletextpad = 0.5
    handlelength = 1.5
    ax1.legend(handles_1, labels_1, handler_map={tuple: HandlerTuple(ndivide=None)}, fancybox=True, framealpha=0.5,ncol=4, columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength)# loc='upper right',
    ax2.legend(handles_2, labels_2, handler_map={tuple: HandlerTuple(ndivide=None)}, fancybox=True, framealpha=0.5,ncol=3, columnspacing=columnspacing, handletextpad=handletextpad,handlelength=handlelength)#loc='upper right',

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)#, wspace=-0.8 #FIXME
    # ax1.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=3)#, ncol=stateN, columnspacing=columnspacing, handletextpad=handletextpad
    # ax2.legend(loc='upper right', fancybox=True, framealpha=0.5,ncol=2)
    ax2.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel)
    ax2.set_ylabel('Flow rate ($\mathrm{m^3/s}$)')
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xticklabels([])
    ax1.set_xlim(t_range[0], t_range[1])
    ax2.set_xlim(t_range[0], t_range[1])
    # ax1.set_ylim(0., 0.3)
    # ax2.set_ylim(0-0.00001, 0.00015+0.00001) #TODO


    #ax2.yaxis.get_offset_text().set_position((-0.1, -0.5))
    #ax1.set_yticklabels([0.05,0.1,0.15,0.2,0.25])

    plt.plot()
    return fig