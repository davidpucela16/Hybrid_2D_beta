#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:18:37 2022

@author: pdavid

SCRIPT FOR THE SINGLE SOURCE AND TO EVALUATE THE NON LINEAR MODEL on a centered position with
both Dirichlet and periodic BCs

"""
#djkflmjaze
import os 
directory='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Updated_BCs/Code'
os.chdir(directory)
directory_script='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Updated_BCs/Tool_Validation'

import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos,pos_to_coords,get_MAE, get_MRE, get_position_cartesian_sources, plot_sketch
from Reconstruction_extended_space import reconstruction_extended_space
from Testing import Testing, extract_COMSOL_data

import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
import pandas
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12,12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


#0-Set up the sources
#1-Set up the domain
alpha=50

Da_t=10
D=1
K0=1
L=240

cells=5
h_coarse=L/cells


#Metabolism Parameters
M=Da_t*D/L**2
phi_0=0.4
conver_residual=5e-5
stabilization=0.5

#Definition of the Cartesian Grid
x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, int(np.around(L/h_coarse)))
y_coarse=x_coarse

#V-chapeau definition
directness=1
print("directness=", directness)

S=2
Rv=L/alpha+np.zeros(S)
R_size=L/alpha
pos_s=np.array([[0.5,0.5]])*L

#ratio=int(40/cells)*2
ratio=int(100*h_coarse//L)

print("h coarse:",h_coarse)
K_eff=K0/(np.pi*Rv**2)


p=np.linspace(0,1,100)
if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")


C_v_array=np.ones(S) 

BC_value=np.array([0,0.2,0,0.2])
BC_type=np.array(['Periodic', 'Periodic', 'Neumann', 'Dirichlet'])


#What comparisons are we making 
COMSOL_reference=1
non_linear=1
Peaceman_reference=1
coarse_reference=1
directory_COMSOL= directory_script + '/COMSOL_output/linear'
directory_COMSOL_metab=directory_script + '/COMSOL_output/metab'

def two_source_comparison_function(pos_s, Rv, cells, L, K_eff, D, directness, 
                                   ratio, C_v_array, BC_type, BC_value,
                                   directory_COMSOL, *reconstruct):
    
    t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)
    Multi_FV_linear, Multi_q_linear=t.Multi()
    t.ratio=1
    FV_FV, FV_q=t.Linear_FV_Peaceman(0)
    q_FEM, FEM_phi_linear, FEM_x_linear, FEM_y_linear, FEM_x_1D_linear, FEM_y_1D_linear, x_1D_linear, y_1D_linear = extract_COMSOL_data(directory_COMSOL)
    if reconstruct:
        Multi_rec,_,_=t.Reconstruct_Multi(0,0,FEM_x_linear, FEM_y_linear)
        
        plt.tricontourf(FEM_x_linear, FEM_y_linear, Multi_rec, levels=100)
        plt.colorbar()
        plt.title("Multi model")
        plt.show()
        
        plt.tricontourf(FEM_x_linear, FEM_y_linear, FEM_phi_linear, levels=100)
        plt.colorbar()
        plt.title("FEM model")
        plt.show()
        return(q_FEM, FV_q, Multi_q_linear, FEM_phi_linear, Multi_rec, FV_FV, FEM_x_linear, FEM_y_linear)
    else:
        return(q_FEM, FV_q, Multi_q_linear)
    

#%% - alpha=100, diff radii, both sources
directness=1
alpha=100
Rv=L/alpha+np.zeros(S)
Rv[0]/=2
R_size=L/alpha
K_eff=alpha*K0/(np.pi*L*Rv)
array_of_dist=np.array([0,1,2,4,6,8,12,16,20,22,24,26])*R_size

Multi_q_00=np.zeros((0,2))
FV_q_00=np.zeros((0,2))
FEM_q_00=np.zeros((0,2))

for d in array_of_dist:
    directory_COMSOL=directory_script + '/COMSOL_output/linear/alpha100_both_sources_diff_radii/d={}'.format(int(d/R_size))
    pos_s=np.array([[0,-d/2-Rv[0]],[0,d/2+Rv[1]]])+np.array([[0.5,0.5],[0.5,0.5]])*L
    print("distance between sources= ", (np.linalg.norm(pos_s[1]-pos_s[0])-np.sum(Rv))/R_size)
    plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)
     
    q_FEM, FV_q, Multi_q_linear=two_source_comparison_function(pos_s, Rv, cells, L, K_eff, D, directness, 
                                                               ratio, C_v_array, BC_type, BC_value,directory_COMSOL)
    
    Multi_q_00=np.concatenate((Multi_q_00, [Multi_q_linear]), axis=0)
    FEM_q_00=np.concatenate((FEM_q_00, [q_FEM]), axis=0)    
    FV_q_00=np.concatenate((FV_q_00, [FV_q]))
    print("MRE: ", get_MRE(q_FEM, Multi_q_linear))

plt.plot(array_of_dist, np.abs(FEM_q_00-Multi_q_00)/FEM_q_00, '-o')

#%%
directness=2

Multi_q_01=np.zeros((0,2))
FV_q_01=np.zeros((0,2))
FEM_q_01=np.zeros((0,2))

for d in array_of_dist:
    directory_COMSOL=directory_script + '/COMSOL_output/linear/alpha100_both_sources_diff_radii/d={}'.format(int(d/R_size))
    pos_s=np.array([[0,-d/2-Rv[0]],[0,d/2+Rv[1]]])+np.array([[0.5,0.5],[0.5,0.5]])*L
    print("distance between sources= ", (np.linalg.norm(pos_s[1]-pos_s[0])-np.sum(Rv))/R_size)
    plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)
     
    q_FEM, FV_q, Multi_q_linear=two_source_comparison_function(pos_s, Rv, cells, L, K_eff, D, directness, 
                                                               ratio, C_v_array, BC_type, BC_value,directory_COMSOL)
    
    Multi_q_01=np.concatenate((Multi_q_01, [Multi_q_linear]), axis=0)
    FEM_q_01=np.concatenate((FEM_q_01, [q_FEM]), axis=0)    
    FV_q_01=np.concatenate((FV_q_01, [FV_q]))
    print("MRE: ", get_MRE(q_FEM, Multi_q_linear))

plt.plot(array_of_dist, np.abs(FEM_q_01-Multi_q_01)/FEM_q_01, '-o')
#%%
plt.plot(array_of_dist, np.abs(FEM_q_01-Multi_q_01)/FEM_q_01, '-o')
plt.plot(array_of_dist, np.abs(FEM_q_00-Multi_q_00)/FEM_q_00, '-o')
plt.title("Both sources, relative error")
plt.show()
#%%
directness=1
C_v_array[0]=0
Multi_q_10=np.zeros((0,2))
FV_q_10=np.zeros((0,2))
FEM_q_10=np.zeros((0,2))

for d in array_of_dist:
    directory_COMSOL=directory_script + '/COMSOL_output/linear/alpha100_source_sink_diff_radii/d={}'.format(int(d/R_size))
    pos_s=np.array([[0,-d/2-Rv[0]],[0,d/2+Rv[1]]])+np.array([[0.5,0.5],[0.5,0.5]])*L
    print("distance between sources= ", (np.linalg.norm(pos_s[1]-pos_s[0])-np.sum(Rv))/R_size)
    plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)
     
    q_FEM, FV_q, Multi_q_linear=two_source_comparison_function(pos_s, Rv, cells, L, K_eff, D, directness, 
                                                               ratio, C_v_array, BC_type, BC_value,directory_COMSOL)
    
    Multi_q_10=np.concatenate((Multi_q_10, [Multi_q_linear]), axis=0)
    FEM_q_10=np.concatenate((FEM_q_10, [q_FEM]), axis=0)    
    FV_q_10=np.concatenate((FV_q_10, [FV_q]))
    print("MRE: ", get_MRE(q_FEM, Multi_q_linear))

directness=2

Multi_q_11=np.zeros((0,2))
FV_q_11=np.zeros((0,2))
FEM_q_11=np.zeros((0,2))

for d in array_of_dist:
    directory_COMSOL=directory_script + '/COMSOL_output/linear/alpha100_source_sink_diff_radii/d={}'.format(int(d/R_size))
    pos_s=np.array([[0,-d/2-Rv[0]],[0,d/2+Rv[1]]])+np.array([[0.5,0.5],[0.5,0.5]])*L
    print("distance between sources= ", (np.linalg.norm(pos_s[1]-pos_s[0])-np.sum(Rv))/R_size)
    plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)
     
    q_FEM, FV_q, Multi_q_linear=two_source_comparison_function(pos_s, Rv, cells, L, K_eff, D, directness, 
                                                               ratio, C_v_array, BC_type, BC_value,directory_COMSOL)
    
    Multi_q_11=np.concatenate((Multi_q_11, [Multi_q_linear]), axis=0)
    FEM_q_11=np.concatenate((FEM_q_11, [q_FEM]), axis=0)    
    FV_q_11=np.concatenate((FV_q_11, [FV_q]))
    print("MRE: ", get_MRE(q_FEM, Multi_q_linear))

plt.plot(array_of_dist, np.abs((FEM_q_11-Multi_q_11)/FEM_q_11), '-o')
plt.plot(array_of_dist, np.abs((FEM_q_10-Multi_q_10)/FEM_q_10), '-o')
plt.title("Source-sink, relative error")
plt.show()


