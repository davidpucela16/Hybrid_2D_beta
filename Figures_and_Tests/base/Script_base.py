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

Malphigui=0

if Malphigui:
    directory='/home/pdavid/Bureau/Updated_BCs_2/Code' #Malpighi
    directory_script='/home/pdavid/Bureau/Updated_BCs_2/Figures_and_Tests/base'
    csv_directory='/home/pdavid/Bureau/Updated_BCs_2/Figures_and_Tests/base/csv_outputs'
else: #Auto_58
    directory='/home/pdavid/Bureau/Code/Updated_BCs_2/Code/'
    directory_script='/home/pdavid/Bureau/Code/Updated_BCs_2/Figures_and_Tests/base'
    csv_directory='/home/pdavid/Bureau/Code/Updated_BCs_2/Figures_and_Tests/base/csv_outputs'
os.chdir(directory)
import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos,pos_to_coords,get_MAE, get_MRE, get_position_cartesian_sources, plot_sketch
from Reconstruction_extended_space import reconstruction_extended_space
from Testing import Testing, extract_COMSOL_data, FEM_to_Cartesian,save_csv, save_var_name_csv
from Reconstruction_functions import coarse_cell_center_rec

import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
import pandas as pd
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8,8),
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

S=1
Rv=L/alpha+np.zeros(S)
pos_s=np.array([[0.5,0.5]])*L

#ratio=int(40/cells)*2
ratio=int(100*h_coarse//L/4)*2

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
Peaceman_reference=0
directory_COMSOL= directory_script + '/COMSOL_output/linear'
directory_COMSOL_metab=directory_script + '/COMSOL_output/metab'


array_of_cells=np.arange(23)+3
save_csv(csv_directory + '/array_of_cells.csv', ['range of cells'],array_of_cells)
array_of_cells=np.squeeze(np.array(pd.read_csv(csv_directory + '/array_of_cells.csv')))


if not os.path.exists(csv_directory): os.mkdir(csv_directory)
if not os.path.exists(csv_directory + '/linear'): os.mkdir(csv_directory + '/linear')    
if not os.path.exists(csv_directory + '/metab'): os.mkdir(csv_directory + '/metab')  

for i in array_of_cells:  
    if not os.path.exists(csv_directory + '/linear/cells={}'.format(i)): os.mkdir(csv_directory + '/linear/cells={}'.format(i))
    if not os.path.exists(csv_directory + '/metab/cells={}'.format(i)): os.mkdir(csv_directory + '/metab/cells={}'.format(i))
    

#%%
plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)
#%%
q_linear, phi_FEM_linear, x_FEM_linear, y_FEM_linear, FEM_x_1D_linear, FEM_y_1D_linear, x_1D_linear, y_1D_linear = extract_COMSOL_data(directory_COMSOL, [1,1,1])

q_metab, phi_FEM_metab, x_FEM_metab, y_FEM_metab, FEM_x_1D_metab, FEM_y_1D_metab, x_1D_metab, y_1D_metab = extract_COMSOL_data(directory_COMSOL_metab, [1,1,1])

#%%

t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)

s_Multi_cart_linear, q_Multi_linear=t.Multi()
Multi_rec_linear,_,_=t.Reconstruct_Multi(0,1, x_FEM_linear, y_FEM_linear)

#%%
c=0
plt.plot(t.x_fine, t.array_phi_field_x_Multi[c], label="Multi")
plt.plot(x_1D_linear, FEM_x_1D_linear, label="COMSOL")
plt.xlabel("x")
plt.legend()
plt.title("linear")
plt.show()

plt.plot(t.y_fine, t.array_phi_field_y_Multi[c], label="Multi")
plt.plot(y_1D_linear, FEM_y_1D_linear, label="COMSOL")
plt.xlabel("y")
plt.legend()
plt.title("linear")
plt.show()

#%%

if non_linear:
    
    Multi_FV_metab, q_Multi_metab=t.Multi(M,phi_0)
    Multi_rec_metab,_,_=t.Reconstruct_Multi(1, 1)
    
    plt.plot(t.x_fine, t.array_phi_field_x_Multi[c], label="Multi")
    plt.plot(x_1D_metab, FEM_x_1D_metab, label="COMSOL")
    plt.xlabel("x")
    plt.title("Metabolism")
    plt.legend()
    plt.show()
    
    plt.plot(t.y_fine, t.array_phi_field_y_Multi[c], label="Multi")
    plt.plot(y_1D_metab, FEM_y_1D_metab, label="COMSOL")
    plt.xlabel("y")
    plt.legend()
    plt.title("Metabolism")
    plt.show()
    
#%% - Comparison Multi with COMSOL

q_FV_metab=np.zeros(len(array_of_cells))
q_Multi_metab=np.zeros(len(array_of_cells))
q_FV_linear=np.zeros(len(array_of_cells))
q_Multi_linear=np.zeros(len(array_of_cells))
phi_Multi_FEM_linear=np.zeros((len(array_of_cells),len(x_FEM_linear)))
phi_Multi_FEM_metab=np.zeros((len(array_of_cells),len(x_FEM_metab)))


err_phi_Multi_linear=np.zeros(len(array_of_cells))
err_phi_Multi_metab=np.zeros(len(array_of_cells))
err_phi_FV_linear=np.zeros(len(array_of_cells))
err_phi_FV_metab=np.zeros(len(array_of_cells))

err_q_Multi_linear=np.zeros(len(array_of_cells))
err_q_Multi_metab=np.zeros(len(array_of_cells))
err_q_FV_linear=np.zeros(len(array_of_cells))
err_q_FV_metab=np.zeros(len(array_of_cells))

#The following are the arrays to calculate the phi-field error with comparing
#only the values at the cell's centers, therefore the reconstruction errors do not intervene
point_err_phi_metab=np.zeros(len(array_of_cells))
point_err_phi_linear=np.zeros(len(array_of_cells))

stabilization_array=np.zeros(len(array_of_cells))+0.5
stabilization_array[array_of_cells>17]=0.2

c=0

residual=np.zeros(())
for cells in array_of_cells[c:]:
    
    h_coarse=L/cells
    
    #Definition of the Cartesian Grid
    x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, int(np.around(L/h_coarse)))
    y_coarse=x_coarse
    ratio=1 #For the comparison we want the coarse resolution   
    directness=int(L/3/h_coarse)+10
    plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)
    
    #Extract COMSOL Data
    q_linear, phi_FEM_linear, x_FEM_linear, y_FEM_linear, FEM_x_1D_linear, FEM_y_1D_linear, x_1D_linear, y_1D_linear = extract_COMSOL_data(directory_COMSOL, [1,1,1])
    q_metab, phi_FEM_metab, x_FEM_metab, y_FEM_metab, FEM_x_1D_metab, FEM_y_1D_metab, x_1D_metab, y_1D_metab = extract_COMSOL_data(directory_COMSOL_metab, [1,1,1])
    
    #We create the testing object
    t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)
    t.stabilization=stabilization_array[c]
    
    #Linear Multiscale model
    s_FV_Multi_linear, q_Multi_linear[c]=t.Multi() #Obtain the values of the unknowns
    #Reconstruct the field using the FEM mesh
    phi_Multi_FEM_linear[c],_,_=t.Reconstruct_Multi(0,1, x_FEM_linear, y_FEM_linear) 

    #Same thing for the non linear model
    s_FV_Multi_metab, q_Multi_metab[c]=t.Multi(M,phi_0)
    phi_Multi_FEM_metab[c],_,_=t.Reconstruct_Multi(1,0, x_FEM_metab, y_FEM_metab)
    
    residual=np.append(residual, t.residual)
    
    #Calculate errors!!!
    err_q_Multi_linear[c]=get_MRE(q_linear,t.q_Multi_linear)
    err_phi_Multi_linear[c]=get_MRE(phi_FEM_linear, phi_Multi_FEM_linear[c])
    err_q_Multi_metab[c]=get_MRE(q_metab,t.q_Multi_metab)
    err_phi_Multi_metab[c]=get_MRE(phi_FEM_metab, phi_Multi_FEM_metab[c])
    
    #############################################################################
    #FV model used as comparison!!
    FV_FV_linear,q_FV_linear[c]=t.Linear_FV_Peaceman(0) #No Peaceman coupling!
    FV_FV_metab,q_FV_metab[c]=t.Metab_FV_Peaceman(M,phi_0,0)

    
    Cart_FEM_linear_field=FEM_to_Cartesian(x_FEM_linear, y_FEM_linear, phi_FEM_linear, 
                                           x_coarse, y_coarse)
    Cart_FEM_metab_field=FEM_to_Cartesian(x_FEM_metab, y_FEM_metab, phi_FEM_metab, 
                                           x_coarse, y_coarse)
    

    
    #To see how well it works the cell center reconstruction:
    Multi_linear_center=coarse_cell_center_rec(x_coarse, y_coarse, s_FV_Multi_linear,
                                               pos_s, t.s_blocks, q_Multi_linear, directness, Rv)
    Multi_metab_center=coarse_cell_center_rec(x_coarse, y_coarse, s_FV_Multi_metab,
                                               pos_s, t.s_blocks, q_Multi_metab, directness, Rv)
    #Calculate errors!!!
    err_q_FV_linear[c]=get_MRE(q_linear, np.array([q_FV_linear[c]]))
    err_q_FV_metab[c]=get_MRE(q_metab, np.array([q_FV_metab[c]]))
    err_phi_FV_linear[c]=get_MRE(np.ndarray.flatten(Cart_FEM_linear_field),FV_FV_linear)
    err_phi_FV_metab[c]=get_MRE(Cart_FEM_metab_field,FV_FV_metab)
    point_err_phi_linear[c]=get_MRE(Cart_FEM_linear_field, Multi_linear_center)
    point_err_phi_metab[c]=get_MRE(Cart_FEM_metab_field,Multi_metab_center)
    
    
    
    
    #Save the simulations
    dir_sim=directory_script + '/csv_outputs/linear/cells={}'.format(cells)
    if not os.path.exists(dir_sim): os.mkdir(dir_sim)
    
    save_csv(dir_sim+'/phi_Multi_FEM_linear.csv', ['x','y','phi'],
            np.array([ x_FEM_linear, y_FEM_linear, phi_Multi_FEM_linear[c]]))

    
    dir_sim=directory_script + '/csv_outputs/metab/cells={}'.format(cells)
    if not os.path.exists(dir_sim): os.mkdir(dir_sim)
    save_csv(dir_sim+'/phi_Multi_FEM_metab.csv', ['x','y','phi'],
            np.array([ x_FEM_linear, y_FEM_linear, phi_Multi_FEM_metab[c]]))
    
    c+=1
    
#%%
save_var_name_csv(csv_directory, point_err_phi_linear)
save_var_name_csv(csv_directory, point_err_phi_metab)

#%% - Save data 
save_csv(csv_directory+'/q_Multi_metab.csv', ['q'],np.array(q_Multi_metab))
save_csv(csv_directory+'/q_Multi_linear.csv', ['q'],np.array(q_Multi_linear))
save_csv(csv_directory+'/q_FV_metab.csv', ['q'],np.array(q_FV_metab))
save_csv(csv_directory+'/q_FV_linear.csv', ['q'],np.array(q_FV_linear))
save_csv(csv_directory+'/q_Multi_linear.csv', ['q'],np.array(q_Multi_linear))
save_csv(csv_directory+'/phi_Multi_FEM_linear.csv', array_of_cells,np.array(phi_Multi_FEM_linear))
save_csv(csv_directory+'/phi_Multi_FEM_metab.csv', array_of_cells,np.array(phi_Multi_FEM_metab))
save_csv(csv_directory+'/err_phi_Multi_linear.csv', ['L1 err'],np.array(err_phi_Multi_linear))
save_csv(csv_directory+'/err_phi_Multi_metab.csv', ['L1 err'],np.array(err_phi_Multi_metab))
save_csv(csv_directory+'/err_phi_FV_linear.csv', ['L1 err'],np.array(err_phi_FV_linear))
save_csv(csv_directory+'/err_q_Multi_linear.csv', ['L1 err'],np.array(err_q_Multi_linear))
save_csv(csv_directory+'/err_q_Multi_metab.csv', ['L1 err'],np.array(err_q_Multi_metab))
save_csv(csv_directory+'/err_q_FV_metab.csv', ['L1 err'],np.array(err_q_FV_metab))



 #%%
R=np.linalg.norm(Rv)
#plt.plot(L/array_of_cells/R, err_phi_Multi_linear,'-o', label='$\\varepsilon^g_\phi$ Multiscale')
plt.plot(L/array_of_cells/R, point_err_phi_linear,'-o', label='$\\varepsilon^g_\phi$ Multiscale')
plt.plot(L/array_of_cells/R, err_q_Multi_linear,'-o', label='$\\varepsilon^g_q$ Multiscale')
plt.plot(L/array_of_cells/R, err_q_FV_linear,'-o', label='$\\varepsilon^g_q$ FV')
plt.plot(L/array_of_cells/R, err_phi_FV_linear,'-o', label='$\\varepsilon^g_\phi$ FV')
plt.yscale('log')
plt.xlabel("h/Rv")
plt.ylabel('relative error')
plt.title("Linear Model")
plt.legend()
plt.show()

plt.plot(L/array_of_cells/R, err_phi_Multi_metab,'-o', label='$\\varepsilon^g_\phi$ Multiscale')
plt.plot(L/array_of_cells/R, err_q_Multi_metab,'-o', label='$\\varepsilon^g_q$ Multiscale')
plt.plot(L/array_of_cells/R, err_q_FV_metab,'-o', label='$\\varepsilon^g_q$ FV')
plt.plot(L/array_of_cells/R, err_phi_FV_metab,'-o', label='$\\varepsilon^g_\phi$ FV')
plt.yscale('log')
plt.xlabel("h/Rv")
plt.ylabel('relative error')
plt.title("Non linear model")
plt.legend()
plt.show()

#%%
# =============================================================================
# validation=0.4370238940521243
# FV_array=np.array([])
# Peac_array=np.array([])
# for ratio in np.array([2,4,6,8,10,20]):
#     print(ratio)
#     t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)
#     _,q_FV=t.Linear_FV_Peaceman(0)
#     _,q_P=t.Linear_FV_Peaceman(1)
#     FV_array=np.append(FV_array, q_FV)
#     Peac_array=np.append(Peac_array, q_P)
# =============================================================================
