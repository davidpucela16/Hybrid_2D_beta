#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:18:37 2022

@author: pdavid

SCRIPT FOR THE SINGLE SOURCE AND TO EVALUATE THE NON LINEAR MODEL on a centered position with
both Dirichlet and periodic BCs

"""

#Define Computer
import os
Malphigui=0
if Malphigui:
    directory='/home/pdavid/Bureau/Hybrid_2D_beta/Code' #Malpighi
    directory_script='/home/pdavid/Bureau/Hybrid_2D_beta/Figures_and_Tests/Boundary'
    csv_directory='/home/pdavid/Bureau/Hybrid_2D_beta/Figures_and_Tests/Boundary/csv_outputs'
else: #Auto_58
    directory='/home/pdavid/Bureau/Code/Hybrid_2D_beta/Code/'
    directory_script='/home/pdavid/Bureau/Code/Hybrid_2D_beta/Figures_and_Tests/Boundary'
    csv_directory='/home/pdavid/Bureau/Code/Hybrid_2D_beta/Figures_and_Tests/Boundary/csv_outputs'
os.chdir(directory)


import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos,pos_to_coords,get_MAE, get_MRE, plot_sketch,get_position_cartesian_sources
from Reconstruction_extended_space import reconstruction_extended_space
from Testing import Testing, FEM_to_Cartesian, extract_COMSOL_data
from Reconstruction_functions import coarse_cell_center_rec

import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
import pandas as pd
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6,6 ),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

import pdb

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
#pos_s=(1-np.array([[0.5,0.05+1/alpha/2]]))*L

pos_s=np.array([[1,0.5]])*L-np.array([Rv[0],0])-np.array([0.5,0])*h_coarse

#ratio=int(40/cells)*2
ratio=int(100*h_coarse//L/2)

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
directory_COMSOL='../Figures_and_Tests/Boundary/COMSOL_output/linear'

#%% 2 - Plot source centers and mesh
#Position image

plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)

#%%

#Compare three solutions FV without coupling, Multiscale, and COMSOL:
dist_array=np.arange(13)*h_coarse/10

q_FEM_array=np.zeros(0)
q_Multi_array=np.zeros(0)
q_FV_array=np.zeros(0)

err_phi_global_array=np.zeros(0)
err_phi_Multi_point=np.zeros(0)
err_phi_FV=np.zeros(0)

dist_array[0]=0.02*h_coarse

c=0
for dist in dist_array:

    current_dir=directory_COMSOL + '/Boundary_dist=0{}'.format(int(dist*10/h_coarse))
    print("folder for c={}: Boundary_dist=0{}".format(c,int(dist*10/h_coarse)))
    pos_s=np.array([[1,0.5]])*L-np.array([Rv[0],0])-np.array([dist,0])
    t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, 1, C_v_array, BC_type, BC_value)
    FV_FV, FV_q=t.Linear_FV_Peaceman(0) #Construct the FV comparison 
    
    t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, 20, C_v_array, BC_type, BC_value)
    s_Multi_cart_linear, q_Multi_linear=t.Multi()
    
# =============================================================================
#     q_file=directory_COMSOL + "/Boundary_dist=0{}/q.txt".format(int(dist*10/h_coarse))
#     q_FEM=pandas.read_fwf(q_file, infer_rows=500).columns.astype(float)[-1]
#     
#     file=directory_COMSOL + '/Boundary_dist=0{}/contour.txt'.format(int(dist*10/h_coarse))
#     df=pandas.read_fwf(file, infer_nrows=500)
#     ref_data=np.array(df).T #reference 2D data from COMSOL
#     FEM_x=ref_data[0]*10**6 #in micrometers
#     FEM_y=ref_data[1]*10**6
#     phi_2D_COM=ref_data[2]
# =============================================================================
    q_FEM, phi_2D_COM, FEM_x, FEM_y, _, _ , _, _ = extract_COMSOL_data(current_dir,[1,1,1])    
    FEM_Multi,_,_=t.Reconstruct_Multi(0,0,FEM_x,FEM_y)
    

    plt.tricontourf(FEM_x, FEM_y, (phi_2D_COM-FEM_Multi)/phi_2D_COM, levels=100)
    plt.title("Absolute error $\phi$-field \n dist from boundary={}".format(dist/h_coarse))
    plt.colorbar(); plt.show()
    
    err_phi_global_array=np.append(err_phi_global_array, get_MRE(phi_2D_COM, FEM_Multi))
    
    q_FEM_array=np.append(q_FEM_array, q_FEM[-1])
    q_Multi_array=np.append(q_Multi_array, q_Multi_linear)
    q_FV_array=np.append(q_FV_array, FV_q)
    
    
    #For the \phi- errors:
    #Firstly, extract the reference solution on the Cartesian grid
    Cart_FEM_linear_field=FEM_to_Cartesian(FEM_x, FEM_y, phi_2D_COM, 
                                           x_coarse, y_coarse)

    #Evaluate the error made with the FV model
    err_phi_FV=np.append(err_phi_FV,get_MRE(np.ndarray.flatten(Cart_FEM_linear_field),FV_FV))
    
    #Then, evaluate the Multi model on the Cartesian grid
    Multi_linear_center=coarse_cell_center_rec(x_coarse, y_coarse, s_Multi_cart_linear,
                                               pos_s, t.s_blocks, q_Multi_linear, directness, Rv)
    #Evaluate the phi error on teh cartesian grid 
    err_phi_Multi_point=np.append(err_phi_Multi_point, get_MRE(Cart_FEM_linear_field, Multi_linear_center))


#%%
plt.plot(dist_array/h_coarse,np.abs(q_FEM_array-q_Multi_array)/q_FEM_array,'--o' ,label="$\\varepsilon^g_q$ Multiscale")
plt.plot(dist_array/h_coarse,np.abs(q_FEM_array-q_FV_array)/q_FEM_array, '--o' ,label="$\\varepsilon^g_q$ FV")
#plt.plot(dist_array/h_coarse,np.abs(err_phi_global_array), '-o' ,label="$\\varepsilon^g_\phi$ Multiscale")
plt.plot(dist_array/h_coarse, np.abs(err_phi_Multi_point), '-o', label="$\\varepsilon^g_\phi$ Multiscale")
plt.plot(dist_array/h_coarse, np.abs(err_phi_FV), '-o', label="$\\varepsilon^g_\phi$ FV")
plt.xlabel("d/h $\mu m$")
plt.ylabel('Relative error')
plt.legend()
plt.show()




#%%
pd.DataFrame(dist_array/h_coarse).to_csv(csv_directory + '/d_array_h.csv', sep=',', index=None)
pd.DataFrame(q_FEM_array).to_csv(csv_directory + '/q_FEM_array.csv', sep=',', index=None)
pd.DataFrame(q_Multi_array).to_csv(csv_directory + '/q_Multi_array.csv', sep=',', index=None)
pd.DataFrame(q_FV_array).to_csv(csv_directory + '/q_FV_array.csv', sep=',', index=None)
pd.DataFrame(err_phi_Multi_point).to_csv(csv_directory + '/err_phi_Multi_point.csv', sep=',', index=None)
pd.DataFrame(err_phi_FV).to_csv(csv_directory + '/err_phi_FV.csv', sep=',', index=None)



