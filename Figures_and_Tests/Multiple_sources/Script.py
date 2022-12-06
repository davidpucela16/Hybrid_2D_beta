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
    directory='/home/pdavid/Bureau/Hybrid_2D_beta/Code' #Malpighi
    directory_script='/home/pdavid/Bureau/Hybrid_2D_beta/Figures_and_Tests/Multiple_sources'
    csv_directory='/home/pdavid/Bureau/Hybrid_2D_beta/Figures_and_Tests/Multiple_sources/csv_outputs'
else: #Auto_58
    directory='/home/pdavid/Bureau/Code/Hybrid_2D_beta/Code/'
    directory_script='/home/pdavid/Bureau/Code/Hybrid_2D_beta/Figures_and_Tests/Multiple_sources'
    csv_directory='/home/pdavid/Bureau/Code/Hybrid_2D_beta/Figures_and_Tests/Multiple_sources/csv_outputs'
os.chdir(directory)
import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos,pos_to_coords,get_MAE, get_MRE, get_position_cartesian_sources, plot_sketch
from Reconstruction_extended_space import reconstruction_extended_space
from Testing import Testing, extract_COMSOL_data, FEM_to_Cartesian, save_csv,array_cartesian_positions
from Reconstruction_functions import coarse_cell_center_rec

import pdb
import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
import pandas as pd
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
directness=9
print("directness=", directness)


#pos_s=(1-np.array([[0.5,0.05+1/alpha/2]]))*L

pos_s1=np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.96,0.43]])
pos_s2=np.array([[0.27,0.6],[0.53,0.65],[0.59,0.62],[0.67,0.69],[0.13,0.75],[0.15,0.93],[0.2,0.87],[0.28,0.98],[0.8,0.85],[0.83,0.92]])
pos_s3=np.concatenate((pos_s1, pos_s2))

pos_s3[:,0]-=0.06
pos_s3[:,1]-=0.03

pos_s=(pos_s3*0.8+0.1)*L

#pos_s[8,0]=127
#pos_s[9,0]=140

S=len(pos_s)
Rv=L/alpha+np.zeros(S)
#ratio=int(40/cells)*2
ratio=int(100*h_coarse//L/2)*2

print("h coarse:",h_coarse)
K_eff=K0/(np.pi*Rv**2)


p=np.linspace(0,1,100)
if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")


C_v_array=np.ones(S) 
C_v_array[[2,5,8,11,14]]=0

BC_value=np.array([0,0,0.3,0.3])
BC_type=np.array(['Neumann',  'Neumann','Dirichlet', 'Dirichlet'])


#What comparisons are we making 
COMSOL_reference=1
non_linear=1
FV_reference=1
coarse_reference=0
directory_COMSOL='../Figures_and_Tests/Multiple_sources/COMSOL_output/linear'
directory_COMSOL_metab='../Figures_and_Tests/Multiple_sources/COMSOL_output/metab'

#%% 2 - Plot source centers and mesh
#Position image

plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)


#%% - Multiscale model

t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)

Multi_FV_linear_1, Multi_q_linear_1=t.Multi()
Multi_rec_linear,_,_=t.Reconstruct_Multi(0,1)

plt.imshow(Multi_rec_linear, origin='lower')
plt.title("bilinear reconstruction \n coupling model Steady State ")
plt.colorbar(); plt.show()

if non_linear:
    Multi_FV_metab_1, Multi_q_metab_1=t.Multi(M,phi_0)
    Multi_rec_metab,_,_=t.Reconstruct_Multi(1, 1)
    plt.imshow(Multi_rec_metab, origin='lower', vmax=np.max(Multi_rec_metab))
    plt.title("bilinear reconstruction \n coupling model Metabolism")
    plt.colorbar(); plt.show()


#%%
t.Linear_FV_Peaceman(0)
t.Metab_FV_Peaceman(M, phi_0, 0)
            
                
#%% 3 - Solve the linear problem

n=non_linear_metab(pos_s, Rv, h_coarse, L, K_eff, D, directness)
n.solve_linear_prob(BC_type, BC_value, C_v_array)

#%% 4 - Reconstruct and plot the linear problem
b=reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(n.s_FV_linear), n.q_linear)), n, L,ratio, directness)
p=b.reconstruction()   
b.reconstruction_boundaries_short(BC_type, BC_value)
b.rec_corners()

Multi_FV_linear=n.s_FV_linear
Multi_q_linear=n.q_linear

plt.imshow(b.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Steady State ")
plt.colorbar(); plt.show()

#%% - COMSOL data


#%% 5 - Solve the non linear problem
if non_linear:
    n.Full_Newton(np.ndarray.flatten(n.s_FV_linear) , np.ndarray.flatten(n.q_linear), conver_residual, M, phi_0)
    
    a=reconstruction_sans_flux(n.arr_unk_metab[-1], n, L,ratio, directness)
    p=a.reconstruction()   
    a.reconstruction_boundaries_short(BC_type, BC_value)
    a.rec_corners()
    plt.imshow(a.rec_final, origin='lower', vmax=np.max(a.rec_final))
    plt.title("bilinear reconstruction \n coupling model Metabolism")
    plt.colorbar(); plt.show()
    
    
    n.assemble_it_matrices_Sampson(n.s_FV_metab, n.q_metab)
    
    Multi_FV_metab=n.s_FV_metab
    
    Multi_q_metab=n.q_metab
    Multi_rec_linear=a.rec_final

#%% 6- Peaceman reference
if Peaceman_reference:
    FV=FV_validation(L, cells*ratio, pos_s, C_v_array, D, K_eff, Rv,BC_type, BC_value, 1)
    Peaceman_FV_linear=FV.solve_linear_system()
    Peaceman_q_linear=FV.get_q_linear()
    Peaceman_mat_linear=Peaceman_FV_linear.reshape(cells*ratio, cells*ratio)
    
    print("R: ", 1/(1/K0 + np.log(0.2*FV.h/Rv)/(2*np.pi*D)))
    
    plt.imshow(Peaceman_mat_linear, origin='lower')
    plt.colorbar()
    plt.title("FV reference solution, linear system\n mesh:{}x{}".format(ratio*cells, ratio*cells))
    plt.show()
    print("MRE steady state system Peaceman", get_MRE(n.q_linear, Peaceman_q_linear))
    
    for i in pos_s:
        pos=coord_to_pos(FV.x, FV.y, i)
        
        plt.plot(Peaceman_mat_linear[pos//len(FV.x),:], label="FV")
        plt.plot(b.rec_final[pos//len(FV.x),:],label="SS no metab")
        plt.legend()
        plt.title("Linear solution")
        plt.show()
    
    if non_linear:
        FV_non_linear=FV.solve_non_linear_system(phi_0,M, stabilization)
        #phi_FV=FV_linear.reshape(cells*ratio, cells*ratio)
        Peaceman_FV_metab=(FV.phi_metab[-1]+FV.Corr_array).reshape(cells*ratio, cells*ratio)
    
    
    
        plt.imshow(Peaceman_FV_metab, origin='lower', vmax=np.max(Peaceman_FV_metab))
        plt.title("FV metab reference")
        plt.colorbar(); plt.show()
        #manual q
        Peaceman_q_metab=-np.dot(FV.A_virgin.toarray()[FV.s_blocks,:],FV.phi_metab[-1])*FV.h**2/D+M*(1-phi_0/(FV.phi_metab[-1, FV.s_blocks]+FV.Corr_array[FV.s_blocks]+phi_0))
        
        print("MRE non linear system", get_MRE(n.q_metab, FV.get_q_metab()))
    
    
        for i in pos_s:
            pos=coord_to_pos(FV.x, FV.y, i)
            
            plt.plot(Peaceman_mat_linear[pos//len(FV.x),:], label="Peaceman")
            plt.plot(Peaceman_FV_metab[pos//len(FV.x),:], label="Peac metab")
            plt.plot(a.rec_final[pos//len(FV.x),:],label="COupling")
            plt.plot(b.rec_final[pos//len(FV.x),:],label="Coupling no metab")
            plt.legend()
            plt.show()

