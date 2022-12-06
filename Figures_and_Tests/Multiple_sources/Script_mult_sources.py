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
ratio=int(100*h_coarse//L/2)

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

#%%

# =============================================================================
# range_cells=np.array([5,7,10,15,20,30])
# save_csv(csv_directory + '/range_cells.csv', ['range of cells'],range_cells)
# 
# =============================================================================
range_cells=np.squeeze(np.array(pd.read_csv(csv_directory + '/range_cells.csv')))

if not os.path.exists(csv_directory): os.mkdir(csv_directory)
if not os.path.exists(csv_directory + '/linear'): os.mkdir(csv_directory + '/linear')    
if not os.path.exists(csv_directory + '/metab'): os.mkdir(csv_directory + '/metab')  

for i in range_cells:  
    if not os.path.exists(csv_directory + '/linear/cells={}'.format(i)): os.mkdir(csv_directory + '/linear/cells={}'.format(i))
    if not os.path.exists(csv_directory + '/metab/cells={}'.format(i)): os.mkdir(csv_directory + '/metab/cells={}'.format(i))
    
#%%




err_q_linear=np.array([])
err_phi_linear=np.array([])
err_q_metab=np.array([])
err_phi_metab=np.array([])

FV_err_q_linear=np.array([])
FV_err_phi_linear=np.array([])

FV_q_linear=np.array([])

FV_err_q_metab=np.array([])
FV_err_phi_metab=np.array([])


for cells in range_cells:
    h_coarse=L/cells
    #Definition of the Cartesian Grid
    x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, int(np.around(L/h_coarse)))
    y_coarse=x_coarse
    ratio=int(100*h_coarse//L/4)*2
    if ratio<2: ratio=2
    directness=np.array([1,2,3,6,12])[np.where(range_cells==cells)[0][0]]
    #directness=60
    plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)
    t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)
    s_Multi_cart_coarse_linear, q_Multi_linear=t.Multi()
    phi_Multi_cart_fine_linear,_,_=t.Reconstruct_Multi(0,0)
    
    
    length=cells*ratio
    x_c, y_c = np.zeros((cells*ratio)**2),np.zeros((cells*ratio)**2)
    for i in range(len(t.y_fine)):
        x_c[i*length:(i+1)*length]=np.around(t.x_fine, decimals=3)
        y_c[i*length:(i+1)*length]=np.around(t.y_fine[i], decimals=3)+np.zeros(length)
        
        
        
    dir_sim=directory_script + '/csv_outputs/linear/cells={}'.format(cells)

    save_csv(dir_sim+'/phi_Multi_cart_fine_linear.csv', 
                                                  ['x','y','phi'],
                                                  np.array([ x_c, y_c,np.ndarray.flatten(phi_Multi_cart_fine_linear)]))
    save_csv(dir_sim+'/q_Multi_linear.csv', ['q'], q_Multi_linear)
    if non_linear: 
        s_Multi_cart_coarse_metab, q_Multi_metab=t.Multi(M,phi_0)
        Multi_rec_cart_metab,_,_=t.Reconstruct_Multi(1,0)
        
        dir_sim=directory_script + '/csv_outputs/metab/cells={}'.format(cells)
    
        if not os.path.exists(dir_sim): 
            os.mkdir(dir_sim)
        save_csv(dir_sim+'/phi_Multi_cart_fine_metab.csv', 
                                                  ['x','y','phi'],
                                                  np.array([ x_c, y_c,np.ndarray.flatten(Multi_rec_cart_metab)]))
        save_csv(dir_sim+'/q_Multi_metab.csv', ['q'], q_Multi_metab)
        
        if FV_reference:
            FV_m=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, 1, C_v_array, BC_type, BC_value)
            phi_FV_cart_metab, q_FV_metab=FV_m.Metab_FV_Peaceman(M, phi_0, 0)
            x_FV, y_FV=array_cartesian_positions(FV_m.x_coarse, FV_m.y_coarse)
            save_csv(dir_sim+'/phi_FV_cart_metab.csv', ['x','y','phi'],
                    np.array([ x_FV, y_FV, np.ndarray.flatten(phi_FV_cart_metab)]))   
            save_csv(dir_sim+'/q_FV_metab.csv', ['q'],q_FV_metab)  
            
    if FV_reference:
        FV=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, 1, C_v_array, BC_type, BC_value)
        phi_FV_cart_linear, q_FV_linear=FV.Linear_FV_Peaceman(0) #FV solution without Peaceman approx
        x_FV, y_FV=array_cartesian_positions(FV.x_coarse, FV.y_coarse)
        save_csv(dir_sim+'/phi_FV_cart_linear.csv', ['x','y','phi'],
                np.array([ x_FV, y_FV, np.ndarray.flatten(phi_FV_cart_linear)]))   
        save_csv(dir_sim+'/q_FV_linear.csv', ['q'],q_FV_linear)      
        
        
    if COMSOL_reference:
        q_linear, phi_FEM_linear, x_FEM_linear, y_FEM_linear = extract_COMSOL_data(directory_COMSOL, [1,1,0])
        phi_Multi_FEM_linear,_,_=t.Reconstruct_Multi(0,0, x_FEM_linear/1e6, y_FEM_linear/1e6)
        
        fig, axs=plt.subplots(2,2)
        im1=axs[0,0].tricontourf(x_FEM_linear, y_FEM_linear,phi_Multi_FEM_linear, levels=100)
        axs[0,0].set_title("Reconstruction of the Multi model - Linear")
        plt.colorbar(im1, ax=axs[0,0])
        
        diff=phi_Multi_FEM_linear-phi_FEM_linear
    
        im2=axs[0,1].tricontourf(x_FEM_linear, y_FEM_linear,diff, levels=100)
        axs[0,1].set_title('Absolute $\phi$-error - Linear')
        plt.colorbar(im2,ax=axs[0,1])
        
        axs[1,0].plot(q_linear, label='COMSOL')
        axs[1,0].plot(q_Multi_linear, label='Multi')
        axs[1,0].legend()
        
        
        phi_Multi_cart_coarse_linear=coarse_cell_center_rec(x_coarse, y_coarse, t.s_Multi_cart_linear, pos_s, t.s_blocks, t.q_Multi_linear, directness, Rv)
        phi_FEM_cart_coarse_linear=FEM_to_Cartesian(x_FEM_linear/1e6, y_FEM_linear/1e6, phi_FEM_linear, x_coarse, y_coarse)
        
        axs[1,1].plot(np.abs(q_linear-q_Multi_linear)/np.abs(q_linear), label="relative error")
        axs[1,1].plot(np.abs(q_linear-q_Multi_linear), label="abs error")
        axs[1,1].legend()
        
        plt.show()
        
        print("MRE in flux estimation= ", get_MRE(q_linear, q_Multi_linear))
        print("MRE in $phi$ - field= ", get_MRE(phi_FEM_cart_coarse_linear, phi_Multi_cart_coarse_linear))
        
        err_q_linear=np.append(err_q_linear, get_MRE(q_linear, q_Multi_linear))
        err_phi_linear=np.append(err_phi_linear, get_MRE(phi_FEM_cart_coarse_linear,phi_Multi_cart_coarse_linear))
        
               
        
        dir_sim=directory_script + '/csv_outputs/linear/cells={}'.format(cells)

        save_csv(dir_sim+'/phi_Multi_FEM_linear.csv', ['x','y','phi'],
                np.array([ x_FEM_linear, y_FEM_linear, phi_Multi_FEM_linear]))
        save_csv(dir_sim+'/phi_FEM_linear.csv', ['x', 'y', 'phi'],
                 np.array([x_FEM_linear, y_FEM_linear, phi_FEM_linear]))
        
        if FV_reference:
            FV_err_q_linear=np.append(FV_err_q_linear, get_MRE(q_linear, q_FV_linear))
            FV_err_phi_linear=np.append(FV_err_phi_linear, get_MRE(np.ndarray.flatten(phi_FEM_cart_coarse_linear), phi_FV_cart_linear))
        
        if non_linear:
            
            q_metab, phi_FEM_metab, FEM_x_metab, FEM_y_metab = extract_COMSOL_data(directory_COMSOL_metab, [1,1,0])
            phi_Multi_FEM_metab,_,_=t.Reconstruct_Multi(1,0, x_FEM_linear/1e6, y_FEM_linear/1e6)
            fig, axs=plt.subplots(2,2)
            im1=axs[0,0].tricontourf(FEM_x_metab, FEM_y_metab,phi_Multi_FEM_metab, levels=100)
            axs[0,0].set_title("Reconstruction of the Multi model - metab")
            plt.colorbar(im1, ax=axs[0,0])
            
            diff=phi_Multi_FEM_metab-phi_FEM_metab
        
            im2=axs[0,1].tricontourf(FEM_x_metab, FEM_y_metab,diff, levels=100)
            axs[0,1].set_title('Absolute $\phi$-error - metab')
            plt.colorbar(im2,ax=axs[0,1])
            
            axs[1,0].plot(q_metab, label='COMSOL')
            axs[1,0].plot(q_Multi_metab, label='Multi')
            axs[1,0].legend()
            
            
            Multi_Cart_phi_metab=coarse_cell_center_rec(x_coarse, y_coarse, t.s_Multi_cart_metab, pos_s, t.s_blocks, t.q_Multi_metab, directness, Rv)
            phi_FEM_cart_coarse_metab=FEM_to_Cartesian(FEM_x_metab/1e6, FEM_y_metab/1e6, phi_FEM_metab, x_coarse, y_coarse)
            
            axs[1,1].plot(np.abs(q_metab-q_Multi_metab)/np.abs(q_metab), label="relative error")
            axs[1,1].plot(np.abs(q_metab-q_Multi_metab), label="abs error")
            axs[1,1].legend()
            
            plt.show()
            
            print("MRE in flux estimation= ", get_MRE(q_metab, q_Multi_metab))
            print("MRE in $phi$ - field= ", get_MRE(phi_FEM_cart_coarse_metab, Multi_Cart_phi_metab))
            err_q_metab=np.append(err_q_metab, get_MRE(q_metab, q_Multi_metab))
            err_phi_metab=np.append(err_phi_metab, get_MRE(phi_FEM_cart_coarse_metab, Multi_Cart_phi_metab))
            
            
            
            dir_sim=directory_script + '/csv_outputs/metab/cells={}'.format(cells)

            save_csv(dir_sim+'/phi_Multi_FEM_metab.csv', ['x','y','phi'],
                    np.array([ FEM_x_metab, FEM_y_metab, phi_Multi_FEM_metab]))
            save_csv(dir_sim+'/phi_FEM_metab.csv', ['x','y','phi'],
                    np.array([ FEM_x_metab, FEM_y_metab, phi_FEM_metab]))
            if FV_reference:
                FV_err_q_metab=np.append(FV_err_q_metab, get_MRE(q_metab, q_FV_metab))
                FV_err_phi_metab=np.append(FV_err_phi_metab, get_MRE(np.ndarray.flatten(phi_FEM_cart_coarse_metab), np.ndarray.flatten(phi_FV_cart_metab)))


save_csv(csv_directory + '/err_q_linear.csv', ['err_q_linear'], err_q_linear)
save_csv(csv_directory + '/err_phi_linear.csv', ['err_phi_linear'], err_phi_linear)
save_csv(csv_directory + '/err_q_metab.csv', ['err_q_metab'], err_q_metab)
save_csv(csv_directory + '/err_phi_metab.csv', ['err_phi_metab'], err_phi_metab)

save_csv(csv_directory + '/err_FV_q_linear.csv', ['err_q_linear'], FV_err_q_linear)
save_csv(csv_directory + '/err_FV_phi_linear.csv', ['err_phi_linear'], FV_err_phi_linear)

save_csv(csv_directory + '/err_FV_q_metab.csv', ['err_q_metab'], FV_err_q_metab)
save_csv(csv_directory + '/err_FV_phi_metab.csv', ['err_phi_metab'], FV_err_phi_metab)

#%%
h=1/range_cells
plt.plot(h,pd.read_csv(csv_directory + '/err_q_linear.csv').to_numpy(), '-s',color='b', markersize=16,label='$\\varepsilon_q^{Multi}$ linear')
plt.plot(h,pd.read_csv(csv_directory + '/err_phi_linear.csv').to_numpy(), '-v',color='b', markersize=16,label='$\\varepsilon_{\phi}^{Multi}$ linear')
plt.plot(h,pd.read_csv(csv_directory + '/err_q_metab.csv').to_numpy(), ':s',color='b', markersize=16,label='$\\varepsilon_q^{Multi}$ metab')
plt.plot(h,pd.read_csv(csv_directory + '/err_phi_metab.csv').to_numpy(), ':v',color='b', markersize=16,label='$\\varepsilon_{\phi}^{Multi}$ metab')

plt.plot(h,pd.read_csv(csv_directory + '/err_FV_q_linear.csv').to_numpy(), '-s',color='r', markersize=16,label='$\\varepsilon_q^{FV}$ linear')
plt.plot(h,pd.read_csv(csv_directory + '/err_FV_phi_linear.csv').to_numpy(), '-v', color='r',markersize=16,label='$\\varepsilon_{\phi}^{FV}$ linear')

plt.plot(h,pd.read_csv(csv_directory + '/err_FV_q_metab.csv').to_numpy(), ':s',color='r', markersize=16,label='$\\varepsilon_q^{FV}$ metab')
plt.plot(h,pd.read_csv(csv_directory + '/err_FV_phi_metab.csv').to_numpy(), ':v', color='r',markersize=16,label='$\\varepsilon_{\phi}^{FV}$ metab')

plt.legend()
plt.xlabel('$\dfrac{h}{R_{cap}}$')
plt.yscale('log')
plt.ylabel('Relative flux error')
plt.xlim(max(h)+0.02, min(h)-0.02)

#%%


