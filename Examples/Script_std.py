#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:18:37 2022

@author: pdavid

SCRIPT FOR THE SINGLE SOURCE AND TO EVALUATE THE NON LINEAR MODEL
"""

import os 
directory='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Updated_BCs/Code'
os.chdir(directory)

import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
from reconst_and_test_module import reconstruction_sans_flux, reconstruction_extended_space
from Small_functions import coord_to_pos,pos_to_coords,get_MAE, get_MRE, get_position_cartesian_sources
from Reconstruction_extended_space import reconstruction_extended_space

import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
import pandas
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6,6 ),
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

M=Da_t*D/L**2
phi_0=0.4
cells=5
h_ss=L/cells
ratio=int(40/cells)*2
print("ratio: ", ratio)
#ratio=12
#Rv=np.exp(-2*np.pi)*h_ss

print("R: ", 1/(1/K0 + np.log(alpha/(5*cells*ratio))/(2*np.pi*D)))

conver_residual=5e-5

stabilization=0.5

validation=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=2
print("directness=", directness)

S=1
Rv=L/alpha+np.zeros(S)
#pos_s=(1-np.array([[0.5,0.05+1/alpha/2]]))*L

pos_s=np.array([[0.5,0.5]])*L

print("alpha: {} must be greater than {}".format(alpha, 5*ratio*cells))
print("h coarse:",h_ss)
K_eff=K0/(np.pi*Rv**2)

p=np.linspace(0,1,100)
if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")


C_v_array=np.ones(S)   

Peaceman_reference=0
directory_COMSOL='../COMSOL_files/base_periodic'
directory_COMSOL_metab='../COMSOL_files/base_periodic/metab'
COMSOL_reference=1
non_linear=1
BC_value=np.array([0,0,0,0])
BC_type=np.array(['Periodic', 'Periodic', 'Dirichlet', 'Dirichlet'])

#%% 2 - Plot source centers and mesh
#Position image
vline=(y_ss[1:]+x_ss[:-1])/2
plt.scatter(pos_s[:,0], pos_s[:,1])
plt.title("Position of the point sources")
for xc in vline:
    plt.axvline(x=xc, color='k', linestyle='--')
for xc in vline:
    plt.axhline(y=xc, color='k', linestyle='--')

plt.xlim([0,L])
plt.ylim([0,L])
plt.ylabel("y ($\mu m$)")
plt.xlabel("x ($\mu m$)")
plt.show()

#%% 3 - Solve the linear problem

n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
n.solve_linear_prob(BC_type, BC_value, C_v_array)

#%% 4 - Reconstruct and plot the linear problem
b=reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(n.phi_FV), n.phi_q)), n, L,ratio, directness)
p=b.reconstruction()   
b.reconstruction_boundaries_short(BC_type, BC_value)
b.rec_corners()



plt.imshow(b.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Steady State ")
plt.colorbar(); plt.show()



#%% 5 - Solve the non linear problem
if non_linear:
    n.Full_Newton(np.ndarray.flatten(n.phi_FV) , np.ndarray.flatten(n.phi_q), conver_residual, M, phi_0)
    
    a=reconstruction_sans_flux(n.metab_phi[-1], n, L,ratio, directness)
    p=a.reconstruction()   
    a.reconstruction_boundaries_short(BC_type, BC_value)
    a.rec_corners()
    plt.imshow(a.rec_final, origin='lower', vmax=np.max(a.rec_final))
    plt.title("bilinear reconstruction \n coupling model Metabolism")
    plt.colorbar(); plt.show()
    
    
    n.assemble_it_matrices_Sampson(n.u, n.q)

#%% 1 
if Peaceman_reference:
    FV=FV_validation(L, cells*ratio, pos_s, C_v_array, D, K_eff, Rv, np.zeros(4))
    FV_linear=FV.solve_linear_system()
    linear_phi_q_FV=FV.get_q()
    FV_linear_mat=FV_linear.reshape(cells*ratio, cells*ratio)
    
    print("R: ", 1/(1/K0 + np.log(0.2*FV.h/Rv)/(2*np.pi*D)))
    
    plt.imshow(FV_linear_mat, origin='lower')
    plt.colorbar()
    plt.title("FV reference solution, linear system\n mesh:{}x{}".format(ratio*cells, ratio*cells))
    plt.show()
    
    for i in pos_s:
        pos=coord_to_pos(FV.x, FV.y, i)
        
        plt.plot(FV_linear_mat[pos//len(FV.x),:], label="FV")
        plt.plot(b.rec_final[pos//len(FV.x),:],label="SS no metab")
        plt.legend()
        plt.title("Linear solution")
        plt.show()
    
    if non_linear:
        FV_non_linear=FV.solve_non_linear_system(phi_0,M, stabilization)
        #phi_FV=FV_linear.reshape(cells*ratio, cells*ratio)
        phi_FV=(FV.phi[-1]+FV.Corr_array).reshape(cells*ratio, cells*ratio)
    
    
    
        plt.imshow(phi_FV, origin='lower', vmax=np.max(phi_FV))
        plt.title("FV metab reference")
        plt.colorbar(); plt.show()
        #manual q
        q_array=-np.dot(FV.A_virgin.toarray()[FV.s_blocks,:],FV.phi[-1])*FV.h**2/D+M*(1-phi_0/(FV.phi[-1, FV.s_blocks]+FV.Corr_array[FV.s_blocks]+phi_0))
        
        print("MRE steady state system", get_MRE(n.phi_q, linear_phi_q_FV))
    
    
        for i in pos_s:
            pos=coord_to_pos(FV.x, FV.y, i)
            
            plt.plot(phi_FV[pos//len(FV.x),:], label="FV")
            plt.plot(a.rec_final[pos//len(FV.x),:],label="SS")
            plt.plot(b.rec_final[pos//len(FV.x),:],label="SS no metab")
            plt.legend()
            plt.show()



#%% 


if COMSOL_reference:
      
    directory_files=directory_COMSOL
    
    q_file=directory_files+ '/q.txt'  
    df=pandas.read_fwf(q_file, infer_rows=500)
    q=np.zeros(0, dtype=float)
    for i in df.to_dict().keys():
        q=np.append(q,float(i))
    
    file=directory_files + '/contour.txt'
    df=pandas.read_fwf(file, infer_nrows=500)
    ref_data=np.array(df).T #reference 2D data from COMSOL
    x_2D_COM=ref_data[0]*10**6 #in micrometers
    y_2D_COM=ref_data[1]*10**6
    phi_2D_COM=ref_data[2]
    
    file_1D=directory_files + '/plot_x.txt'
    plot_through=pandas.read_fwf(file_1D, infer_nrows=500)
    plot_1D=np.array(plot_through).T
    x_1D=plot_1D[0]*10**6
    plot_x=plot_1D[1]
    
    file_1D=directory_files + '/plot_y.txt'
    plot_through=pandas.read_fwf(file_1D, infer_nrows=500)
    plot_1D=np.array(plot_through).T
    y_1D=plot_1D[0]*10**6
    plot_y=plot_1D[1]
    
    from reconst_and_test_module import reconstruction_extended_space    
    r=reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, directness)
    r.phi_FV=n.phi_FV
    r.phi_q=n.phi_q
    r.set_up_manual_reconstruction_space(x_2D_COM, y_2D_COM)
    r.full_rec(C_v_array, BC_value, BC_type)
       

    plt.tricontourf(r.FEM_x, r.FEM_y, r.u, levels=100)
    plt.colorbar()
    plt.title("u FEM rec")    
    plt.show()
    plt.tricontourf(r.FEM_x, r.FEM_y, r.SL, levels=100)
    plt.colorbar()
    plt.title("SL FEM rec")
    plt.show()
    plt.tricontourf(r.FEM_x, r.FEM_y, r.DL, levels=100)
    plt.colorbar()
    plt.title("DL FEM rec")
    plt.show()
    plt.tricontourf(r.FEM_x, r.FEM_y, r.u+ r.SL+r.DL, levels=100)
    plt.colorbar()
    plt.title("Full FEM rec")
    plt.show()    
    plt.tricontourf(r.FEM_x, r.FEM_y, r.u+r.DL+r.SL-phi_2D_COM, levels=80)
    plt.colorbar()
    plt.title("Absolute error $\phi_{model} - \phi_{FEM}$")
    plt.show()
    print("$\phi(x) MRE${}".format( get_MAE(phi_2D_COM,r.u+r.DL+r.SL )))
    
    p_x,p_y=get_position_cartesian_sources(b.x, b.y, pos_s)
    
    plt.plot(x_1D, plot_x, label='COMSOL_x')
    plt.plot(b.x, b.rec_final[p_y[0],:], label='Coup_x')
    plt.legend()
    plt.title("Plot through source center")
    plt.xlabel("x")
    plt.ylabel("$\phi$")
    plt.show()
    
    plt.plot(y_1D, plot_y, label='COMSOL_y')
    plt.plot(b.y, b.rec_final[:,p_x[0]], label='Coup_y')
    plt.legend()
    plt.title("Plot through source center")
    plt.xlabel("y")
    plt.ylabel("$\phi$")
    plt.show()
    
    if non_linear:
        directory_files=directory_COMSOL_metab
        
        q_file=directory_files+ '/q.txt'
        df=pandas.read_fwf(q_file, infer_rows=500)
        q_metab=np.zeros(0, dtype=float)
        for i in df.to_dict().keys():
            q_metab=np.append(q_metab,float(i))
        
        file=directory_files + '/contour.txt'
        df=pandas.read_fwf(file, infer_nrows=500)
        ref_data=np.array(df).T #reference 2D data from COMSOL
        x_2D_metab=ref_data[0]*10**6 #in micrometers
        y_2D_metab=ref_data[1]*10**6
        phi_2D_metab=ref_data[2]
        
        file_1D=directory_files + '/plot_x.txt'
        plot_through=pandas.read_fwf(file_1D, infer_nrows=500)
        plot_1D=np.array(plot_through).T
        x_1D_metab=plot_1D[0]*10**6
        plot_x_metab=plot_1D[1]
        
        file_1D=directory_files + '/plot_y.txt'
        plot_through=pandas.read_fwf(file_1D, infer_nrows=500)
        plot_1D=np.array(plot_through).T
        y_1D_metab=plot_1D[0]*10**6
        plot_y_metab=plot_1D[1]
        
        metab_r=reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, directness)
        metab_r.phi_FV=n.metab_phi
        metab_r.phi_q=n.metab_q
        metab_r.set_up_manual_reconstruction_space(x_2D_COM, y_2D_COM)
        metab_r.full_rec(C_v_array, BC_value, BC_type)
           
    
        plt.tricontourf(metab_r.FEM_x, metab_r.FEM_y, metab_r.u, levels=100)
        plt.colorbar()
        plt.title("u FEM rec")    
        plt.show()
        plt.tricontourf(metab_r.FEM_x, metab_r.FEM_y, metab_r.SL, levels=100)
        plt.colorbar()
        plt.title("SL FEM rec")
        plt.show()
        plt.tricontourf(metab_r.FEM_x, metab_r.FEM_y, metab_r.DL, levels=100)
        plt.colorbar()
        plt.title("DL FEM rec")
        plt.show()
        plt.tricontourf(metab_r.FEM_x, metab_r.FEM_y, metab_r.u+ metab_r.SL+metab_r.DL, levels=100)
        plt.colorbar()
        plt.title("Full FEM rec")
        plt.show()    
        plt.tricontourf(metab_r.FEM_x, metab_r.FEM_y, metab_r.u+metab_r.DL+metab_r.SL-phi_2D_metab, levels=80)
        plt.colorbar()
        plt.title("Absolute error $\phi_{model} - \phi_{FEM}$")
        plt.show()
        print("$\phi(x) MRE${}".format( get_MAE(phi_2D_metab,metab_r.u+metab_r.DL+metab_r.SL )))
        
        p_x,p_y=get_position_cartesian_sources(a.x, a.y, pos_s)
        
        plt.plot(x_1D_metab, plot_x_metab, label='COMSOL_x')
        plt.plot(a.x, a.rec_final[p_y[0],:], label='Coup_x')
        plt.legend()
        plt.title("Plot through source center")
        plt.xlabel("x")
        plt.ylabel("$\phi$")
        plt.show()
        
        plt.plot(y_1D_metab, plot_y_metab, label='COMSOL_y')
        plt.plot(a.y, a.rec_final[:,p_x[0]], label='Coup_y')
        plt.legend()
        plt.title("Plot through source center")
        plt.xlabel("y")
        plt.ylabel("$\phi$")
        plt.show()