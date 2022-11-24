#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:18:37 2022

@author: pdavid

SCRIPT FOR THE SINGLE SOURCE AND TO EVALUATE THE NON LINEAR MODEL on a centered position with
both Dirichlet and periodic BCs

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
directness=4
print("directness=", directness)

S=1
Rv=L/alpha+np.zeros(S)
#pos_s=(1-np.array([[0.5,0.05+1/alpha/2]]))*L

pos_s=np.array([[0.9,0.5]])*L

print("alpha: {} must be greater than {}".format(alpha, 5*ratio*cells))
print("h coarse:",h_ss)
K_eff=K0/(np.pi*Rv**2)

p=np.linspace(0,1,100)
if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")


C_v_array=np.ones(S)   

COMSOL_reference=1


Peaceman_reference=1
directory_COMSOL='../Examples/Single_source/boundary'
#directory_COMSOL_metab='../Examples/Single_source/base_Dirichlet/metab'
BC_value=np.array([0,0,0,0])
BC_type=np.array(['Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet'])
#BC_type=np.array(['Dirichlet', 'Dirichlet', 'Neumann', 'Neumann'])


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

LP=assemble_SS_2D_FD(pos_s, Rv, h_ss,L, K_eff, D,directness)   
LP.solve_problem(BC_type, BC_value, C_v_array)


#%% 4 - Reconstruct and plot the linear problem
b=reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(LP.s_FV), LP.q)),LP, L,ratio, directness)
p=b.reconstruction()   
b.reconstruction_boundaries_short(BC_type, BC_value)
b.rec_corners()


plt.imshow(b.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Steady State ")
plt.colorbar(); plt.show()


#%% 1 
if Peaceman_reference:
    FV=FV_validation(L, cells*ratio, pos_s, C_v_array, D, K_eff, Rv, np.zeros(4))
    FV_linear=FV.solve_linear_system()
    Peaceman_q_linear=FV.get_q_linear()
    FV_mat_linear=FV_linear.reshape(cells*ratio, cells*ratio)
    
    print("R: ", 1/(1/K0 + np.log(0.2*FV.h/Rv)/(2*np.pi*D)))
    
    plt.imshow(FV_mat_linear, origin='lower')
    plt.colorbar()
    plt.title("FV reference solution, linear system\n mesh:{}x{}".format(ratio*cells, ratio*cells))
    plt.show()
    print("MRE steady state system Peaceman", get_MRE(LP.q, Peaceman_q_linear))
    
    for i in pos_s:
        pos=coord_to_pos(FV.x, FV.y, i)
        
        plt.plot(FV_mat_linear[pos//len(FV.x),:], label="FV")
        plt.plot(b.rec_final[pos//len(FV.x),:],label="SS no metab")
        plt.legend()
        plt.title("Linear solution")
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
    r.phi_FV=LP.s_FV
    r.phi_q=LP.q
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
    print("$\phi(x) MAE$ linear= {}".format( get_MAE(phi_2D_COM,r.u+r.DL+r.SL )))
    
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
    print("The MRE in flux linear prob: {}".format(get_MRE(q[1], b.phi_q)))
    