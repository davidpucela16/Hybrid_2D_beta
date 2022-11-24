#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:56:40 2022

@author: pdavid

This script has been created as part of the virgin code to have something with 
the basic tools I'm commonly using, so everytime I wanna start some simulations
I have a script somewhat ready'
"""


#%% - First set the proper directory
import os 
directory='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Updated_BCs/Code'
os.chdir(directory)

#%%

import pdb
import numpy as np 
import matplotlib.pyplot as plt
from module_2D_coupling_FV_nogrid import * 
import reconst_and_test_module as post
import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6,6 ),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'lines.linewidth' : 2
         }
pylab.rcParams.update(params)

#separation of scales
alpha=50

#physical parameters

D=1
K0=1

#Geometry
L=240
cells=5
h_ss=L/cells
ratio=int(50/cells)*2 #ratio for the reconstruction and validation 
print("ratio: ", ratio)
#ratio=12
#Rv=np.exp(-2*np.pi)*h_ss


print("R: ", 1/(1/K0 + np.log(alpha/(5*cells*ratio))/(2*np.pi*D)))
validation=True


#%%- Construction of the mesh 
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=1
print("directness=", directness)

#%% - Definition of the source problem
pos_s=np.array([[0.75,0.75]])*L
pos_s=np.array([[0.5,0.5]])*L

#pos_s=np.array([[0.77,0.77]])*L
S=len(pos_s)

C_v_array=np.ones(S)  

Rv=L/alpha+np.zeros(S)

print("alpha: {} must be greater than {} \n for the Peaceman validation".format(alpha, 5*ratio*cells))
print("h coarse:",h_ss)
K_eff=K0/(np.pi*Rv**2)

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

#%% - 1st solve the system through the hybrid model 
from module_2D_coupling_FV_nogrid import * 

T=assemble_SS_2D_FD(pos_s, Rv, h_ss,L, K_eff, D,directness)
T.pos_arrays()
T.initialize_matrices()

#Set Dirichlet BC:
dirichlet_array=np.array([0,0,0,0])
BC_type=np.array(['Periodic', 'Periodic','Dirichlet','Dirichlet'])
BC_type=np.array(['Dirichlet','Dirichlet','Dirichlet','Dirichlet'])

T.solve_problem(BC_type, dirichlet_array, C_v_array)

#%%
from Reconstruction_functions import *

pepa=post.real_NN_rec(T.x,T.y, T.phi_FV,pos_s, T.s_blocks, T.phi_q, ratio, T.h, directness, Rv, np.zeros(S)+K0, C_v_array)
plt.imshow(pepa.rec_plus_phibar,origin='lower')
plt.title("Slow potential piece-wise constant \n functions")
plt.colorbar()
plt.show()
plt.imshow(pepa.add_singular(directness)+pepa.rec,origin='lower'); plt.colorbar()
plt.title("$\phi$-field using piece wise constant functions \n for the slow potential")
plt.show()


#%% 3 - Bilinear Reconstruction
import reconst_and_test_module as post

b=post.reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(T.phi_FV), T.phi_q)), T, L,ratio, directness)
p=b.reconstruction()   
plt.imshow(b.rec_final, origin='lower')
plt.title("$\phi$-field using bilinear reconstruction \n for the slow potential")
#plt.colorbar(); 


BC_values=np.array([0,0,0,0])
b.reconstruction_boundaries_short(BC_type, BC_values)
plt.imshow(b.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Steady State ")
#plt.colorbar(); 

b.rec_corners()
plt.imshow(b.rec_final, origin='lower')
plt.title("$\phi$-field using bilinear reconstruction \n for the slow potential")
plt.colorbar(); plt.show()


#%%

s,r = b.get_u_pot(C_v_array)
plt.imshow(r, origin='lower')
plt.colorbar()
plt.title('Rapid potential')
plt.show()

plt.imshow(s, origin='lower')
plt.colorbar()
plt.title('Slow potential')
plt.show()

#%%

pp=np.argmax(r)//len(b.x)

plt.plot(b.x,b.rec_final[pp], label='Bilinear \nreconstruction')
plt.plot(b.x, s[pp], label='rapid')
plt.plot(b.x, r[pp], label='slow')
plt.ylabel('$\phi$')
plt.xlabel('x')
plt.title('Plots through the line y={}'.format(b.y[pp]))
plt.legend()

#%% - 1st - FV resolution as a reference

FV=FV_validation(L, cells*ratio, pos_s, C_v_array, D, K_eff, Rv, dirichlet_array)
FV_linear=FV.solve_linear_system()
FV_linear_mat=FV_linear.reshape(cells*ratio, cells*ratio)

#%% Plots FV reference solution - Peaceman Coupling

plt.imshow(FV_linear_mat, origin='lower')
plt.colorbar()
plt.title("FV reference solution, linear system\n mesh:{}x{}".format(ratio*cells, ratio*cells))
plt.show()

#%%
print("relative error in the flux esimation: ", get_MRE(FV.get_q(FV_linear), T.phi_q))

#%% - Peyrounette like figure
"""This section is to produce the single source figure. Skip is there 
are multiple sources"""
if not (len(pos_s)>1):
    length=len(x_ss)*ratio
    
    
    pos_s_i=coord_to_pos(b.x, b.y, pos_s[0])//length
    l2=int(coord_to_pos(b.x, b.y, pos_s[0])%length)
    
    plot_array=np.zeros(length)
    
    #recover the values at the FV cells' center:
    S=post.coarse_cell_center_rec(x_ss, y_ss, T.phi_FV, pos_s, T.s_blocks, T.phi_q, ratio, h_ss, directness, Rv)
    S[np.argmax(S)//cells, np.argmax(S)%cells]=np.sum(C_v_array)-np.sum(T.phi_q/T.K_0)
    
    A=post.tool_piece_wise_constant_ratio(ratio, S)
    reg=post.tool_piece_wise_constant_ratio(ratio, T.phi_FV)
    
    plot_array=np.concatenate((A[pos_s_i, :l2], b.rec_final[pos_s_i, l2:]))
    
    
    plt.scatter(np.arange(length), plot_array)
    
    
    u,pot=b.get_u_pot(C_v_array)
    
    
    avg_rapid, avg_slow=get_average_rapid(T.phi_FV, T.phi_q, directness, pos_s,T.x, T.y, T.s_blocks,
                                          Rv, T.h, C_v_array, K_eff)
    avg_phi=avg_rapid+avg_slow
    
    avg_slow_fine=post.tool_piece_wise_constant_ratio(ratio, avg_slow)
    avg_rapid_fine=post.tool_piece_wise_constant_ratio(ratio, avg_rapid)
    avg_phi_fine=avg_slow_fine+avg_rapid_fine
    
    plt.figure()
    for xc in vline:
        plt.axvline(x=xc, color='k', linestyle='--')
    
    
    #right side 
    #FV averaged concentration
    plt.plot(b.x[l2:3*ratio], avg_phi_fine[pos_s_i,l2:3*ratio], color='darkcyan', label='avg. $\phi$')
    plt.plot(b.x[3*ratio:4*ratio], avg_phi_fine[pos_s_i,3*ratio:4*ratio], color='darkcyan')
    plt.plot(b.x[4*ratio:5*ratio], avg_phi_fine[pos_s_i,4*ratio:5*ratio], color='darkcyan')  
    plt.scatter(x_ss[2:], avg_phi[2,2:], label='$\phi$ cell center', color='darkcyan')
    #Reference solution
    plt.plot(b.x[l2:], b.rec_final[pos_s_i, l2:], label='reference', color='red')
    #plt.plot(b.x, b.rec_final[pos_s_i], label='reference', color='red')
    
    
    #plt.plot(b.x[l2:],b.rec_final[pos_s_i, l2:], label='reconstructed')
    plt.plot(b.x[ratio:l2+1],avg_slow_fine[pos_s_i, ratio:l2+1], color='green')
    plt.scatter(x_ss[1:3], avg_slow[2,1:3] ,label='$\mathcal{s}$ cell center', color='green')
    plt.plot(b.x[:ratio],avg_slow_fine[pos_s_i, :ratio],linestyle='--' ,color='green')
    plt.plot(b.x[ratio:l2+1],pot[pos_s_i, ratio:l2+1], label='$\mathcal{r}$', color='slategrey')
    plt.plot(b.x[:ratio],pot[pos_s_i, :ratio], linestyle='--' ,color='slategrey')
    #plt.plot(b.x[:ratio], A[pos_s_i, :ratio], color='darkcyan')
    #plt.scatter(x_ss[0], S[2,0], color='darkcyan')
    #plt.legend()
    plt.ylim(-0.05,0.4)
    
#%% - Peyrounette like figure
"""This section is to produce the single source figure. Skip is there 
are multiple sources"""
if not (len(pos_s)>1):
    length=len(x_ss)*ratio
    
    
    pos_s_i=coord_to_pos(b.x, b.y, pos_s[0])//length
    l2=int(coord_to_pos(b.x, b.y, pos_s[0])%length)
    
    plot_array=np.zeros(length)
    
    #recover the values at the FV cells' center:
    S=post.coarse_cell_center_rec(x_ss, y_ss, T.phi_FV, pos_s, T.s_blocks, T.phi_q, ratio, h_ss, directness, Rv)
    S[np.argmax(S)//cells, np.argmax(S)%cells]=np.sum(C_v_array)-np.sum(T.phi_q/T.K_0)
    
    A=post.tool_piece_wise_constant_ratio(ratio, S)
    reg=post.tool_piece_wise_constant_ratio(ratio, T.phi_FV)
    
    plot_array=np.concatenate((A[pos_s_i, :l2], b.rec_final[pos_s_i, l2:]))
    
    
    plt.scatter(np.arange(length), plot_array)
    
    
    u,pot=b.get_u_pot(C_v_array)
    
    
    avg_rapid, avg_slow=get_average_rapid(T.phi_FV, T.phi_q, directness, pos_s,T.x, T.y, T.s_blocks,
                                          Rv, T.h, C_v_array, K_eff)
    avg_phi=avg_rapid+avg_slow
    
    avg_slow_fine=post.tool_piece_wise_constant_ratio(ratio, avg_slow)
    avg_rapid_fine=post.tool_piece_wise_constant_ratio(ratio, avg_rapid)
    avg_phi_fine=avg_slow_fine+avg_rapid_fine
    
    plt.figure()
    for xc in vline:
        plt.axvline(x=xc, color='k', linestyle='--')
    
    
    #right side 
    #FV averaged concentration
    plt.plot(b.x[l2:3*ratio], avg_phi_fine[pos_s_i,l2:3*ratio], color='darkcyan', label='avg. $\phi$')
    plt.plot(b.x[3*ratio:4*ratio], avg_phi_fine[pos_s_i,3*ratio:4*ratio], color='darkcyan')
    plt.plot(b.x[4*ratio:5*ratio], avg_phi_fine[pos_s_i,4*ratio:5*ratio], color='darkcyan')  
    plt.scatter(x_ss[2:], avg_phi[2,2:], label='$\phi$ cell center', color='darkcyan')
    #Reference solution
    plt.plot(b.x[l2:], b.rec_final[pos_s_i, l2:], label='reference', color='red')
    #plt.plot(b.x, b.rec_final[pos_s_i], label='reference', color='red')
    
    
    #plt.plot(b.x[l2:],b.rec_final[pos_s_i, l2:], label='reconstructed')
    plt.plot(b.x[ratio:l2+1],avg_slow_fine[pos_s_i, ratio:l2+1], color='green')
    plt.scatter(x_ss[:3], avg_slow[2,:3] ,label='$\mathcal{s}$ cell center', color='green')
    plt.plot(b.x[:ratio],avg_slow_fine[pos_s_i, :ratio] ,color='green')
    plt.plot(b.x[ratio:l2+1],pot[pos_s_i, ratio:l2+1], label='$\mathcal{r}$', color='slategrey')
    plt.plot(b.x[:ratio],np.zeros(ratio), color='slategrey')
    #plt.plot(b.x[:ratio], A[pos_s_i, :ratio], color='darkcyan')
    #plt.scatter(x_ss[0], S[2,0], color='darkcyan')
    #plt.legend()
    plt.ylim(-0.05,0.4)
    


#%%
if len(pos_s) > 1:
    plt.plot(T.phi_q, label="MyCode")
    plt.plot(FV.get_q(FV_linear), label="FV reference")
    plt.legend()
    plt.show()


#%% - Plots 
print("MRE steady state system", get_MRE(n.phi_q, FV.get_q(FV_linear)))

#%%
for i in pos_s:
    pos=coord_to_pos(FV.x, FV.y, i)
    
    plt.plot(FV_linear_mat[pos//len(FV.x),:], label="FV")
    plt.plot(b.rec_final[pos//len(FV.x),:],label="SS no metab")
    plt.legend()
    plt.title("Linear solution")
    plt.show()


for i in pos_s:
    pos=coord_to_pos(FV.x, FV.y, i)
    
    plt.plot(phi_FV[pos//len(FV.x),:], label="FV")
    plt.plot(a.rec_final[pos//len(FV.x),:],label="SS")
    plt.plot(b.rec_final[pos//len(FV.x),:],label="SS no metab")
    plt.legend()
    plt.show()

# =============================================================================
# c=0
# for i in pos_s:
#     pos=coord_to_pos(FV.x, FV.y, i)
#     
#     plt.plot(FV.x,phi_FV[pos//len(FV.x),:], label="FV")
#     plt.scatter(n.x,(n.rec_sing+n.u).reshape(cells, cells)[n.s_blocks[c]//cells,:],label="SS")
#     plt.plot(b.x,b.rec_final[pos//len(FV.x),:],label="SS no metab")
#     plt.legend()
#     plt.show()
#     c+=1
# =============================================================================

print("relative errors")
print(np.abs(n.phi[-1,-S:]-FV.get_q(FV.phi[-1]))/FV.get_q(FV.phi[-1]))

print("absolute error")
print(np.abs(n.phi[-1,-S:]-FV.get_q(FV.phi[-1])))

print("L2_error")
print(get_L2(n.phi[-1,-S:], FV.get_q(FV.phi[-1])))

print("MRE", get_MRE(n.phi[-1,-S:], FV.get_q(FV.phi[-1])))

u,pot=b.get_u_pot(C_v_array)


plt.subplots(2,2, figsize=(12,12))

plt.subplot(2,2,1)
plt.imshow(pot, origin='lower')
plt.title("SL + DL in $\overline{\Omega}$")
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(b.rec_potentials, origin="lower")
plt.title("SL in $\overline{\Omega} (old singular term)$")
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(u, origin="lower")
plt.title("u")
plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(b.rec_final, origin="lower")
plt.title("$\phi$")
plt.colorbar()

#%%
a =  bilinear_interpolation(np.array([1,0,0,0]), 20)
a =  bilinear_interpolation(np.array([0,1,0,0]), 20)

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.linspace(0,1,20)
Y = np.linspace(0,1,20)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y,a, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0,1)
ax.zaxis.set_major_locator(LinearLocator(3))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

ax.set_xlabel('$x$', fontsize=20, rotation=150)
ax.set_ylabel('$y$', fontsize=20)
ax.set_title("$B_1(\mathbf{x}_1)$")
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
