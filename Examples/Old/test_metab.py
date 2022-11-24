#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:18:37 2022

@author: pdavid
"""
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
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


#0-Set up the sources
#1-Set up the domain
alpha=1000

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
directness=1
print("directness=", directness)
#pos_s=np.array([[x_ss[2], y_ss[2]],[x_ss[4], y_ss[4]]])
#pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6],[2,2]])-np.array([0.25,0.25])
#pos_s/=2
#pos_s=np.array([[1.25,1.25],[1.25,1.75], [1.75,1.75],[1.75,1.25]])
#pos_s=np.array([[4.3,4.3],[4.3,5.5], [3.5,4.5],[3.5,3.5]])


#pos_s=np.array([[0.41,0.41],[0.7,0.7],[0.3,0.47],[0.8,0.2]])*L
pos_s1=np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.96,0.43]])
pos_s2=np.array([[0.27,0.6],[0.55,0.65],[0.59,0.66],[0.67,0.67],[0.13,0.75],[0.15,0.93],[0.2,0.87],[0.28,0.98],[0.8,0.85],[0.83,0.92]])
pos_s=(np.concatenate((pos_s1, pos_s2))*0.8+0.1)*L
pos_s=(np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.96,0.43]])*0.6+0.2)*L
#pos_s=np.array([[0.5,0.5]])*L
pos_s=(np.random.random((6,2))*0.6+0.2)*L

pos_s=(np.array([[2.47 , 3.84],
       [1.56, 1.85],
       [4.29, 5.39],
       [5.2, 1.68],
       [4.39, 4.38],
       [4.59, 3.44]])/7*0.6+0.2)*L

#pos_s=np.array([[0.47,0.47],[0.53,0.53]])*L
pos_s=np.array([[0.5,0.5]])*L
S=len(pos_s)

Rv=L/alpha+np.zeros(S)

print("alpha: {} must be greater than {}".format(alpha, 5*ratio*cells))
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


p=np.linspace(0,1,100)
if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")


C_v_array=np.ones(S)   

#%% 1 

FV=FV_validation(L, cells*ratio, pos_s, C_v_array, D, K_eff, Rv, np.zeros(4))
FV_linear=FV.solve_linear_system()
FV_linear_mat=FV_linear.reshape(cells*ratio, cells*ratio)

#%%
print("R: ", 1/(1/K0 + np.log(0.2*FV.h/Rv)/(2*np.pi*D)))

#%% Plots FV reference solution - Peaceman Coupling

plt.imshow(FV_linear_mat, origin='lower')
plt.colorbar()
plt.title("FV reference solution, linear system\n mesh:{}x{}".format(ratio*cells, ratio*cells))
plt.show()


#%% 5
FV_non_linear=FV.solve_non_linear_system(phi_0,M, stabilization)
#phi_FV=FV_linear.reshape(cells*ratio, cells*ratio)
phi_FV=(FV.phi[-1]+FV.Corr_array).reshape(cells*ratio, cells*ratio)
phi_SS=(FV.phi[0]+FV.Corr_array).reshape(cells*ratio, cells*ratio)


#%%

plt.imshow(phi_SS, origin='lower', vmax=np.max(phi_FV))
plt.title("FV linear reference")
plt.colorbar(); plt.show()


#%%
plt.imshow(phi_FV, origin='lower', vmax=np.max(phi_FV))
plt.title("FV metab reference")
plt.colorbar(); plt.show()


#%% 2

n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
n.solve_linear_prob(np.zeros(4), C_v_array)


#manual q
q_array=-np.dot(FV.A_virgin.toarray()[FV.s_blocks,:],FV.phi[-1])*FV.h**2/D+M*(1-phi_0/(FV.phi[-1, FV.s_blocks]+FV.Corr_array[FV.s_blocks]+phi_0))

#%%
print("MRE steady state system", get_MRE(n.phi_q, FV.get_q(FV_linear)))

plt.plot(n.phi_q, label="MyCode")
plt.plot(FV.get_q(FV_linear), label="FV reference")
plt.legend()
plt.show()



#%% 3
b=post.reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(n.phi_FV), n.phi_q)), n, L,ratio, directness)
p=b.reconstruction()   
b.reconstruction_boundaries(np.array([0,0,0,0]))
b.rec_corners()
plt.imshow(b.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Steady State ")
plt.colorbar(); plt.show()



#%%
for i in pos_s:
    pos=coord_to_pos(FV.x, FV.y, i)
    
    plt.plot(FV_linear_mat[pos//len(FV.x),:], label="FV")
    plt.plot(b.rec_final[pos//len(FV.x),:],label="SS no metab")
    plt.legend()
    plt.title("Linear solution")
    plt.show()

#%% 4
n.Full_Newton(np.ndarray.flatten(n.phi_FV) , np.ndarray.flatten(n.phi_q), conver_residual, M, phi_0)
a=post.reconstruction_sans_flux(n.phi[-1], n, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()
plt.imshow(a.rec_final, origin='lower', vmax=np.max(a.rec_final))
plt.title("bilinear reconstruction \n coupling model Metabolism")
plt.colorbar(); plt.show()


n.assemble_it_matrices_Sampson(n.u, n.q)



#%%
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





