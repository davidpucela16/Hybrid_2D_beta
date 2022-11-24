#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:56:23 2022

@author: pdavid
"""

# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
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
          'figure.figsize': (8,8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


#0-Set up the sources
#1-Set up the domain
alpha=80
D=1
K0=1
L=240

cells=5
h_ss=L/cells
ratio=int(40/cells)*2
print("ratio: ", ratio)
#ratio=12
#Rv=np.exp(-2*np.pi)*h_ss


print("alpha: {} must be greater than {}".format(alpha, 5*ratio*cells))
print("h coarse:",h_ss)



x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=0
print("directness=", directness)

pos_s=(np.array([[2.47 , 3.84],
       [1.56, 1.85],
       [4.29, 5.39],
       [5.2, 1.68],
       [4.39, 4.38],
       [4.59, 3.44]])/7*0.6+0.2)*L

#pos_s=np.array([[0.5,0.5]])*L
S=len(pos_s)


#Position image

vline=(y_ss[1:]+x_ss[:-1])/2
c=0
for i in pos_s:
    plt.scatter(i[0], i[1], label="{}".format(c))
    c+=1
plt.title("Position of the point sources")
for xc in vline:
    plt.axvline(x=xc, color='k', linestyle='--')
for xc in vline:
    plt.axhline(y=xc, color='k', linestyle='--')
plt.xlim([0,L])
plt.ylim([0,L])
plt.legend()
plt.ylabel("y ($\mu m$)")
plt.xlabel("x ($\mu m$)")
plt.show()

Rv=L/alpha+np.zeros(S)
K_eff=K0/(np.pi*(L/alpha)**2)


case_number=2
C_v_array=np.ones(S)   


if case_number==1:
    C_v_array[-1]=0
elif case_number==2:
    C_v_array[[0,-1]]=0
else:
    print("error")






#%%
n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
n.solve_linear_prob(np.zeros(4), C_v_array)




#%% 3
b=post.reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(n.phi_FV), n.phi_q)), n, L,ratio, directness)
p=b.reconstruction()   
b.reconstruction_boundaries(np.array([0,0,0,0]))
b.rec_corners()
#plt.imshow(b.rec_final, origin='lower')
plt.imshow(b.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Steady State ")
plt.colorbar(); plt.show()



#%%

vmax=np.max(b.rec_final)
vmin=np.min(b.rec_potentials)

u,pot=b.get_u_pot(C_v_array)
c=0
# =============================================================================
# for i in pos_s:
#     pos=coord_to_pos(b.x, b.y, i)
#     plt.plot(b.x,b.rec_final[pos//len(b.x),:],label="SS no metab")
#     plt.plot(b.x,u[pos//len(b.x),:], label="u")
#     plt.plot(b.x,pot[pos//len(b.x),:], label="SL + DL in $\overline{\Omega}$")
#     plt.plot(b.x,pot[pos//len(b.x),:], label="SL + DL in $\overline{\Omega}$")
#     
# 
#     plt.xlabel("y ($\mu m$)")
#     plt.ylabel("$\phi$")
#     plt.legend()
#     plt.title("Linear solution")
#     plt.show()
#     c+=1
# =============================================================================
    

#%%
COMSOL_q=np.array([0.4183,0.4978,0.4337,0.4802,0.3626,0.3736])
COMSOL_q_s=np.array([0.4679,	0.5255,	0.4813	,0.5435,	0.4655,	-0.2911])
COMSOL_q_2s=np.array([-0.1822,	0.583,	0.5257,	0.5703,	0.5189,	-0.2415])
COMSOL=np.vstack((COMSOL_q,COMSOL_q_s,COMSOL_q_2s))

Com_to_plot=COMSOL[case_number]

plt.scatter(np.arange(6),n.phi_q, label="MyCode")
plt.plot(Com_to_plot, label="COMSOL")
plt.legend()
plt.show()


print("Mean absolute error", np.sum(np.abs(Com_to_plot-n.phi_q))/6)

#%%
file=open("../../6_Sources_COMSOL/2D_data_2sinks.txt", "r")
a=file.read().splitlines()


FEM_x=np.array([], dtype=float)
FEM_y=np.array([], dtype=float)

value=np.array([], dtype=float)
for i in a:
    line=[float(k) for k in i.split()]
    value=np.append(value, line[-1])
    FEM_x=np.append(FEM_x, line[0])
    FEM_y=np.append(FEM_y, line[1])
    
#plt.scatter(position, value, label="comsol")
plt.scatter(b.x,b.rec_final[:,int(cells*ratio/2)],label="SS no metab")
plt.legend()


r=post.reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, directness); 
r.solve_linear_prob(np.zeros(4), C_v_array); 
r.set_up_manual_reconstruction_space(FEM_x, FEM_y)
r.reconstruction_manual()
r.reconstruction_boundaries(np.zeros(4))
phi=r.u+r.DL+r.SL
levels=np.linspace(0,np.max(phi)*1.1, 8)
plt.tricontourf(FEM_x, FEM_y, value, levels=levels); plt.colorbar(); plt.show()
plt.tricontourf(FEM_x, FEM_y, phi, levels=levels); plt.colorbar(); plt.show()
plt.tricontourf(FEM_x, FEM_y, phi-value); plt.colorbar(); plt.show()

print("L2 err phi= ", (np.sum((value-phi)**2)/np.sum(value**2))**0.5)
