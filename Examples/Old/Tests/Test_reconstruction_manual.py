#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:09:35 2022

@author: pdavid
"""


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
alpha=500

D=1
K0=1


L=3

cells=5
h_ss=L/cells
Rv=L/alpha

print("h coarse:",h_ss)
K_eff=K0/(np.pi*Rv**2)


x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=1
print("directness=", directness)

pos_s=np.array([[0.3,0.3],[0.5,0.5],[0.3,0.5]])*L
S=len(pos_s)

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

C_v_array=np.ones(S) -1  


#%%

x_fine=np.linspace(0,L,100)
y_fine=np.linspace(0,L,100)

temp_x=np.array([])
temp_y=np.array([])

for i in x_fine:
    temp_x=np.concatenate((temp_x, x_fine))
    temp_y=np.concatenate((temp_y, np.zeros(len(x_fine))+i))


r=post.reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, 2); 
r.solve_linear_prob(np.zeros(4), np.array([1])); 
r.set_up_manual_reconstruction_space(temp_x, temp_y)
r.reconstruction_manual()
r.reconstruction_boundaries(np.zeros(4))
phi=r.u+r.DL+r.SL
plt.imshow(phi.reshape(100,100), origin='lower'); plt.colorbar()

#%%

x=np.linspace(0.5,9.5,10)
y=x.copy()

a=post.bilinear_interpolation(np.arange(4), 10)

rec_single=np.array([])

for j in y:
    for i in x:
        rec_single=np.append(rec_single, single_value_bilinear_interpolation(np.array([i,j]), np.array([[0,0],[0,1],[1,0],[1,1]])*10, np.arange(4)))