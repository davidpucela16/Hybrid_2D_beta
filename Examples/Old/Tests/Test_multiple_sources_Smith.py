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

def get_plots_through_sources(phi_mat, SS_phi_mat,pos_s, rec_x,rec_y, orig_y):
    c=0
    vline=(orig_y[1:]+orig_y[:-1])/2
    for i in pos_s:
        pos=coord_to_pos(rec_x, rec_y, i)
        pos_x=int(pos%len(rec_x))
        plt.plot(rec_y, phi_mat[:,pos_x], label="coupling")
        plt.plot(rec_y, SS_phi_mat[:,pos_x], label="SS validation")
        plt.title("Concentration plot passing through source {}".format(c))
        plt.xlabel("position y ($\mu m$)")
        plt.ylabel("$\phi [kg m^{-1}]$")
        plt.axvline(x=i[1], color='r')
        for xc in vline:
            plt.axvline(x=xc, color='k', linestyle='--')
    
        plt.legend()
        plt.show()
        
def get_plots_through_sources_peaceman(phi_mat,peaceman,pos_s, rec_x,rec_y, orig_y):
    c=0
    vline=(orig_y[1:]+orig_y[:-1])/2
    for i in pos_s:
        pos=coord_to_pos(rec_x, rec_y, i)
        pos_x=int(pos%len(rec_x))
        plt.plot(rec_y, phi_mat[:,pos_x], label="coupling")
        plt.scatter(rec_y, peaceman[:,pos_x], label="Peaceman")
        plt.plot()
        plt.axvline(x=i[1], color='r')
        for xc in vline:
            plt.axvline(x=xc, color='k', linestyle='--')
        plt.title("Concentration plot passing through source {}".format(c))
        plt.xlabel("position y ($\mu m$)")
        plt.ylabel("$\phi [kg m^{-1}]$")
        plt.legend()
        plt.show()
        c+=1
        
#0-Set up the sources
#1-Set up the domain
D=1
L=240
cells=5
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=int(40/cells)*2
#Rv=np.exp(-2*np.pi)*h_ss






x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=2
print("directness=", directness)

#Position image
pos_s1=np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.96,0.43]])
pos_s2=np.array([[0.27,0.6],[0.55,0.65],[0.59,0.66],[0.67,0.67],[0.13,0.75],[0.15,0.93],[0.2,0.87],[0.28,0.98],[0.8,0.85],[0.83,0.92]])
pos_s=(np.concatenate((pos_s1, pos_s2))*0.8+0.1)*L


#pos_s=np.array([[0.5,0.5]])*L
S=len(pos_s)

alpha=50
Rv=np.zeros(S)+L/alpha
print("alpha= ", alpha)
K0=1
K_eff=K0/(np.pi*Rv**2)
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
#plt.legend()
plt.ylabel("y ($\mu m$)")
plt.xlabel("x ($\mu m$)")
plt.show()

C_v_array=np.ones(S)   
#C_v_array[[0,-1]]=0


#%%
n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
n.solve_linear_prob(np.zeros(4), C_v_array)
phi_FV=n.phi_FV
phi_q=n.phi_q



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

#From hear onwards I should validate it with the COMSOL simulation instead of the Peaceman and SS that I had before
