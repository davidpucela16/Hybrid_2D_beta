#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This tests if made for 
"""

import os
os.chdir('/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Validated_2D_Code/FV_metab_dimensional/Tests')
os.chdir('..')
directory=os.getcwd()

import numpy as np 
import matplotlib.pyplot as plt
from module_2D_coupling_FV_nogrid import * 
import reconst_and_test_module as post
import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab

import pandas 
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6,6 ),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

#0-Set up the sources
#1-Set up the domain
D=1
L=10
cells=14
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=12
#Rv=np.exp(-2*np.pi)*h_ss

alpha=50

Rv=L/alpha+np.zeros(2)
Rv[0]/=2

K0=1

C_comsol=alpha*K0/(2*np.pi*L)

K_eff=alpha*K0/(np.pi*Rv*L)



x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=4
print("directness=", directness)



#%% Set up of the flux values and the separation arrays
if alpha==100:
    q_COMSOL_sources=np.array([[0.4602,0.4912,0.5297,0.5601,0.5783],[0.4587,0.4889,0.5258,0.5566,0.5793]]).T
    q_COMSOL_sink=np.array([[-0.2081,-0.1624,-0.1025,-0.0619,-0.0402],[0.6668,0.6513,0.6283,0.6185,0.6195]]).T

if alpha==50:
    q_COMSOL_sources=np.array([[0.2895,0.3125,0.3426,0.3644,0.3757],[0.5789,0.5946,0.6201,0.6558,0.7087]]).T
    q_COMSOL_sink=np.array([[-0.1028,-0.0763,-0.0409,-0.0172,-0.0054],[0.6817,0.6709,0.6611,0.673,0.7141]]).T
    


d=np.array([2.1,4,10,20,30])  #array of the separations!!
q_array_source=np.zeros((0,2))
p1=np.array([0.1,0.1])*L+L/2

l=d*(L/alpha)/np.sqrt(2)

array_of_pos=p1[0]-l
#%% First simulations for all sources 


C_v_array=np.array([1,1])
c=0
for i in array_of_pos:
    
    p2=np.array([i,i])
    pos_s=np.array([p1,p2])
    S=len(pos_s)
    
    r=post.reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, directness)
    r.solve_linear_prob(np.zeros(4), C_v_array)
    phi_FV=r.phi_FV #values on the FV cells
    phi_q=r.phi_q #values of the flux
    
    q_array_source=np.vstack((q_array_source, phi_q))
    
    
    if c==0 or c==4:
        sep=int(d[c])
        file='../' + 'Double_source_COMSOL/Double_source_diff/Double_source_alpha' +  str(alpha) + '_2D_d' + str(sep) + '_sources_diff.txt'
        df=pandas.read_fwf(file)
        ref_data=np.array(df).T #reference 2D data from COMSOL
        phi_bar_COMSOL=np.max(ref_data[2])
        r.set_up_manual_reconstruction_space(ref_data[0], ref_data[1])
        r.reconstruction_manual()
        r.reconstruction_boundaries(np.zeros(4))
        phi_MyCode=r.u+r.DL+r.SL
        
        lev=np.linspace(-0.05,np.max(ref_data[-1]),8)
        
        fig, axs = plt.subplots(1,2, figsize=(18,8),constrained_layout=True)
        fig.suptitle("Comparison of the contours for d={}, alpha={}".format(sep, alpha), fontsize=25)
        im=axs[0].tricontourf(ref_data[0], ref_data[1], ref_data[2], levels=lev)
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].set_title("COMSOL reference")
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='10%', pad=0.15)
        fig.colorbar(im, cax=cax,orientation='vertical')
        
        im2=axs[1].tricontourf(ref_data[0], ref_data[1], phi_MyCode,levels=lev)
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].set_title("MyCode")
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='10%', pad=0.15)
        fig.colorbar(im2, cax=cax,orientation='vertical')
        plt.show()
        
    c+=1


#%%

col=[  'pink','c', 'blue']
fig, ax=plt.subplots()

side=(directness+0.5)*h_ss*2

vline=(y_ss[1:]+x_ss[:-1])/2
plt.scatter(pos_s[:,0], pos_s[:,1], s=100, c='r')
for c in range(len(pos_s)):
    center=pos_to_coords(r.x, r.y, r.s_blocks[c])
    
    ax.add_patch(Rectangle(tuple(center-side/2), side, side,
                 edgecolor = col[c],
                 facecolor = col[c],
                 fill=True,
                 lw=5, zorder=0))
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

#%%

array_code=q_array_source[:,0]
array_COMSOL=q_COMSOL_sources[:,0]

fig, axs = plt.subplots(2,3, figsize=(30,20),constrained_layout=True)
fig.suptitle("$\mathbb{C}_1 = 1$ and $\mathbb{C}_2 = 1$", fontsize=25)
axs[0,0].plot(d/(L/alpha), array_code, label='MyCode')
axs[0,0].plot(d/(L/alpha), array_COMSOL, label='COMSOL')
axs[0,0].legend()
axs[0,0].set_xlabel("$distance/R_v$")
axs[0,0].set_ylabel("q")
axs[0,0].set_title("$q_1 source (Up right)$")

axs[0,1].plot(d/(L/alpha),np.abs(array_code-array_COMSOL)/array_COMSOL)
axs[0,1].set_xlabel("$distance/R_v$")
axs[0,1].set_ylabel("rel error")
axs[0,1].set_title("relative error")


axs[0,2].plot(d/(L/alpha),np.abs(array_code-array_COMSOL))
axs[0,2].set_ylabel("error")
axs[0,2].set_xlabel("$distance/R_v$")
axs[0,2].set_title("absolute error")

array_code=q_array_source[:,1]
array_COMSOL=q_COMSOL_sources[:,1]

axs[1,0].plot(d/(L/alpha), array_code, label='MyCode')
axs[1,0].plot(d/(L/alpha), array_COMSOL, label='COMSOL')
axs[1,0].legend()
axs[1,0].set_xlabel("$distance/R_v$")
axs[1,0].set_ylabel("q")
axs[1,0].set_title("$q_2 (source)$")

axs[1,1].plot(d/(L/alpha),np.abs(array_code-array_COMSOL)/np.abs(array_COMSOL))
axs[1,1].set_xlabel("$distance/R_v$")
axs[1,1].set_ylabel("rel error")
axs[1,1].set_title("relative error")


axs[1,2].plot(d/(L/alpha),np.abs(array_code-array_COMSOL))
axs[1,2].set_ylabel("error")
axs[1,2].set_xlabel("$distance/R_v$")
axs[1,2].set_title("absolute error")
plt.show()

#%%
q_array_sink=np.zeros((0,2))
C_v_array=np.array([0,1])
c=0
for i in array_of_pos:
    
    p2=np.array([i,i])
    pos_s=np.array([p1,p2])
    S=len(pos_s)
    
    r=post.reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, directness)
    r.solve_linear_prob(np.zeros(4), C_v_array)
    phi_FV=r.phi_FV #values on the FV cells
    phi_q=r.phi_q #values of the flux
    
    q_array_sink=np.concatenate((q_array_sink, np.array([phi_q])), axis=0)
    if c==0 or c==4:
        sep=int(d[c])
        file='../../' + 'Double_source_COMSOL/Double_source_diff/Double_source_alpha' +  str(alpha) + '_2D_d' + str(sep) + '_SourceSink_diff.txt'
        df=pandas.read_fwf(file)
        ref_data=np.array(df).T #reference 2D data from COMSOL
        phi_bar_COMSOL=np.max(ref_data[2])
        r.set_up_manual_reconstruction_space(ref_data[0], ref_data[1])
        r.reconstruction_manual()
        r.reconstruction_boundaries(np.zeros(4))
        phi_MyCode=r.u+r.DL+r.SL
        
        lev=np.linspace(-0.05,np.max(ref_data[-1]),8)
        
        fig, axs = plt.subplots(1,2, figsize=(18,8),constrained_layout=True)
        fig.suptitle("Comparison of the contours for d={}, alpha={}".format(sep, alpha), fontsize=25)
        im=axs[0].tricontourf(ref_data[0], ref_data[1], ref_data[2], levels=lev)
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].set_title("COMSOL reference")
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='10%', pad=0.15)
        fig.colorbar(im, cax=cax,orientation='vertical')
        
        im2=axs[1].tricontourf(ref_data[0], ref_data[1], phi_MyCode,levels=lev)
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].set_title("MyCode")
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='10%', pad=0.15)
        fig.colorbar(im2, cax=cax,orientation='vertical')
        plt.show()
    c+=1

#%%
a=post.reconstruction_sans_flux(np.concatenate((phi_FV, phi_q)), r, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()
plt.imshow(a.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model")
plt.colorbar(); plt.show()
#%%





array_code=q_array_sink[:,0]
array_COMSOL=q_COMSOL_sink[:,0]


fig, axs = plt.subplots(2,3, figsize=(25,15),constrained_layout=True)
fig.suptitle("$\mathbb{C}_1 = 1$ and $\mathbb{C}_2 = 0$", fontsize=25)
axs[0,0].plot(d/(L/alpha), array_code, label='MyCode')
axs[0,0].plot(d/(L/alpha), array_COMSOL, label='COMSOL')
axs[0,0].legend()
axs[0,0].set_xlabel("$distance/R_v$")
axs[0,0].set_ylabel("q")
axs[0,0].set_title("$q_1 (source)$")

axs[0,1].plot(d/(L/alpha),np.abs(array_code-array_COMSOL)/array_COMSOL)
axs[0,1].set_xlabel("$distance/R_v$")
axs[0,1].set_ylabel("rel error")
axs[0,1].set_title("relative error")


axs[0,2].plot(d/(L/alpha),np.abs(array_code-array_COMSOL))
axs[0,2].set_ylabel("error")
axs[0,2].set_xlabel("$distance/R_v$")
axs[0,2].set_title("absolute error")

array_code=q_array_sink[:,1]
array_COMSOL=q_COMSOL_sink[:,1]

axs[1,0].plot(d/(L/alpha), array_code, label='MyCode')
axs[1,0].plot(d/(L/alpha), array_COMSOL, label='COMSOL')
axs[1,0].legend()
axs[1,0].set_xlabel("$distance/R_v$")
axs[1,0].set_ylabel("q")
axs[1,0].set_title("$q_2 (sink)$")

axs[1,1].plot(d/(L/alpha),np.abs(array_code-array_COMSOL)/np.abs(array_COMSOL))
axs[1,1].set_xlabel("$distance/R_v$")
axs[1,1].set_ylabel("rel error")
axs[1,1].set_title("relative error")


axs[1,2].plot(d/(L/alpha),np.abs(array_code-array_COMSOL))
axs[1,2].set_ylabel("error")
axs[1,2].set_xlabel("$distance/R_v$")
axs[1,2].set_title("absolute error")
plt.show()
