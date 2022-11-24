#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This tests if made for 
"""

import os 
directory='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Updated_BCs/Code'
os.chdir(directory)

import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos, get_MRE

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
D=1
L=10
cells=5
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=20
#Rv=np.exp(-2*np.pi)*h_ss

alpha=100
diff_radii=True
sources=True #If both are source 
if sources:
    C_v_array=np.ones(2)
else:
    C_v_array=np.array([1,0])


if diff_radii==True:
    Rv=np.array([L/alpha, L/alpha/2]) #The small one is set as sink
    if sources:
        if alpha==50:
            q_COMSOL=np.array([[0.5748,0.5935,0.6052,0.6216,0.6341,0.6451,0.6557],[0.2849,0.3101,0.324,0.3416,0.3534,0.3624,0.3698]]).T
        elif alpha==100:
            q_COMSOL=np.array([[0.525,0.5396,0.5486,0.5611,0.5702,0.5775,0.5838],[0.2603,0.2821,0.294,0.3092,0.3194,0.3271,0.3334]]).T
    else:
        if alpha==50:
            q_COMSOL=np.array([[0.6812,0.671,0.6661,0.6628,0.6631,0.6657,0.6703],[-0.1064,-0.0774,-0.061,-0.0412,-0.029,-0.0206,-0.0146]]).T
        
        if alpha==100:
            q_COMSOL=np.array([[0.6478,0.6352,0.6287,0.6222,0.6192,0.6178,0.6174],[-0.1228,-0.0956,-0.0801,-0.0611,-0.0491,-0.0404,-0.0336]]).T
            
else:
    Rv=np.zeros(2)+L/alpha
    if alpha==50:
        q_COMSOL=np.array([[0.5005,0.5447,0.566,0.5941,0.6142,0.6305,0.6452],[0.5005,0.5447,0.566,0.5941,0.6142,0.6305,0.6452]]).T
    elif alpha==100:
        q_COMSOL=np.array([[0.4507,0.4507,0.5031,0.5252,0.5406,0.5526,0.5627],[0.4507,0.4507,0.5031,0.5252,0.5406,0.5526,0.5627]]).T
K0=1

K_eff=alpha*K0/(np.pi*L*Rv)

x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=1
print("directness=", directness)

d=np.array([2,4,6,10,14,18,22])  #array of the separations!!
dist=d*L/alpha
q_array_source=np.zeros((0,2))

p1=np.array([L*0.5,L*0.5])
both_sources=True

#%%
directory_files='../Double_source_COMSOL/Double_source_diff/alpha' + str(alpha)
if sources:
     directory_files+="/sources"
else:
    directory_files+="/SourceSink"

#%% TO DELETE LATER
# =============================================================================
# q_MyCode=np.zeros((0,2))
# L2_array=np.array([])
# i=dist[k]
# pos_s=np.array([[0.5*L-i/2, 0.5*L],[0.5*L+i/2, 0.5*L]])
# print(pos_s)
# 
# 
# r=post.reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, directness)
# r.solve_linear_prob(np.zeros(4), C_v_array)
# phi_FV=r.phi_FV #values on the FV cells
# phi_q=r.phi_q #values of the flux
# 
# file=directory_files + '/d{}_2D.txt'.format(int(d[k]))
# df=pandas.read_fwf(file)
# ref_data=np.array(df).T #reference 2D data from COMSOL
# 
# r.set_up_manual_reconstruction_space(ref_data[0], ref_data[1])
# r.reconstruction_manual()
# r.reconstruction_boundaries(np.zeros(4))
# phi_MyCode=r.u+r.DL+r.SL
#   
# file_1D=directory_files + '/d{}_1D.txt'.format(int(d[k]))
# df_1D=pandas.read_fwf(file_1D)
# data_1D=np.array(df_1D).T #reference 2D data from COMSOL
# r.set_up_manual_reconstruction_space(data_1D[0], np.zeros(len(data_1D[0]))+L/2)
# r.reconstruction_manual()
# r.reconstruction_boundaries(np.zeros(4))
# phi_MyCode_1D=r.u+r.DL+r.SL
# 
# fig, axs=plt.subplots(2,3, figsize=(16,8))
# 
# col=[  'pink','c', 'blue']
# side=(directness+0.5)*h_ss*2
# vline=(y_ss[1:]+x_ss[:-1])/2
# axs[0,0].scatter(pos_s[:,0], pos_s[:,1], s=100, c='r')
# for c in range(len(pos_s)):
#     center=pos_to_coords(r.x, r.y, r.s_blocks[c])
#     
#     axs[0,0].add_patch(Rectangle(tuple(center-side/2), side, side,
#                  edgecolor = col[c],
#                  facecolor = col[c],
#                  fill=True,
#                  lw=5, zorder=0))
# axs[0,0].set_title("Position of the point sources")
# for xc in vline:
#     axs[0,0].axvline(x=xc, color='k', linestyle='--')
# for xc in vline:
#     axs[0,0].axhline(y=xc, color='k', linestyle='--')
# axs[0,0].set_xlim([0,L])
# axs[0,0].set_ylim([0,L])
# axs[0,0].set_ylabel("y ($\mu m$)")
# axs[0,0].set_xlabel("x ($\mu m$)")
# phi_1D_COMSOL=data_1D[1,:-1].astype(float)
# axs[0,1].scatter(data_1D[0,:-1],phi_1D_COMSOL , s=5, label='COMSOL')
# axs[0,1].scatter(data_1D[0,:-1],phi_MyCode_1D[:-1], s=5)
# axs[0,1].legend()
# 
# axs[0,2].scatter(data_1D[0,:-1],np.abs(phi_1D_COMSOL-phi_MyCode_1D[:-1]))
# 
# levs=np.linspace(0, np.max(ref_data[2]),100)
# axs[1,0].tricontourf(ref_data[0], ref_data[1], ref_data[2],levels=levs)
# axs[1,0].set_title("COMSOL")
# axs[1,1].tricontourf(ref_data[0], ref_data[1], phi_MyCode,levels=levs)
# axs[1,1].set_title("MYCode")
# axs[1,2].tricontourf(ref_data[0], ref_data[1], np.abs(ref_data[2]-phi_MyCode),levels=levs/10)
# 
# print("relative error for each flux estimation", (phi_q-q_COMSOL[k,:])/q_COMSOL[k,:])
# L2=np.sum((phi_MyCode-ref_data[2])**2/np.sum(ref_data[2]**2))**0.5
# print("L2 norm for the $\phi$-field",L2 )
# L2_array=np.append(L2_array, L2)
# 
# q_MyCode=np.vstack((q_MyCode, phi_q))
# =============================================================================
#%%
q_MyCode=np.zeros((0,2))
L2_array=np.array([])
if diff_radii:
    for k in range(len(d)):
        i=dist[k]
        pos_s=np.array([[0.5*L-i/2, 0.5*L],[0.5*L+i/2, 0.5*L]])
        print(pos_s)
        
        r=post.reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, directness)
        r.solve_linear_prob(np.zeros(4), C_v_array)
        phi_FV=r.phi_FV #values on the FV cells
        phi_q=r.phi_q #values of the flux
        
        file=directory_files + '/d{}_2D.txt'.format(int(d[k]))
        df=pandas.read_fwf(file)
        ref_data=np.array(df).T #reference 2D data from COMSOL
        
        r.set_up_manual_reconstruction_space(ref_data[0], ref_data[1])
        r.reconstruction_manual()
        r.reconstruction_boundaries(np.zeros(4))
        phi_MyCode=r.u+r.DL+r.SL
          
        file_1D=directory_files + '/d{}_1D.txt'.format(int(d[k]))
        df_1D=pandas.read_fwf(file_1D)
        data_1D=np.array(df_1D).T #reference 2D data from COMSOL
        r.set_up_manual_reconstruction_space(data_1D[0], np.zeros(len(data_1D[0]))+L/2)
        r.reconstruction_manual()
        r.reconstruction_boundaries(np.zeros(4))
        phi_MyCode_1D=r.u+r.DL+r.SL
        
        fig, axs=plt.subplots(2,3, figsize=(16,8))
        
        col=[  'pink','c', 'blue']
        side=(directness+0.5)*h_ss*2
        vline=(y_ss[1:]+x_ss[:-1])/2
        axs[0,0].scatter(pos_s[:,0], pos_s[:,1], s=100, c='r')
        for c in range(len(pos_s)):
            center=pos_to_coords(r.x, r.y, r.s_blocks[c])
            
            axs[0,0].add_patch(Rectangle(tuple(center-side/2), side, side,
                         edgecolor = col[c],
                         facecolor = col[c],
                         fill=True,
                         lw=5, zorder=0))
        axs[0,0].set_title("Position of the point sources")
        for xc in vline:
            axs[0,0].axvline(x=xc, color='k', linestyle='--')
        for xc in vline:
            axs[0,0].axhline(y=xc, color='k', linestyle='--')
        axs[0,0].set_xlim([0,L])
        axs[0,0].set_ylim([0,L])
        axs[0,0].set_ylabel("y ($\mu m$)")
        axs[0,0].set_xlabel("x ($\mu m$)")
        phi_1D_COMSOL=data_1D[1,:-1].astype(float)
        axs[0,1].scatter(data_1D[0,:-1],phi_1D_COMSOL , s=5, label='COMSOL')
        axs[0,1].scatter(data_1D[0,:-1],phi_MyCode_1D[:-1], s=5)
        axs[0,1].legend()
        
        axs[0,2].scatter(data_1D[0,:-1],np.abs(phi_1D_COMSOL-phi_MyCode_1D[:-1]))
        
        levs=np.linspace(0, np.max(ref_data[2]),100)
        axs[1,0].tricontourf(ref_data[0], ref_data[1], ref_data[2],levels=levs)
        axs[1,0].set_title("COMSOL")
        axs[1,1].tricontourf(ref_data[0], ref_data[1], phi_MyCode,levels=levs)
        axs[1,1].set_title("MYCode")
        axs[1,2].tricontourf(ref_data[0], ref_data[1], np.abs(ref_data[2]-phi_MyCode),levels=levs/10)
        
        print("relative error for each flux estimation", (phi_q-q_COMSOL[k,:])/q_COMSOL[k,:])
        L2=np.sum((phi_MyCode-ref_data[2])**2/np.sum(ref_data[2]**2))**0.5
        print("L2 norm for the $\phi$-field",L2 )
        L2_array=np.append(L2_array, L2)
        
        q_MyCode=np.vstack((q_MyCode, phi_q))
        
else:
    for k in range(len(d)):
        i=dist[k]
        pos_s=np.array([[0.5*L-i/2, 0.5*L],[0.5*L+i/2, 0.5*L]])
        print(pos_s)
        
        C_v_array=np.ones(len(pos_s))
        
        r=post.reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, directness)
        r.solve_linear_prob(np.zeros(4), C_v_array)
        phi_FV=r.phi_FV #values on the FV cells
        phi_q=r.phi_q #values of the flux
        
        q_MyCode=np.vstack((q_MyCode, phi_q))
        
#%%
if diff_radii:
    fig, ax=plt.subplots(2,3, figsize=(16,8))
    
    ax[0,0].plot(d, q_COMSOL[:,0], label='Comsol')
    ax[0,0].plot(d, q_MyCode[:,0], label='MyCode')
    ax[0,0].legend()
    
    ax[0,1].plot(d, q_COMSOL[:,1], label='Comsol')
    ax[0,1].plot(d, q_MyCode[:,1], label='MyCode')
    ax[0,1].legend()
    
    ax[0,2].plot(d, L2_array, label='$L_2$-error')
    
    ax[1,0].scatter(d, np.abs(q_MyCode[:,0]-q_COMSOL[:,0])/q_COMSOL[:,0])
    
    ax[1,1].scatter(d, np.abs(q_MyCode[:,1]-q_COMSOL[:,1])/q_COMSOL[:,1])

else:
    fig, ax=plt.subplots(2,2, figsize=(16,8))
    
    ax[0,0].plot(d, q_COMSOL[:,0], label='Comsol')
    ax[0,0].plot(d, q_MyCode[:,0], label='MyCode')
    ax[0,0].legend()
    
    ax[0,1].plot(d, q_COMSOL[:,1], label='Comsol')
    ax[0,1].plot(d, q_MyCode[:,1], label='MyCode')
    ax[0,1].legend()
    
    
    ax[1,0].plot(d, np.abs(q_MyCode[:,0]-q_COMSOL[:,0])/q_COMSOL[:,0])
    
    ax[1,1].plot(d, np.abs(q_MyCode[:,1]-q_COMSOL[:,1])/q_COMSOL[:,1])


