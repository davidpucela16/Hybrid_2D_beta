#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:14:22 2022

@author: pdavid
"""



import sys
# insert at 1, 0 is the script path (or '' in REPL)


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

def get_plots_through_sources(phi_mat, SS_phi_mat,pos_s, rec_x,rec_y):
    for i in pos_s:
        pos=coord_to_pos(rec_x, rec_y, i)
        pos_x=int(pos%len(rec_x))
        plt.plot(rec_y, phi_mat[:,pos_x], label="coupling")
        plt.plot(rec_y, SS_phi_mat[:,pos_x], label="validation")
        plt.axvline(x=i[1])
        plt.legend()
        plt.show()
        
def get_plots_through_sources_peaceman(phi_mat,peaceman,pos_s, rec_x,rec_y):
    for i in pos_s:
        pos=coord_to_pos(rec_x, rec_y, i)
        pos_x=int(pos%len(rec_x))
        plt.plot(rec_y, phi_mat[:,pos_x], label="coupling")
        plt.scatter(rec_y, peaceman[:,pos_x], label="validation")
        plt.plot()
        plt.axvline(x=i[1])
        plt.legend()
        plt.show()

#0-Set up the sources
#1-Set up the domain
D=1
L=5
cells=10
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=(50//cells)*2
#Rv=np.exp(-2*np.pi)*h_ss
Rv=0.01
C0=1
K_eff=1/(np.pi*Rv**2)

validation=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=5
print("directness=", directness)
#pos_s=np.array([[x_ss[2], y_ss[2]],[x_ss[4], y_ss[4]]])
#pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6],[2,2]])-np.array([0.25,0.25])
#pos_s/=2
#pos_s=np.array([[1.25,1.25],[1.25,1.75], [1.75,1.75],[1.75,1.25]])
#pos_s=np.array([[4.3,4.3],[4.3,5.5], [3.5,4.5],[3.5,3.5]])


pos_s=np.array([[0.506,0.506]])*L


#Position image
pos_s1=np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.96,0.43]])
pos_s2=np.array([[0.27,0.6],[0.55,0.65],[0.59,0.66],[0.67,0.67],[0.13,0.75],[0.15,0.93],[0.2,0.87],[0.28,0.98],[0.8,0.85],[0.83,0.92]])
pos_s=(np.concatenate((pos_s1, pos_s2))*0.8+0.1)*L
S=len(pos_s)

pos_s=(np.array([[2.4746205 , 3.84194922],
       [1.56678779, 1.85388858],
       [4.28902858, 5.38971457],
       [5.19870931, 1.67982104],
       [4.39671925, 4.37907858],
       [4.59228223, 3.44110346]])/7*0.6+0.2)*L


S=len(pos_s)
plt.scatter(pos_s[:,0], pos_s[:,1])
plt.title("Position of the point sources")
plt.ylabel("y ($\mu m$)")
plt.xlabel("x ($\mu m$)")
plt.show()

#%%

C_v_array=np.ones(S)

t=post.reconstruction_coupling(ratio, pos_s, Rv, h_ss,L, K_eff, D,directness)
t.solve_steady_state_problem(np.array([0,0,0,0]), np.ones(S))
t.retrieve_concentration_dual_mesh()
t.execute_interpolation(t.phi_q, t.phi_FV)
rec=t.rec_u+t.rec_bar_S

plt.imshow(rec, origin='lower', extent=[0,L,0,L])
plt.colorbar()
plt.title('Molecular concentration new')
plt.ylabel("y ($\mu m$)")
plt.xlabel("x ($\mu m$)")
plt.show()




#%%
#This cell is to produce the figure for the presentation
P=1/np.sqrt(np.pi)*L
xx=np.concatenate((np.linspace(Rv, P, 50)[::-1], np.linspace(Rv, P, 50)))

ref_FV=FV_reference(ratio, t.h, pos_s, np.ones(S), D, K_eff, Rv, L)
noc_sol, noc_lenx, noc_leny,noc_q, noc_B, noc_A, noc_s_blocks,noc_x,noc_y=ref_FV.sol, len(ref_FV.x), len(ref_FV.y), ref_FV.q_array, ref_FV.B, ref_FV.A, ref_FV.s_blocks, ref_FV.x, ref_FV.y

FV=noc_sol.reshape(noc_lenx, noc_leny)

Sbar=t.rec_bar_S
u=t.rec_u

plt.figure()
plt.plot(noc_x,u[50,:], label="u"); 
plt.plot(noc_x,Sbar[50,:], label="$\overline{S}$"); 
plt.plot(noc_x,rec[50,:], label="$\phi = \overline{S} + u$")
plt.plot(noc_x,FV[50,:], label="Reference FV")
tit="\n {}x{} mesh".format(cells, cells)
plt.title("Example of $\overline{S}$, u, $\phi$, for single source problem" + tit)
plt.legend()
plt.show()


#%%

t=assemble_SS_2D_FD(pos_s, Rv, h_ss,L, K_eff, D, directness)
t.pos_arrays()
t.initialize_matrices()
M=t.assembly_sol_split_problem(np.array([0,0,0,0]))
t.B[-S:]=C_v_array*C0
#t.B[-np.random.randint(0,S,int(S/2))]=0
sol=np.linalg.solve(M, t.B)
phi_FV=sol[:-S].reshape(len(t.x), len(t.y))
phi_q=sol[-S:]

# =============================================================================
# m=real_NN_rec(t.x, t.y, sol[:-len(pos_s)], t.pos_s, t.s_blocks, sol[-len(pos_s):], ratio, t.h, 1, t.Rv)
# m.add_singular(1)
# fin_rec=m.add_singular(1)+m.rec
# plt.imshow(fin_rec, origin='lower'); plt.colorbar()
# plt.show()
# print(fin_rec[:,-1])
# =============================================================================


#%%
#Reconstruction microscopic field
#pdb.set_trace()
a=post.reconstruction_sans_flux(sol, t, L, ratio, directness)

p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()
plt.imshow(a.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model linear old")
plt.colorbar(); plt.show()


#%%
n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
n.solve_linear_prob(np.zeros(4), C_v_array)

M=0.2
phi_0=0.4

p=np.linspace(0,1,100)
if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")


n.Full_Newton(np.ndarray.flatten(n.phi_FV) , np.ndarray.flatten(n.phi_q), 0.005, M, phi_0)
a=post.reconstruction_sans_flux(n.phi[-1], n, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()
plt.imshow(a.rec_final, origin='lower', vmax=np.max(a.rec_final))
plt.title("bilinear reconstruction \n coupling model Metabolism")
plt.colorbar(); plt.show()