#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 12:54:16 2021

@author: pdavid
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 19:33:13 2021

@author: pdavid
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:52:20 2021

@author: pdavid
"""

import os

os.chdir('/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Validated_2D_Code/FV_metab_dimensional')
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
D=1
L=10
cells=5
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=12
#Rv=np.exp(-2*np.pi)*h_ss

alpha=100

Rv=L/alpha+np.zeros(2)

C0=2*np.pi*Rv*D
K_eff=C0/(np.pi*Rv**2)

x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
#directness=int(cells/4)
directness=2
print("directness=", directness)



#%%

array_of_pos=np.array([0.75,0.62,0.19,-0.165,-0.51])+L/2 #position of the second source 
q_array_source=np.zeros((0,2))
p1=np.array([0.9,0.9])+L/2

array_of_sep=np.sqrt(2*(array_of_pos-p1[0])**2)
for i in array_of_pos:
    p2=np.array([i,i])
    pos_s=np.array([p1,p2])
    print(pos_s-L/2)
    S=len(pos_s)
    t=assemble_SS_2D_FD(pos_s, Rv, h_ss,L, K_eff, D, directness)
    t.pos_arrays()
    t.initialize_matrices()
    M=t.assembly_sol_split_problem(np.array([0,0,0,0]))
    t.H0[-S:]=-np.ones(S)
    #t.B[-np.random.randint(0,S,int(S/2))]=0
    sol=np.linalg.solve(M, -t.H0)
    phi_FV=sol[:-S].reshape(len(t.x), len(t.y))
    phi_q=sol[-S:]
    
    q_array_source=np.concatenate((q_array_source, np.array([phi_q])), axis=0)

q_COMSOL_sources=np.array([0.7217,0.7591,0.8046,0.8248,0.8394])

#%%
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

#%%
q_array_sink=np.zeros((0,2))
for i in array_of_pos:
    p2=np.array([i,i])
    pos_s=np.array([p1,p2])
    print(pos_s-L/2)
    S=len(pos_s)
    t=assemble_SS_2D_FD(pos_s, Rv, h_ss,L, K_eff, D, directness)
    t.pos_arrays()
    t.initialize_matrices()
    M=t.assembly_sol_split_problem(np.array([0,0,0,0]))
    t.H0[-S:]=-np.ones(S)
    t.H0[-1]= 0
    #t.B[-np.random.randint(0,S,int(S/2))]=0
    sol=np.linalg.solve(M, -t.H0)
    phi_FV=sol[:-S].reshape(len(t.x), len(t.y))
    phi_q=sol[-S:]
    
    q_array_sink=np.concatenate((q_array_sink, np.array([phi_q])), axis=0)
    
q_COMSOL_sink=np.array([[0.4687,-0.1074],
                       [0.4654,-0.0852],
                       [0.4577,-0.0543],
                       [0.4552,-0.0416],
                       [0.4538,-0.0332]])

#%%
array_code=np.sum(q_array_source, axis=1)
array_COMSOL=q_COMSOL_sources


fig, axs = plt.subplots(1,3, figsize=(15,5),constrained_layout=True)
fig.suptitle("$\mathbb{C}_1 = 1$ and $\mathbb{C}_2 = 1$", fontsize=25)
c=axs[0].plot(array_of_sep/(L/alpha), array_code, label='MyCode')
axs[0].plot(array_of_sep/(L/alpha), array_COMSOL, label='COMSOL')
axs[0].legend()
axs[0].set_xlabel("$distance/R_v$")
axs[0].set_ylabel("q")
axs[0].set_title("$q_1+q_2$")

b=axs[1].plot(array_of_sep/(L/alpha),np.abs(array_code-array_COMSOL)/array_COMSOL)
axs[1].set_xlabel("$distance/R_v$")
axs[1].set_ylabel("rel error")
axs[1].set_title("relative error")


axs[2].plot(array_of_sep/(L/alpha),np.abs(array_code-array_COMSOL))
axs[2].set_ylabel("error")
axs[2].set_xlabel("$distance/R_v$")
axs[2].set_title("absolute error")

plt.show()

array_code=q_array_sink[:,0]
array_COMSOL=q_COMSOL_sink[:,0]


fig, axs = plt.subplots(2,3, figsize=(15,10),constrained_layout=True)
fig.suptitle("$\mathbb{C}_1 = 1$ and $\mathbb{C}_2 = 0$", fontsize=25)
axs[0,0].plot(array_of_sep/(L/alpha), array_code, label='MyCode')
axs[0,0].plot(array_of_sep/(L/alpha), array_COMSOL, label='COMSOL')
axs[0,0].legend()
axs[0,0].set_xlabel("$distance/R_v$")
axs[0,0].set_ylabel("q")
axs[0,0].set_title("$q_1 (source)$")

axs[0,1].plot(array_of_sep/(L/alpha),np.abs(array_code-array_COMSOL)/array_COMSOL)
axs[0,1].set_xlabel("$distance/R_v$")
axs[0,1].set_ylabel("rel error")
axs[0,1].set_title("relative error")


axs[0,2].plot(array_of_sep/(L/alpha),np.abs(array_code-array_COMSOL))
axs[0,2].set_ylabel("error")
axs[0,2].set_xlabel("$distance/R_v$")
axs[0,2].set_title("absolute error")

array_code=q_array_sink[:,1]
array_COMSOL=q_COMSOL_sink[:,1]

axs[1,0].plot(array_of_sep/(L/alpha), array_code, label='MyCode')
axs[1,0].plot(array_of_sep/(L/alpha), array_COMSOL, label='COMSOL')
axs[1,0].legend()
axs[1,0].set_xlabel("$distance/R_v$")
axs[1,0].set_ylabel("q")
axs[1,0].set_title("$q_2 (sink)$")

axs[1,1].plot(array_of_sep/(L/alpha),np.abs(array_code-array_COMSOL)/np.abs(array_COMSOL))
axs[1,1].set_xlabel("$distance/R_v$")
axs[1,1].set_ylabel("rel error")
axs[1,1].set_title("relative error")


axs[1,2].plot(array_of_sep/(L/alpha),np.abs(array_code-array_COMSOL))
axs[1,2].set_ylabel("error")
axs[1,2].set_xlabel("$distance/R_v$")
axs[1,2].set_title("absolute error")
plt.show()

