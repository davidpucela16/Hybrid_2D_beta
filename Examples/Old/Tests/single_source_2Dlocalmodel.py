#!/usr/bin/env python
# coding: utf-8

# As the name of the file indicates, here the tests for the local operator splitting technique for a single source and a coarse mesh is done. Several off-centering tests are included, and the validation is made with two "point-source" schemes that are the Peaceman coupling model [D.W.Peaceman, 1978], and the full Solution Splitting model [Gjerde et al., 2019 ENSAIM]
# I decided to only do the validation with the refined Peaceman solution, since for one source it is quite accurate
# In[6]:

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas 
from matplotlib.patches import Rectangle


#from tabulate import tabulate

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10,10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.rcParams['font.size'] = '20'

        
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
#Rv=np.exp(-2*np.pi)*h_ss

case_number=0

alpha=np.array([100,800])[case_number]

K0=1

ratio=16

directory='../Single_source_COMSOL/'

# In[8]:


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


pos_s=np.array([[1,1]])*0.5*L
S=len(pos_s)

Rv=np.zeros(S)+L/alpha
K_eff=1/(np.pi*Rv**2)
vline=(y_ss[1:]+x_ss[:-1])/2
fig, ax=plt.subplots()

side=(directness+0.5)*h_ss*2
lower_corner=tuple(np.squeeze(pos_s)-side/2)
plt.scatter(pos_s[:,0], pos_s[:,1], color='r')
ax.add_patch(Rectangle(lower_corner, side, side,
             edgecolor = 'pink',
             facecolor = 'blue',
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
r=post.reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, directness)
r.solve_linear_prob(np.zeros(4), np.array([1]))
phi_FV=r.phi_FV #values on the FV cells
phi_q=r.phi_q #values of the flux



if alpha==100:
    phi_q_COMSOL=0.6117
elif alpha==800:
    phi_q_COMSOL=0.5088



# In[10]: Comparison with COMSOL
file=directory + 'SingleSource_alpha' + str(alpha) + '_2D_center.txt'
df=pandas.read_fwf(file)
ref_data=np.array(df).T #reference 2D data from COMSOL
phi_bar_COMSOL=np.max(ref_data[2])

r.set_up_manual_reconstruction_space(ref_data[0], ref_data[1])
r.reconstruction_manual()
r.reconstruction_boundaries(np.zeros(4))
phi_MyCode=r.u+r.DL+r.SL
u=r.u
DL=r.DL
SL=r.SL

#%% Comparison of the contour

plt.tricontourf(ref_data[0], ref_data[1], ref_data[2]-phi_MyCode); plt.colorbar()
plt.title("Contour of the absolute error commited \n Comarison with COMSOL solution $10^4$ unknowns \n")


#%% 
#####################################################################################################"
#   OLD RECONSTRUCTION ALGORITHM FOR THE COMPARISON WITH THE PEACEMAN SOLUTION
#############################################################################################

#Reconstruction microscopic field
#pdb.set_trace()
a=post.reconstruction_sans_flux(np.concatenate((phi_FV, phi_q)), r, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()
plt.imshow(a.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model")
plt.colorbar(); plt.show()

#A posteriori estimation of the phi_bar in my code
phi_bar=1-phi_q/K0


# In[ ]:
#comparison with Peaceman
FV=FV_validation(L, cells*ratio, pos_s, np.array([1]), D, K_eff, Rv)
FV_linear=FV.solve_linear_system()
#
FV_linear_plot=FV_linear.copy() #to plot since it has \bar{\phi}
FV_linear_plot[FV.s_blocks]-=FV.get_q(FV_linear)*np.log(0.2*alpha/cells/ratio)/(2*np.pi*D)
FV_linear_plot=FV_linear_plot.reshape(cells*ratio, cells*ratio)


FV_linear_mat=FV_linear.reshape(cells*ratio, cells*ratio)
p_q=FV.get_q(FV_linear)

p_x, p_y=FV.x, FV.y

#%%
print("relative error q ref Peaceman:", np.abs((phi_q-p_q)/p_q))
print("relative error q ref Comsol:", np.abs((phi_q-phi_q_COMSOL)/phi_q_COMSOL))


# In[ ]:

get_plots_through_sources_peaceman(a.rec_final,FV_linear_plot,pos_s, p_x,p_y,r.y)
plt.show()

#%%

file=directory + 'SingleSource_alpha' + str(alpha) + '_1D_center.txt'

df=pandas.read_fwf(file)
data_1D=np.array(df).T #reference 2D data from COMSOL

plt.plot(data_1D[0], data_1D[1], label="COMSOL data")
plt.plot(p_x,a.rec_final[int(len(p_y)/2), :], label="MyCode")
plt.axhline(y=phi_bar, color='k', linestyle='--', label="$\overline{\phi}$ MyCode")
plt.legend()
plt.show()

# In[ ]:


# =============================================================================
# pos=coord_to_pos(FV.x, FV.y, pos_s[0])
# pos_x=int(pos%len(FV.x))
# 
# fig, axs = plt.subplots(2,2, figsize=(15,15))
# fig.tight_layout(pad=4.0)
# axs[1,0].plot(FV.y, a.rec_final[:,pos_x], label="coupling")
# axs[1,0].scatter(FV.y, FV_linear_plot[:,pos_x], label="Peaceman", c='r')
# axs[1,0].set_title("absolute error of the flux \n estimation for ratio={}".format(ratio))
# 
# axs[1,0].set_ylabel("absolute value [$kg m^{-1} s^{-1}$]")
# axs[1,0].set_xlabel("source ID")
# axs[1,0].legend()
# 
# d=axs[1,1].scatter(np.arange(len(p_q)),(1e3)*np.abs(p_q-phi_q)/np.abs(p_q))
# axs[1,1].set_title("relative error * $10^{3}$")
# axs[1,1].set_ylabel("relative err")
# axs[1,1].set_xlabel("source ID")
# 
# b=axs[0,1].imshow(FV_linear_plot, extent=[0,L,0,L],origin='lower')
# axs[0,1].set_xlabel("$\mu$m")
# axs[0,1].set_ylabel("$\mu$m")
# axs[0,1].set_title("validation reconstruction")
# divider = make_axes_locatable(axs[0,1])
# cax = divider.append_axes('right', size='10%', pad=0.05)
# fig.colorbar(b, cax=cax,orientation='vertical')
# 
# c=axs[0,0].imshow((a.rec_final-FV_linear_mat)*1e3, extent=[0,L,0,L], origin='lower')
# axs[0,0].set_xlabel("$\mu$m")
# axs[0,0].set_ylabel("$\mu$m")
# axs[0,0].set_title("absolute error of the reconstructed $\phi$ \n multiplied by $10^3$")
# divider = make_axes_locatable(axs[0,0])
# cax = divider.append_axes('right', size='10%', pad=0.05)
# fig.colorbar(c, cax=cax,orientation='vertical')
# plt.show()
# 
# =============================================================================

# In[ ]:

L_off=h_ss/2
subdiv=5

off_center=np.vstack((np.concatenate((np.zeros(subdiv),np.linspace(0,L_off,subdiv))),
                        np.concatenate((np.linspace(0,L_off,subdiv), np.linspace(0,L_off,subdiv))))).T

off_center=np.array([[ 0.,  0.],
       [ 0.,  6.],
       [ 0., 12.],
       [ 0., 18.],
       [ 0., 24.],
       [ 0.,  0.],
       [ 6.,  6.],
       [12., 12.],
       [18., 18.],
       [24., 24.]])*h_ss/48  #These are the values introduced in COMSOL


mat_errors_peac=np.zeros([subdiv,subdiv])
mat_errors_COMSOL=np.zeros([subdiv,subdiv])

#For alpha=100, the COMSOL results:
if alpha==100:
    q_COMSOL=np.array([0.6117,0.6119, 0.6123,0.6129, 0.6139, 0.6117,0.6123,0.6132,0.6147,0.6168])    
elif alpha==800:
    q_COMSOL=np.array([0.5088,0.5089,0.5091,0.5096,0.5103,0.5088,0.5089,0.5095,0.5104,0.5117])

q_array_off=np.array([])
q_array_peac=np.array([])


positions_matrix=np.vstack((np.concatenate((np.zeros(subdiv), np.arange(subdiv))),
                        np.concatenate((np.arange(subdiv), np.arange(subdiv))))).T.astype(int)
c=0
for i in off_center:
    pos_s=np.array([[0.5,0.5]])*L+np.array(i)
    S=len(pos_s)
    r=post.reconstruction_extended_space(pos_s, Rv, h_ss, L, K_eff, D, directness)
    r.solve_linear_prob(np.zeros(4), np.array([1]))
    phi_FV=r.phi_FV #values on the FV cells
    phi_q=r.phi_q #values of the flux
    #comparison with 
    FV=FV_validation(L, cells*ratio, pos_s, np.array([1]), D, K_eff, Rv)
    FV_linear=FV.solve_linear_system()
    p_q=FV.get_q(FV_linear)
    
    print("\n")
    print("ERROR FOR RELATIVE OFF-CENTERING. distance/h= {}".format(np.array(i)/h_ss))
    mat_errors_peac[positions_matrix[c,1], positions_matrix[c,0]]=get_MRE(p_q, phi_q)
    mat_errors_COMSOL[positions_matrix[c,1], positions_matrix[c,0]]=get_MRE(q_COMSOL[c], phi_q)
    q_array_off=np.append(q_array_off, phi_q)
    q_array_peac=np.append(q_array_off, p_q)
    c+=1
# =============================================================================
#         FV_linear_plot=FV_linear.copy() #to plot since it has \bar{\phi}
#         FV_linear_plot[FV.s_blocks]-=FV.get_q(FV_linear)*np.log(0.2*alpha/cells/ratio)/(2*np.pi*D)
#         FV_linear_plot=FV_linear_plot.reshape(cells*ratio, cells*ratio)
#         a=post.reconstruction_sans_flux(sol, t, L,ratio, directness)
#         p=a.reconstruction()   
#         a.reconstruction_boundaries(np.array([0,0,0,0]))
#         a.rec_corners()
#         get_plots_through_sources_peaceman(a.rec_final,FV_linear_plot,pos_s, p_x,p_y,t.y)
# =============================================================================

#%%

plt.imshow(mat_errors_peac, origin="lower", extent=[0,L_off, 0, L_off])
plt.colorbar()
plt.title("Peaceman")
plt.xlabel("x ($\mu m$)")
plt.ylabel("y ($\mu m$)")
plt.show()
plt.imshow(mat_errors_COMSOL, origin="lower", extent=[0,L_off, 0, L_off])
plt.colorbar()
plt.title("COMSOL")
plt.xlabel("x ($\mu m$)")
plt.ylabel("y ($\mu m$)")
plt.show()

#%%
#The worst position is the last calculated (the corner of a FV cell)
#Let's compare contours

file=directory + 'SingleSource_alpha' + str(alpha) + '_2D_corner.txt'
df=pandas.read_fwf(file)
ref_data=np.array(df).T #reference 2D data from COMSOL
phi_bar_COMSOL=np.max(ref_data[2])

r.set_up_manual_reconstruction_space(ref_data[0], ref_data[1])
r.reconstruction_manual()
r.reconstruction_boundaries(np.zeros(4))
phi_MyCode=r.u+r.DL+r.SL
u=r.u
DL=r.DL
SL=r.SL

#%% Comarison of the contour for the worst case scenario 
plt.tricontourf(ref_data[0], ref_data[1], ref_data[2]-phi_MyCode); plt.colorbar()
plt.title("Contour of the absolute error commited \n Comarison with COMSOL solution $10^4$ unknowns \n")


#%%
##############################################################################
#   PRE-RECORDED RESUTLS, for the old version of the flux estimation!
##############################################################################


if alpha==100 and directness==2 and cells==5:
#alpha=100, directness=2, cells, L_off=h_ss/2
    err=np.array([[0.00035475, 0.        , 0.        , 0.        , 0.        ],
           [0.00068144, 0.0013345 , 0.        , 0.        , 0.        ],
           [0.00133358, 0.        , 0.00280302, 0.        , 0.        ],
           [0.00230823, 0.        , 0.        , 0.00524834, 0.        ],
           [0.00392534, 0.        , 0.        , 0.        , 0.00866718]])

    
if alpha==100 and directness==1 and cells==5:
#alpha=100, directness=2, cells, L_off=h_ss/2
    err=np.array([[0.00213009, 0.        , 0.        , 0.        , 0.        ],
           [0.00245637, 0.0031074 , 0.        , 0.        , 0.        ],
           [0.00310998, 0.        , 0.00456293, 0.        , 0.        ],
           [0.00409438, 0.        , 0.        , 0.00696084, 0.        ],
           [0.00574039, 0.        , 0.        , 0.        , 0.01027151]])

        
if alpha==800 and directness==1 and cells==5:
#alpha=100, directness=2, cells, L_off=h_ss/2
    err=np.array([[0.00194115, 0.        , 0.        , 0.        , 0.        ],
       [0.00213738, 0.00213684, 0.        , 0.        , 0.        ],
       [0.002531  , 0.        , 0.0033056 , 0.        , 0.        ],
       [0.00351674, 0.        , 0.        , 0.00503712, 0.        ],
       [0.00490344, 0.        , 0.        , 0.        , 0.00750622]])

    
if alpha==800 and directness==2 and cells==5:
#alpha=100, directness=2, cells, L_off=h_ss/2
    err=np.array([[0.00046434, 0.        , 0.        , 0.        , 0.        ],
       [0.00066071, 0.0006609 , 0.        , 0.        , 0.        ],
       [0.00105273, 0.        , 0.00184005, 0.        , 0.        ],
       [0.00203036, 0.        , 0.        , 0.00361008, 0.        ],
       [0.00339262, 0.        , 0.        , 0.        , 0.00616817]])

plt.plot(err[:,0], label="no interpolation");
plt.plot(mat_errors_COMSOL[:,0], label="interpolation")
plt.title("vertical")
plt.xlabel("position")
plt.ylabel("relative flux error")
plt.legend()
plt.show()
plt.plot(err[np.arange(5), np.arange(5)], label="no interpolation");
plt.plot(mat_errors_COMSOL[np.arange(5), np.arange(5)], label="interpolation")
plt.title("diagonal")
plt.xlabel("position")
plt.ylabel("relative flux error")
plt.legend()
plt.show()
