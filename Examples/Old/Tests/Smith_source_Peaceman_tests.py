#!/usr/bin/env python
# coding: utf-8

# In[1]:
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8,8 ),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'lines.linewidth': 3}
pylab.rcParams.update(params)


#0-Set up the sources
#1-Set up the domain
alpha=50

Da_t=10

D=1
C0=1


L=240

M=Da_t*D/L**2
phi_0=0.4
cells=6
h_ss=L/cells
ratio=int(50/cells)*2
print("ratio: ", ratio)
#ratio=12
#Rv=np.exp(-2*np.pi)*h_ss


print("R: ", 1/(1/C0 + np.log(alpha/(5*cells*ratio))/(2*np.pi*D)))


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
pos_s=np.array([[0.47,0.47],[0.53,0.53]])*L
pos_s1=np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.96,0.43]])
pos_s2=np.array([[0.27,0.6],[0.52,0.65],[0.59,0.66],[0.67,0.67],[0.13,0.75],[0.15,0.93],[0.2,0.87],[0.28,0.98],[0.8,0.85],[0.83,0.92]])
pos_s=(np.concatenate((pos_s1, pos_s2))*0.8+0.1)*L
#pos_s=(np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.96,0.43]])*0.6+0.2)*L
#pos_s=np.array([[0.5,0.5]])*L
#pos_s=(np.random.random((6,2))*0.6+0.2)*L



pos_s=np.array([pos_s[4]])
S=len(pos_s)
Rv=L/alpha+np.zeros(S)

print("alpha: {} must be greater than {}".format(alpha, 5*ratio*cells))
print("h coarse:",h_ss)
K_eff=C0/(np.pi*Rv**2)
#Position image

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

p=np.linspace(0,1,100)
if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")


C_v_array=np.ones(S)   


# In[2]:


FV=FV_validation(L, cells*ratio, pos_s, C_v_array, D, K_eff, Rv)
FV_linear=FV.solve_linear_system()
FV_linear_rcr=FV_linear.copy()
FV_linear_rcr[FV.s_blocks]+=FV.get_corr_array(FV_linear)

FV_linear_mat=FV_linear.reshape(cells*ratio, cells*ratio)
FV_linear_rcr_mat=FV_linear_rcr.reshape(cells*ratio, cells*ratio)

#%%
print("R: ", 1/(1/C0 + np.log(0.2*FV.h/Rv)/(2*np.pi*D)))


# In[3]:


FV.get_corr_array(FV_linear)


#%% Plots FV reference solution - Peaceman Coupling

plt.imshow(FV_linear_rcr_mat, origin='lower')
plt.colorbar()
plt.title("FV reference solution, linear system\n mesh:{}x{}".format(ratio*cells, ratio*cells))
plt.show()


# In[5]:


#Linear system with coupling model
n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
n.solve_linear_prob(np.zeros(4), C_v_array)


# In[6]:


b=post.reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(n.phi_FV), n.phi_q)), n, L,ratio, directness)
p=b.reconstruction()   
b.reconstruction_boundaries(np.array([0,0,0,0]))
b.rec_corners()

#%%

c=post.reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(n.phi_FV), n.phi_q)), n, L,ratio, directness)
p=c.reconstruction()   
c.reconstruction_boundaries(np.array([0,0,0,0]))
c.rec_corners()

#%%
fig, axs = plt.subplots(1,2, figsize=(15,15))
fig.tight_layout(pad=4.0)
im=axs[0].imshow(b.rec_final, origin='lower')
axs[0].set_title("bilinear reconstruction \n coupling model Steady State ")
axs[0].set_ylabel("absolute value [$kg m^{-1} s^{-1}$]")
axs[0].set_xlabel("source ID")


axs[1].imshow(FV_linear_rcr_mat, origin='lower',vmax=np.max(FV_linear_mat*1.1))
axs[1].set_title("FV linear reference")
axs[1].set_ylabel("absolute value [$kg m^{-1} s^{-1}$]")
axs[1].set_xlabel("source ID")
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.savefig("filename.pdf", bbox_inches = 'tight',
    pad_inches = 0)


# In[7]:


#manual q
q_array_linear=-np.dot(FV.A_virgin.toarray()[FV.s_blocks,:],FV_linear)*FV.h**2/D
for i in pos_s:
    pos=coord_to_pos(FV.x, FV.y, i)
    
    plt.scatter(FV.x,FV_linear_rcr_mat[pos//len(FV.x),:], label="FV", color='b')
    plt.plot(FV.x,b.rec_final[pos//len(FV.x),:],label="SS no metab", color='r')
    plt.legend()
    plt.title("Linear solution")
    plt.show()


# In[8]:


#%% 5
FV_non_linear=FV.solve_non_linear_system(phi_0,M, stabilization)
#phi_FV=FV_linear.reshape(cells*ratio, cells*ratio)
phi_FV=(FV.phi[-1]+FV.Corr_array).reshape(cells*ratio, cells*ratio)


#%%

plt.imshow(FV_linear_mat, origin='lower',vmax=np.max(FV_linear_mat*1.1))
plt.title("FV linear reference")
plt.colorbar(); plt.show()


#%%
plt.imshow(phi_FV, origin='lower', vmax=np.max(FV_linear_mat*1.1))
plt.title("FV metab reference")
plt.colorbar(); plt.show()


#manual q
q_array=-np.dot(FV.A_virgin.toarray()[FV.s_blocks,:],FV.phi[-1])*FV.h**2/D+M*(1-phi_0/(FV.phi[-1, FV.s_blocks]+FV.Corr_array[FV.s_blocks]+phi_0))


# In[9]:


#manual q
q_array=-np.dot(FV.A_virgin.toarray()[FV.s_blocks,:],FV.phi[-1])*FV.h**2/D+M*(1-phi_0/(FV.phi[-1, FV.s_blocks]+FV.Corr_array[FV.s_blocks]+phi_0))


# In[10]:


print("MRE steady state system", get_MRE(n.phi_q, FV.get_q(FV_linear)))

plt.plot(np.arange(S),n.phi_q, label="MyCode", marker='o')
plt.plot(np.arange(S),FV.get_q(FV_linear), label="FV Peaceman reference", marker='o')
plt.legend()
plt.show()


# In[11]:


n.Full_Newton(np.ndarray.flatten(n.phi_FV) , np.ndarray.flatten(n.phi_q), conver_residual, M, phi_0)
a=post.reconstruction_sans_flux(n.phi[-1], n, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()


fig, axs = plt.subplots(1,2, figsize=(15,15))
fig.tight_layout(pad=4.0)
im=axs[0].imshow(a.rec_final, origin='lower', vmax=np.max(phi_FV*1.1))
axs[0].set_title("bilinear reconstruction \n coupling model Metabolism")
axs[0].set_ylabel("absolute value [$kg m^{-1} s^{-1}$]")
axs[0].set_xlabel("source ID")

axs[1].imshow(phi_FV, origin='lower', vmax=np.max(phi_FV*1.1))
axs[1].set_title("FV metab reference")
axs[1].set_ylabel("absolute value [$kg m^{-1} s^{-1}$]")
axs[1].set_xlabel("source ID")
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im, cax=cbar_ax)

n.assemble_it_matrices_Sampson(n.u, n.q)

#%% Initial try to do a 3D surface plot!!

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = b.x
Y = b.y
X, Y = np.meshgrid(X, Y)
R = b.rec_final
Z=R
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# In[12]:


plt.rcParams['font.size'] = '20'
plt.imshow(a.rec_final/(a.rec_final+phi_0)*(ratio*cells/L)**2, origin='lower', extent=[0,L,0,L])
#plt.title("metabolism")
plt.ylabel("y [$\mu m$]")
plt.xlabel("x [$\mu m$]")
plt.colorbar()



# In[14]:



plt.subplots(1,2, figsize=(15,10))
plt.subplot(1,2,1)
plt.plot(n.phi[-1,-S:], label="MyCode", marker='o')
plt.plot(FV.get_q(FV.phi[-1]), label="FV reference", marker='o')
plt.title("flux q estimation metabolism")
plt.legend()
plt.subplot(1,2,2)
plt.plot(n.phi_q, label="MyCode", marker='o')
plt.plot(FV.get_q(FV_linear), label="FV reference", marker='o')
plt.legend()
plt.title("Flux q estimation linear problem")
plt.show()

print("\nMRE q estimation non_linear", get_MRE(n.phi[-1,-S:], FV.get_q(FV.phi[-1])))
print("\nMRE q estimation linear", get_MRE(n.phi_q, FV.get_q(FV_linear)))




# In[15]:


for i in pos_s:
    pos=coord_to_pos(FV.x, FV.y, i)
    
    plt.scatter(FV.x,FV_linear_rcr_mat[pos//len(FV.x),:], label="FV no metab", marker='o', color='b')
    plt.scatter(FV.x,phi_FV[pos//len(FV.x),:], label="FV", marker='o',color='g')
    plt.plot(FV.x,a.rec_final[pos//len(FV.x),:],label="SS", color='r')
    plt.plot(FV.x,b.rec_final[pos//len(FV.x),:],label="SS no metab", color='y')
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


# In[16]:


print("relative errors in the q estimation metabolism")
print(np.abs(n.phi[-1,-S:]-FV.get_q(FV.phi[-1]))/FV.get_q(FV.phi[-1]))

print("\nabsolute error in the q estimation metabolism")
print(np.abs(n.phi[-1,-S:]-FV.get_q(FV.phi[-1])))

print("\nL2_error in the q estimation metabolism")
print(get_L2(n.phi[-1,-S:], FV.get_q(FV.phi[-1])))

print("\nMRE q estimation non_linear", get_MRE(n.phi[-1,-S:], FV.get_q(FV.phi[-1])))
print("\nMRE q estimation linear", get_MRE(n.phi_q, FV.get_q(FV_linear)))





#%% - Plots for the Glasgow Poster Figure
plt.figure(figsize=(9,8))
X,Y=np.meshgrid(b.x, b.y)
plt.contourf(X,Y, b.rec_final, levels=np.linspace(0,np.max(b.rec_final), 100))
plt.xlabel("x [$\mu m$]")
plt.ylabel("x [$\mu m$]")
plt.title("$\phi$ - linear problem reconstruction")
plt.colorbar(format='%.2f')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/Fig_Glasgow/phi.png', transparent=True, bbox_inches='tight')


NN=post.coarse_NN_rec(x_ss, y_ss, b.phi_FV.reshape(len(x_ss), len(x_ss)), n.pos_s, n.s_blocks, b.phi_q, ratio, h_ss, directness, Rv)

plt.imshow(NN, extent=[0,L,0,L])
plt.xlabel("x [$\mu m$]")
plt.ylabel("y [$\mu m$]")
plt.title("$\phi$ - linear problem")
plt.colorbar(format='%.2f')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/Fig_Glasgow/NN.png', transparent=True, bbox_inches='tight')


plt.imshow((1-phi_0*(a.rec_final+phi_0)**-1)*M*h_ss**2, origin='lower')
plt.title("bilinear reconstruction \n coupling model Metabolism")
plt.ylabel("y [$\mu m$]")
plt.xlabel("x [$\mu m$]")
plt.title("Non - linear metabolism")
plt.colorbar(format='%.2f')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/Fig_Glasgow/Metabolism.png', transparent=True, bbox_inches='tight')

plt.imshow(c.rec_final-a.rec_final, origin='lower')
plt.ylabel("y [$\mu m$]")
plt.xlabel("x [$\mu m$]")
plt.colorbar(format='%.2f')

plt.imshow(a.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Metabolism")
plt.ylabel("y [$\mu m$]")
plt.xlabel("x [$\mu m$]")
plt.colorbar(format='%.2f')


# In[ ]: COMSOL Tests
import pandas
sources=1
if sources:
    directory_files='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Validated_2D_Code/Smith_sources/alpha50_sources'
else:
    directory_files='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Validated_2D_Code/Smith_sources/alpha50_SourceSink'

mesh=7
file=directory_files + '/Contour_mesh{}.txt'.format(int(mesh))
df=pandas.read_fwf(file, skiprows=8)
ref_data=np.squeeze(np.array(df).T) #reference 2D data from COMSOL

r=post.reconstruction_extended_space(pos_s, Rv, h_ss,L, K_eff, D,directness)
r.solve_linear_prob(np.zeros(4),C_v_array)
r.set_up_manual_reconstruction_space(ref_data[0], ref_data[1])
r.reconstruction_manual()
r.reconstruction_boundaries(np.zeros(4))
phi_MyCode=r.u+r.DL+r.SL

plt.tricontourf(ref_data[0], ref_data[1], ref_data[2], levels=np.linspace(0,1,100))
plt.colorbar()
plt.title('COMSOL RESULTS MESH-SIZE={}'.format(int(ref_data.size/3)))
plt.show()
plt.tricontourf(ref_data[0], ref_data[1], phi_MyCode-ref_data[2], levels=np.linspace(-0.1,0.1,100))
plt.colorbar()
plt.title('absolute error')
plt.show()
fileD=directory_files + '/mesh_{}.txt'.format(int(mesh))
df=pandas.read_fwf(fileD, skiprows=5)

com_q=df.columns.astype(float) #reference 2D data from COMSOL

plt.plot(com_q, label='Reference')
plt.plot(r.phi_q, label='MyCode')
plt.legend()
plt.show()
