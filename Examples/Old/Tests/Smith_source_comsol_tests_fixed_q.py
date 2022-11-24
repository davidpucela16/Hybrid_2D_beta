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
ratio=int(50/cells)*2+2
print("ratio: ", ratio)
#ratio=12
#Rv=np.exp(-2*np.pi)*h_ss




conver_residual=5e-5

stabilization=0.5

validation=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=10
print("directness=", directness)
#pos_s=np.array([[x_ss[2], y_ss[2]],[x_ss[4], y_ss[4]]])
#pos_s=np.array([[3.5,3.8],[3.4,3.4], [4.1, 3.6],[2,2]])-np.array([0.25,0.25])
#pos_s/=2
#pos_s=np.array([[1.25,1.25],[1.25,1.75], [1.75,1.75],[1.75,1.25]])
#pos_s=np.array([[4.3,4.3],[4.3,5.5], [3.5,4.5],[3.5,3.5]])


#pos_s=np.array([[0.41,0.41],[0.7,0.7],[0.3,0.47],[0.8,0.2]])*L
pos_s=np.array([[0.47,0.47],[0.53,0.53]])*L
pos_s1=np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.91,0.43]])
pos_s2=np.array([[0.27,0.6],[0.52,0.65],[0.59,0.66],[0.67,0.67],[0.13,0.75],[0.15,0.93],[0.2,0.87],[0.28,0.98],[0.8,0.85],[0.83,0.92]])
pos_s=(np.concatenate((pos_s1, pos_s2))*0.6+0.2)*L
#pos_s=(np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.96,0.43]])*0.6+0.2)*L
#pos_s=np.array([[0.5,0.5]])*L
pos_s=(np.random.random((6,2))*0.6+0.2)*L



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


#Linear system with coupling model
n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
n.solve_linear_prob(np.zeros(4), C_v_array)


#%% q fixed
b=1/3
q_array=np.ones(S)*b
phi_FV=np.linalg.solve(n.A_matrix, -n.b_matrix.dot(q_array))
real_C_v_array=n.c_matrix.dot(phi_FV)+ n.d_matrix.dot(q_array)
phi_bar=real_C_v_array-b/K0

phi_com=np.array([0.5533,0.7292,0.7854,0.8064,0.9753,0.5343,0.6891,0.7958,0.9284,1.0414,0.9418,0.7355,0.6808,0.7554,0.662,0.7114,0.6424])

# In[6]:


b=post.reconstruction_sans_flux(np.concatenate((phi_FV, q_array)), n, L,ratio, directness)
p=b.reconstruction()   
b.reconstruction_boundaries(np.array([0,0,0,0]))
b.rec_corners()


#%%µ

plt.imshow(b.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Steady State")
plt.xlabel('x')
plt.ylabel('y')



# In[11]:
n.Full_Newton(np.ndarray.flatten(n.phi_FV) , np.ndarray.flatten(n.phi_q), conver_residual, M, phi_0)
a=post.reconstruction_sans_flux(n.phi[-1], n, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()
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
sources=0
if sources:
    directory_files='../../Smith_sources/alpha50_sources'
else:
    directory_files='../../Smith_sources/alpha50_SourceSink'

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




plt.plot(np.arange(10),r*2, 'r', label="Current model d=9.8 $\mu m$")
plt.plot(np.arange(10),r*1, 'b', label="Current model d=0.98 $\mu m$")
plt.plot(np.arange(10),r*1.1, 'r--' ,label="FV no coupling d=9.8 $\mu m$")
plt.plot(np.arange(10),r*1.2, 'b--', label="FV no coupling d=0.98 $\mu m$")
plt.plot(np.arange(10),r*1.1, 'r:' ,label="Explicit model d=9.8 $\mu m$")
plt.plot(np.arange(10),r*1.2, 'b:', label="Explicit model d=0.98 $\mu m$")
plt.legend()
plt.show()





