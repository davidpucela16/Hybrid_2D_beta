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
          'figure.figsize': (12,12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.rcParams['font.size'] = '14'

#0-Set up the sources
#1-Set up the domain
alpha=50
validation=1
Da_t=10

D=1
K0=1


L=240

M=Da_t*D/L**2
phi_0=0.4
cells=5
h_ss=L/cells
#ratio=int(80/cells)*
ratio=20
print("ratio: ", ratio)
#ratio=12
#Rv=np.exp(-2*np.pi)*h_ss

print("h coarse:",h_ss)




conver_residual=5e-5

stabilization=0.5

validation=True
x_ss=np.linspace(h_ss/2, L-h_ss/2, int(np.around(L/h_ss)))
y_ss=x_ss
directness=1
print("directness=", directness)

pos_s=np.array([[0.5,0.5]])*L
S=len(pos_s)
Rv=np.zeros(S)+L/alpha
K_eff=K0/(np.pi*Rv**2)
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


p=np.linspace(0,1,100)
if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")


C_v_array=np.ones(S)   

#%% 1 

# =============================================================================
# FV=FV_validation(L, cells*ratio, pos_s, C_v_array, D, K_eff, Rv)
# FV_linear=FV.solve_linear_system()
# 
# FV_linear_plot=FV_linear.copy() #to plot since it has \bar{\phi}
# FV_linear_plot[FV.s_blocks]+=FV.get_q(FV_linear)*np.log(0.2*alpha/cells/ratio)/(2*np.pi*D)
# FV_linear_plot=FV_linear_plot.reshape(cells*ratio, cells*ratio)
# 
# FV_linear[FV.s_blocks]+=FV.get_corr_array(FV_linear)
# 
# FV_linear_mat=FV_linear.reshape(cells*ratio, cells*ratio)
# =============================================================================


#%% 2

n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
#pdb.set_trace()
n.solve_linear_prob(np.zeros(4), C_v_array)

#%%
plt.plot(n.phi_q, label="MyCode")
plt.legend()
plt.show()



#%% 3
b=post.reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(n.phi_FV), n.phi_q)), n, L,ratio, directness)
p=b.reconstruction()   
b.reconstruction_boundaries(np.array([0,0,0,0]))
b.rec_corners()
plt.imshow(b.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Steady State ")
plt.colorbar(); plt.show()



#%%


vmax=np.max(b.rec_final)
vmin=np.min(b.rec_potentials)
u,pot=b.get_u_pot(C_v_array)
c=0




plt.subplots(2,2, figsize=(12,12))

plt.subplot(2,2,1)
plt.imshow(pot, origin='lower', vmax=vmax, vmin=-0.1*vmax)
plt.title("SL + DL in $\overline{\Omega}$")
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(b.rec_potentials, origin="lower", vmax=0, vmin=vmin)
plt.title("SL in $\overline{\Omega}$ (old singular term)")
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(u, origin="lower", vmax=vmax, vmin=-0.1*vmax)
plt.title("u")
plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(b.rec_final, origin="lower", vmax=vmax, vmin=-0.1*vmax)
plt.title("$\phi$")
plt.colorbar()
plt.show()

i=pos_s[0]
pos=coord_to_pos(b.x, b.y, i)
plt.plot(b.x,b.rec_final[pos//len(b.x),:],label="SS no metab", linewidth=5)
plt.plot(b.x,u[pos//len(b.x),:], label="u")
plt.plot(b.x,pot[pos//len(b.x),:], label="SL + DL in $\overline{\Omega}$")

plt.xlabel("y ($\mu m$)")
plt.ylabel("$\phi$")
plt.legend()
plt.title("Linear solution")

c+=1

if validation:
    COMSOL_q_values=np.array([ 0.7885,0.656,0.6117,0.4997])
    
    
    COMSOL_alpha=np.array([10,50,100,1000])
    case_number=np.where(COMSOL_alpha==alpha)
    COMSOL_phi_bar=1-COMSOL_q_values[case_number]/K0
    plt.axhline(y=COMSOL_phi_bar, 
            color='k', linestyle='--', label="Reference value $\overline{\phi}$")
    plt.show()
    if K0!=1:
        print("\n\n\n ERROR!! \n\n\n")

    case="Single_source_alpha{}.txt".format(alpha)
    file=open("../../Single_source_COMSOL/" + case, "r")
    a=file.readlines()   
    
    
    position=np.array([], dtype=float)
    value=np.array([], dtype=float)
    for i in a:
        value=np.append(value, float(i.split(" ")[-1]))
        position=np.append(position, float(i.split(" ")[0]))
        
    plt.plot(position, value, label="comsol")
    plt.scatter(b.x,b.rec_final[:,int(cells*ratio/2)], color='r', s=10,label="SS no metab")
    plt.xlabel("y ($\mu m$)")
    plt.ylabel("$\phi$")
    plt.legend()
    plt.show()
    
    q=COMSOL_q_values[case_number]
    print("MRE q= ", (q-b.phi_q)/q)
    

plt.subplots(1,2, figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(b.x,b.rec_final[pos//len(b.x),:],label="$\phi$", linewidth=5)
plt.plot(b.x,u[pos//len(b.x),:], label="u")
plt.plot(b.x,pot[pos//len(b.x),:], label="SL + DL in $\overline{\Omega}$")
plt.axhline(y=COMSOL_phi_bar, color='k', linestyle='--', label="Reference value $\overline{\phi}$")
plt.xlabel("y ($\mu m$)")
plt.ylabel("$\phi$")
plt.legend(loc='center right')
plt.title("Linear solution")


plt.subplot(1,2,2)
plt.plot(position, value, label="comsol")
plt.scatter(b.x,b.rec_final[:,int(cells*ratio/2)], color='r', s=10,label="SS no metab")
plt.xlabel("y ($\mu m$)")
plt.ylabel("$\phi$")
plt.title("Reconstruction vs Reference")
plt.legend()


#%%
aa=int(ratio)
pot_fig=np.zeros(pot.shape)
pot_fig[:]=np.nan
pot_fig[aa:-aa,aa:-aa]=pot[aa:-aa,aa:-aa]

u_fig=np.zeros(u.shape)
u_fig[:]=np.nan
u_fig[aa:-aa,aa:-aa]=u[aa:-aa,aa:-aa]

i=pos_s[0]
pos=coord_to_pos(b.x, b.y, i)
fig, ax=plt.subplots(2,2, figsize=(12,12))
fig.tight_layout(pad=2.5)
im1=ax[0,0].imshow(pot_fig, origin='lower', vmax=vmax, vmin=-0.1*vmax)
ax[0,0].set_title("$\phi_\\beta$ in $\widehat{V}_k$")
#plt.colorbar()

im2=ax[0,1].imshow(u, origin="lower", vmax=vmax, vmin=-0.1*vmax)
ax[0,1].set_title("u in $\widehat{V}_k$ and $\phi$ outside")
#plt.colorbar()

im3=ax[1,0].imshow(b.rec_final, origin="lower", vmax=vmax, vmin=-0.1*vmax)
ax[1,0].set_title("$\phi = \phi_\\beta + \phi_\Omega$")
#plt.colorbar()


ax[1,1].plot(b.x,b.rec_final[pos//len(b.x),:],label="$\phi$", linewidth=5)
ax[1,1].plot(b.x,u_fig[pos//len(b.x),:], label="u in $\widehat{V}_k$", linewidth=3)
ax[1,1].plot(b.x,pot_fig[pos//len(b.x),:], label="$\phi_\\beta$ in $\widehat{V}_k$", linewidth=5)

ax[1,1].set_xlabel("y ($\mu m$)")
ax[1,1].set_ylabel("$\phi$")
ax[1,1].legend(loc=7)
ax[1,1].set_title("Plots through the mid-section")

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im3, cax=cbar_ax)
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pdfs/One_source_example.png', transparent=True)

#%%


#%% Initial try to do a 3D surface plot!!

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as ticker
def surf_plot(x, y, surface, m,M,L, title):
    

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = x
    Y = y
    X, Y = np.meshgrid(X, Y)
    Z = surface
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                           linewidth=0, antialiased=False, vmin=m, vmax=M)
    
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(ticker.FixedLocator(np.linspace(m,M,4)))
    ax.set_zlim(m, M)
    ax.set_xlim(0,L)
    ax.set_ylim(0,L)
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set_xlabel("x [$\mu m$]", labelpad=15)
    ax.set_ylabel("y [$\mu m$]", labelpad=15)
    ax.set_title(title)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.35, aspect=5)
    
m=np.min(np.concatenate((pot, u)))
M=np.max(np.concatenate((pot, u)))
surf_plot(b.x[aa:-aa], b.y[aa:-aa], pot_fig[aa:-aa,aa:-aa], m,M,L, '$\phi_\\beta$ in $\widehat{V}$')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/One_source_example/phi_beta.png', transparent=True, bbox_inches='tight')

surf_plot(b.x[aa:-aa], b.y[aa:-aa], u_fig[aa:-aa,aa:-aa],m,M,L, '$\phi_\Omega$ in $\widehat{V}$')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/One_source_example/phi_Omega.png', transparent=True, bbox_inches='tight')

surf_plot(b.x, b.y, b.rec_final, m, M,L,'$\phi$ in $\Omega_\sigma$')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/One_source_example/phi.png', transparent=True, bbox_inches='tight')

#%%
plt.plot(b.x,b.rec_final[pos//len(b.x),:],label="$\phi$", linewidth=5)
plt.plot(b.x,u_fig[pos//len(b.x),:], label="u in $\widehat{V}_k$", linewidth=3)
plt.plot(b.x,pot_fig[pos//len(b.x),:], label="$\phi_\\beta$ in $\widehat{V}_k$", linewidth=5)

plt.xlabel("y ($\mu m$)")
plt.ylabel("$\phi$")
plt.legend(loc=7)
plt.title("Plots through the mid-section")
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/One_source_example/OneD.png', transparent=True)