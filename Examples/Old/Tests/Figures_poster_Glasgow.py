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


import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab

import pandas 
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15,15),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.rcParams['font.size'] = '30'

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


#%% 2

n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
#pdb.set_trace()
n.solve_linear_prob(np.zeros(4), C_v_array)
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
#plt.savefig('/home/pdavid/Bureau/Figssssss/One_source_example.png')
#plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pdfs/One_source_example.png', transparent=True)


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
    ax.set_xlabel("x [$\mu m$]", labelpad=50)
    ax.set_ylabel("y [$\mu m$]", labelpad=50)
    ax.set_title(title, fontsize=50)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.35, aspect=5)
    
m=np.min(np.concatenate((pot, u)))
M=np.max(np.concatenate((pot, u)))
surf_plot(b.x[aa:-aa], b.y[aa:-aa], pot_fig[aa:-aa,aa:-aa], m,M,L, '$\phi_\\beta$ in $\widehat{V}$')
#plt.savefig('/home/pdavid/Bureau/Figssssss/phi_beta.png', transparent=True, bbox_inches='tight')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/One_source_example/phi_beta.png', transparent=True, bbox_inches='tight')

#%%

surf_plot(b.x[aa:-aa], b.y[aa:-aa], u_fig[aa:-aa,aa:-aa],m,M,L, '$\phi_\Omega$ in $\widehat{V}$')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/One_source_example/phi_Omega.png', transparent=True, bbox_inches='tight')
#plt.savefig('/home/pdavid/Bureau/Figssssss/phi_Omega.png', transparent=True, bbox_inches='tight')

#%%
rr=b.rec_final.copy()
rr[aa,aa:-aa]=0
rr[aa:-aa,aa]=0
rr[-aa,aa:-aa]=0
rr[aa:-aa,-aa]=0
surf_plot(b.x, b.y, rr, m, M,L,'$\phi$ in $\Omega_\sigma$')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/One_source_example/phi.png', transparent=True, bbox_inches='tight')
#plt.savefig('/home/pdavid/Bureau/Figssssss/phi.png', transparent=True, bbox_inches='tight')

#%%
plt.plot(b.x,b.rec_final[pos//len(b.x),:],label="$\phi$", linewidth=5)
plt.plot(b.x,u_fig[pos//len(b.x),:], label="u in $\widehat{V}_k$", linewidth=3)
plt.plot(b.x,pot_fig[pos//len(b.x),:], label="$\phi_\\beta$ in $\widehat{V}_k$", linewidth=5)

plt.xlabel("y ($\mu m$)")
plt.ylabel("$\phi$")
plt.legend(loc=7)
plt.title("Plots through the mid-section")
#plt.savefig('/home/pdavid/Bureau/Figssssss/OneD.png', transparent=True)
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/One_source_example/OneD.png', transparent=True)



#%% NOW 2 SOURCES!!!!



#0-Set up the sources
#1-Set up the domain
D=1
L=10
cells=7
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=20
#Rv=np.exp(-2*np.pi)*h_ss

alpha=50
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
directness=5
print("directness=", directness)

d=np.array([2,4,6,10,14,18,22])  #array of the separations!!
dist=d*L/alpha
q_array_source=np.zeros((0,2))

p1=np.array([L*0.5,L*0.5])
both_sources=True

#%%
directory_files='../../Double_source_COMSOL/Double_source_diff/alpha' + str(alpha)
if sources:
     directory_files+="/sources"
else:
    directory_files+="/SourceSink"

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
        q_MyCode=np.vstack((q_MyCode, phi_q))
        
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


#%% Surf plot for the double source
i=dist[3]
pos_s=np.array([[0.5*L-i/2, 0.5*L],[0.5*L+i/2, 0.5*L]])
n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
#pdb.set_trace()
n.solve_linear_prob(np.zeros(4), C_v_array)
c=post.reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(n.phi_FV), n.phi_q)), n, L,ratio, directness)
c.reconstruction()   
c.reconstruction_boundaries(np.array([0,0,0,0]))
c.rec_corners()
plt.imshow(b.rec_final, origin='lower')
#surf_plot(c.x, c.y, c.rec_final, 0,0.5,L, "Double source")

plt.imshow(c.rec_final, origin='lower',extent=[0,240,0,240])
plt.title("Double source")
plt.colorbar()
plt.xlabel('$\mu$m')
plt.ylabel('$\mu$m')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Milan_CMBE_2022/Script_figures/double_source.png', transparent=True, bbox_inches='tight')
#plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/Double_source/contour_p.png', transparent=True, bbox_inches='tight')
#plt.savefig('/home/pdavid/Bureau/Figssssss/contour_p.png', transparent=True, bbox_inches='tight')
#%%
plt.plot(d, L2_array*100, linewidth=8)
plt.xlabel('$a/R_v$')
plt.ylabel("$\epsilon$ (%)")
plt.title('L2 error in field $\phi$', fontsize=50)
#plt.savefig('/home/pdavid/Bureau/Figssssss/phi_error.png', transparent=True, bbox_inches='tight')
#plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/Double_source/phi_error.png', transparent=True, bbox_inches='tight')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Milan_CMBE_2022/Script_figures/L2.png', transparent=True, bbox_inches='tight')
#%%
plt.plot(d, 100*np.abs((q_MyCode[:,0]-q_COMSOL[:,0])/q_COMSOL[:,0]), label='source 1, R=9.8 $\mu m$', linewidth=8)
plt.plot(d, 100*np.abs((q_MyCode[:,1]-q_COMSOL[:,1])/q_COMSOL[:,1]), label='source 2, R=4.9 $\mu m$ ', linewidth=8)
plt.legend()
plt.xlabel('$a/R_v$')
plt.ylabel("$\epsilon$ (%)")
plt.title('Relative error on $q$', fontsize=50)
#plt.savefig('/home/pdavid/Bureau/Figssssss/q_error.png', transparent=True, bbox_inches='tight')
#plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/Double_source/q_error.png', transparent=True, bbox_inches='tight')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Milan_CMBE_2022/Script_figures/q.png', transparent=True, bbox_inches='tight')


