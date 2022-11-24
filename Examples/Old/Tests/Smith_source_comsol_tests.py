#!/usr/bin/env python
# coding: utf-8

"""
The goal here is to perform a full analysis of the simplified 2D network from A F Smith.
At the end, it will provide an error analysis that can be put on the Glasgow poster



"""

# In[1]:
import sys
sys.path.append('..')
import pandas
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
          'figure.figsize': (15,15),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large', 
         'lines.linewidth': 5}

plt.rcParams['font.size'] = '20'
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
cells=60
h_ss=L/cells
ratio=int(50/cells)*2+2
print("ratio: ", ratio)
#ratio=12
#Rv=np.exp(-2*np.pi)*h_ss



#For the non-linear iterative problem
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
pos_s1=np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.91,0.43]])
pos_s2=np.array([[0.27,0.6],[0.52,0.65],[0.59,0.66],[0.67,0.67],[0.13,0.75],[0.15,0.93],[0.2,0.87],[0.28,0.98],[0.8,0.85],[0.83,0.92]])
pos_s=(np.concatenate((pos_s1, pos_s2))*0.8+0.1)*L
#pos_s=(np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.96,0.43]])*0.6+0.2)*L
#pos_s=np.array([[0.5,0.5]])*L
#pos_s=(np.random.random((6,2))*0.6+0.2)*L


err={}
size={}
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

sources=0
C_v_array=np.ones(S)   
if not sources:
    C_v_array[[2,6,8,13,15]]=0


# In[11]:
n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
n.solve_linear_prob(np.zeros(4), C_v_array)

n.Full_Newton(np.ndarray.flatten(n.phi_FV) , np.ndarray.flatten(n.phi_q), conver_residual, M, phi_0)
a=post.reconstruction_sans_flux(n.phi[-1], n, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries(np.array([0,0,0,0]))
a.rec_corners()
n.assemble_it_matrices_Sampson(n.u, n.q)


#%%

err={}
size={}

#%%
# =============================================================================
# fileD='/home/pdavid/Bureau/SS_malpighi/2D_cartesian/Validated_2D_Code/Smith_sources/temporary.txt'
# df=pandas.read_fwf(fileD, skiprows=5)
# 
# 
# 
# com_q=df.columns.astype(float) #reference 2D data from COMSOL
# plt.plot(com_q, label="comsol")
# plt.plot(n.phi_q)
# plt.legend()
# =============================================================================
# In[ ]: COMSOL Tests. this is where the test begin!!!!
#Currently, the full tests are only available for SourceSink

for alpha in [500,50]:
    if sources:
        #directory_files='../Smith_sources/alpha{}_sources'.format(alpha)
        directory_files='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Validated_2D_Code/Smith_sources/alpha{}_sources'.format(alpha)
    else:
        #directory_files='../Smith_sources/alpha{}_SourceSink'.format(alpha)
        directory_files='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Validated_2D_Code/Smith_sources/alpha{}_SourceSink'.format(alpha)
    
    
    if alpha==50:
        loop=np.array([7,6,5,4,3,2,1])
    if alpha==500:
        loop=np.array([5,4,3,2,1])
        
    #for i in loop:
    for i in loop:
        print(alpha, i)
        mesh=i
        file=directory_files + '/Contour_mesh{}.txt'.format(int(mesh))
        df=pandas.read_fwf(file, skiprows=8)
        ref_data=np.squeeze(np.array(df).T)#reference 2D data from COMSOL
        
        r=post.reconstruction_extended_space(pos_s, Rv, h_ss,L, K_eff, D,directness)
        r.solve_linear_prob(np.zeros(4),C_v_array)
        r.set_up_manual_reconstruction_space(ref_data[0], ref_data[1])
        size['alpha{}_mesh{}'.format(alpha,i)]=ref_data[0].size
        r.reconstruction_manual()
        r.reconstruction_boundaries(np.zeros(4))
        phi_MyCode=r.u+r.DL+r.SL
        
        plt.tricontourf(ref_data[0], ref_data[1], ref_data[2], levels=np.linspace(0,0.6,100))
        plt.colorbar()
        plt.title('COMSOL RESULTS MESH-SIZE={}'.format(int(ref_data.size/3)))
        plt.show()
        
        plt.tricontourf(ref_data[0], ref_data[1], phi_MyCode, levels=np.linspace(0,np.max(ref_data[2]),100))
        plt.colorbar()
        plt.title('MyCode Result'.format(int(ref_data.size/3)))
        plt.show()
        
        
        plt.tricontourf(ref_data[0], ref_data[1], phi_MyCode-ref_data[2], levels=np.linspace(-0.1,0.1,100))
        plt.colorbar()
        plt.title('absolute error')
        plt.show()
        fileD=directory_files + '/mesh{}.txt'.format(int(mesh))
        df=pandas.read_fwf(fileD, skiprows=5)
        
        com_q=df.columns.astype(float) #reference 2D data from COMSOL
        
        
        
        if i==np.max(loop):
            print("reference!",i)
            ref=com_q
        else:
        
            err['alpha{}_mesh{}'.format(alpha,i)]=get_MRE(ref, com_q)
        plt.plot(com_q, label='Reference')
        plt.plot(r.phi_q, label='MyCode')
        plt.scatter(np.arange(S), ref, label='current Com')
        plt.legend()
        plt.show()
    cells=6
    ratios=np.array([5,10,30,40])
    for ratio in ratios:
        print("cells FV simulation", cells*ratio)
        comp=FV_validation(L, cells*ratio, pos_s, C_v_array, D, K_eff, Rv)
        comp.set_up_system()
        comp_phi=comp.solve_linear_system()
        comp_q=comp.get_q(comp_phi)
        err['alpha{}_FVcomp_cells{}'.format(alpha, cells*ratio)]=get_MRE(ref, comp_q)

    cells_arr=np.array([3,4,5,6,12,24, 48])
    for i in cells_arr:
        cells=i
        h_ss=L/cells
        Rv=L/alpha+np.zeros(S)

        K_eff=C0/(np.pi*Rv**2)
        n=non_linear_metab(pos_s, Rv, h_ss, L, K_eff, D, directness)
        n.solve_linear_prob(np.zeros(4), C_v_array)
        err['alpha{}_MyCode_cells{}'.format(alpha, cells)]=get_MRE(ref, n.phi_q)
        


#%%
cells=6
er_comsol_50=np.array([])
size_comsol_50=np.array([])
for i in np.arange(6)+1:
    er_comsol_50=np.append(er_comsol_50, err['alpha50_mesh{}'.format(i)])
    size_comsol_50=np.append(size_comsol_50, size['alpha50_mesh{}'.format(i)])

er_comsol_500=np.array([])
size_comsol_500=np.array([])
for i in np.array([3,2,1,0])+1:
    er_comsol_500=np.append(er_comsol_500, err['alpha500_mesh{}'.format(i)])
    size_comsol_500=np.append(size_comsol_500, size['alpha500_mesh{}'.format(i)])

er_FV_50=np.zeros(len(ratios))
er_FV_500=np.zeros(len(ratios))
for i in range(len(ratios)):
    er_FV_50[i]=err['alpha50_FVcomp_cells{}'.format(cells*ratios[i])]
    er_FV_500[i]=err['alpha500_FVcomp_cells{}'.format(cells*ratios[i])]

er_MyCode_50=np.zeros(len(cells_arr))
er_MyCode_500=np.zeros(len(cells_arr))
for i in range(len(cells_arr)):
     er_MyCode_50[i]=err['alpha50_MyCode_cells{}'.format(cells_arr[i])]
     er_MyCode_500[i]=err['alpha500_MyCode_cells{}'.format(cells_arr[i])]   
     
#%%

plt.plot(cells_arr, er_MyCode_50, color='r' ,label='Multiscale R/L=50')
plt.plot(cells_arr, er_MyCode_500, color='b',label='Multiscale R/L=500')

plt.plot(cells*ratios,er_FV_50, '--',color='r',  label='FV no coupling R/L=50')
plt.plot(cells*ratios,er_FV_500, '--', color='b' ,label='FV no coupling R/L=500')

plt.plot(size_comsol_50[:-2],er_comsol_50[:-2], ':',color='r' , label='Reference R/L=50')
plt.plot(size_comsol_500[:],er_comsol_500[:], ':', color='b' ,label='Reference R/L=500')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title("Mean relative error for $q$", fontsize=50)
plt.ylabel('MRE', fontsize=50)
plt.xlabel("# of unknowns of system", fontsize=50)
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/Fig_Glasgow/q_error.png', transparent=True, bbox_inches='tight')

#%%
import json
with open('/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Validated_2D_Code/Smith_sources/convert.txt', 'w') as convert_file:
     convert_file.write(json.dumps(err))
    

# In[6]:


b=post.reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(n.phi_FV), n.phi_q)), n, L,ratio, directness)
p=b.reconstruction()   
b.reconstruction_boundaries(np.array([0,0,0,0]))
b.rec_corners()


#%%

plt.imshow(b.rec_final, origin='lower')
plt.title("bilinear reconstruction \n coupling model Steady State")
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()





#%% - Plots for the Glasgow Poster Figure
mi,ma=0,0.5
X,Y=np.meshgrid(b.x, b.y)
plt.imshow(b.rec_final, extent=[0,L,0,L], origin='lower', vmin=mi, vmax=ma)
plt.xlabel("x [$\mu m$]")
plt.ylabel("x [$\mu m$]")
plt.title("$\phi$ - reconstructed", fontsize=50)
plt.colorbar(format='%.2f')
#plt.savefig('/home/pdavid/Bureau/Figssssss/phi.png', transparent=True, bbox_inches='tight')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/Fig_Glasgow/phi.png', transparent=True, bbox_inches='tight')
#%%
#%%

NN=post.coarse_NN_rec(x_ss, y_ss, b.phi_FV.reshape(len(x_ss), len(x_ss)), n.pos_s, n.s_blocks, b.phi_q, ratio, h_ss, directness, Rv)

plt.imshow(NN,origin='lower' ,extent=[0,L,0,L], vmin=mi, vmax=ma)
plt.xlabel("x [$\mu m$]")
plt.ylabel("y [$\mu m$]")
plt.title("Non linear problem \n Concentration field $\phi$", fontsize=50)
plt.colorbar(format='%.2f')
#plt.savefig('/home/pdavid/Bureau/Figssssss/NN.png', transparent=True, bbox_inches='tight')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/Fig_Glasgow/NN.png', transparent=True, bbox_inches='tight')

#%%
plt.imshow((1-phi_0*(a.rec_final+phi_0)**-1)*M*h_ss**2, origin='lower',extent=[0,L,0,L])
plt.title("bilinear reconstruction \n coupling model Metabolism")
plt.ylabel("y [$\mu m$]")
plt.xlabel("x [$\mu m$]")
plt.title("Non - linear problem \n Metabolism term", fontsize=50)
plt.colorbar(format='%.2f')
#plt.savefig('/home/pdavid/Bureau/Figssssss/Metabolism.png', transparent=True, bbox_inches='tight')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/Fig_Glasgow/Metabolism.png', transparent=True, bbox_inches='tight')

#%%
plt.imshow(a.rec_final, origin='lower', extent=[0,L,0,L],vmin=mi, vmax=ma)
plt.title("Non - linear problem \n Reconstructed field $\phi$", fontsize=50)
plt.ylabel("y [$\mu m$]")
plt.xlabel("x [$\mu m$]")
plt.colorbar(format='%.2f')
#plt.savefig('/home/pdavid/Bureau/Figssssss/Full_problem.png', transparent=True, bbox_inches='tight')
plt.savefig('/home/pdavid/Bureau/PhD/Presentations/Glasgow/Figures_poster/pngs/Fig_Glasgow/Full_problem.png', transparent=True, bbox_inches='tight')


#%% FIXED Q
b=1/3
q_array=np.ones(S)*b
phi_FV=np.linalg.solve(n.A_matrix, -n.b_matrix.dot(q_array))
real_C_v_array=n.c_matrix.dot(phi_FV)+ n.d_matrix.dot(q_array)
phi_bar=real_C_v_array-b/K0
