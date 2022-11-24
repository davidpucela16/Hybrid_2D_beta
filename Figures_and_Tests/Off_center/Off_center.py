# -*- coding: utf-8 -*-

#Define Computer
import os
Malphigui=0
if Malphigui:
    directory='/home/pdavid/Bureau/Updated_BCs_2/Code' #Malpighi
    directory_script='/home/pdavid/Bureau/Updated_BCs_2/Figures_and_Tests/Off_center'
    csv_script='/home/pdavid/Bureau/Updated_BCs_2/Figures_and_Tests/Off_center/csv_outputs'
else: #Auto_58
    directory='/home/pdavid/Bureau/Code/Updated_BCs_2/Code/'
    directory_script='/home/pdavid/Bureau/Code/Updated_BCs_2/Figures_and_Tests/Off_center'
    csv_script='/home/pdavid/Bureau/Code/Updated_BCs_2/Figures_and_Tests/Off_center/csv_outputs'
os.chdir(directory)



import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Testing import Testing
from Module_Coupling import assemble_SS_2D_FD
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos, plot_sketch, get_MRE


#0-Set up the sources
#1-Set up the domain
D=1
L=5
cells=5
h_coarse=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=10
#Rv=np.exp(-2*np.pi)*h_ss

C0=2*np.pi

Da_t=10
#Metabolism Parameters
M=Da_t*D/L**2
phi_0=0.4
conver_residual=5e-5
stabilization=0.5

x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, int(L//h_coarse))
y_coarse=x_coarse
directness=1


pos_s=np.array([[0.5,0.5]])*L
S=len(pos_s)
Rv=0.001+np.zeros(S)

K_eff=C0/(np.pi*Rv**2)

print(pos_s)
print(x_coarse)

C_v_array=np.ones(S)

BC_type=np.array(["Dirichlet", "Dirichlet", "Dirichlet", "Dirichlet"])
BC_value=np.zeros(4)

# =============================================================================
# BC_value=np.array([0,0,0,0.2])
# BC_type=np.array(['Neumann', 'Neumann', 'Neumann', 'Dirichlet'])
# =============================================================================


#What comparisons are we making 
COMSOL_reference=0
non_linear=0
Peaceman_reference=1
coarse_reference=1

#%%
plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)

#%%

t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)

s_Multi_cart_linear, q_Multi_linear=t.Multi()
Multi_rec_linear,_,_=t.Reconstruct_Multi(0,1)

FV_solution, FV_q=t.Linear_FV_Peaceman(1)
FV_matrix=FV_solution.reshape(cells*ratio, cells*ratio)





#%% - Validation

plt.imshow(FV_matrix,extent=[0,L,0,L], origin='lower'); plt.colorbar();
plt.title("Reference Peaceman model")
plt.show()

#%%
plt.imshow(Multi_rec_linear-FV_matrix,extent=[0,L,0,L], origin='lower'); plt.colorbar();
plt.title("Validation - coupling model")
plt.show()
base_q=FV_q
print("MRE q base= ", get_MRE(FV_q, q_Multi_linear))


# In[ ]:


def get_plots_through_sources(phi_mat, SS_phi_mat ,pos_s, rec_x,rec_y):
    for i in pos_s:
        pos=coord_to_pos(rec_x, rec_y, i)
        pos_x=int(pos%len(rec_x))
        plt.plot(rec_y, phi_mat[:,pos_x], label="coupling")
        plt.plot(rec_y, SS_phi_mat[:,pos_x], label="validation")
        plt.axvline(x=i[1])
        plt.legend()
        plt.show()
        
        


get_plots_through_sources(Multi_rec_linear, FV_matrix,pos_s, t.x_fine, t.y_fine)




# =============================================================================
# pos=coord_to_pos(SS.x, SS.y, pos_s[0])
# pos_x=int(pos%len(SS.x))
# sol_FV.reshape(len_x_FV, len_y_FV)[49,pos_x]=0
# plt.plot(SS.y, Multi_rec_linear[:,pos_x], label="coupling")
# plt.plot(SS.y, phi_SS[:,pos_x], label="validation")
# plt.plot(SS.y, sol_FV.reshape(len_x_FV, len_y_FV)[:,pos_x], label="Peaceman")
# 
# 
# =============================================================================
# OFF CENTERING

def full_L2_comarison(pos_s, Rv, h_coarse, x_coarse, y_coarse, K_eff, D, directness, C0, ratio, L, no_interpolation, *non_linear ):
    
    t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)
    t.no_interpolation=no_interpolation #if ==1 there won't be interpolation
    if non_linear:
        M,phi_0=non_linear
        s_FV, q = t.Multi(M,phi_0)
        sol_FV,FV_q_array=t.Metab_FV_Peaceman(M, phi_0, 1)
        
    else:
        #solve the problem through the multiscale model 
        s_FV, q = t.Multi()
        #The reconstructed 2D field:
        Multi_rec_linear,_,_ = t.Reconstruct_Multi(0,0)
        
    
        #Compute the refined solution using Peaceman coupling
        sol_FV, FV_q_array=t.Linear_FV_Peaceman(1)
        sol_FV=sol_FV.reshape(cells*ratio, cells*ratio)
    
        
    return(FV_q_array, q)

def full_L2_comarison_no_multi(pos_s, Rv, h_coarse, x_coarse, y_coarse, K_eff, D, directness, C0, ratio, L, no_interpolation,*non_linear):
    
    t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)
    if non_linear:
        M,phi_0=non_linear
        #Compute the refined solution using Peaceman coupling
        _,Peac_q=t.Metab_FV_Peaceman(M,phi_0,1)
    
        t.ratio=1
        _, FV_q=t.Metab_FV_Peaceman(M,phi_0,0)
    else:
        #Compute the refined solution using Peaceman coupling
        _,Peac_q=t.Linear_FV_Peaceman(1)
    
    
        t.ratio=1
        _, FV_q=t.Linear_FV_Peaceman(0)
    

    return(Peac_q, FV_q)


# Let's evaluate the off-centering

# In[118]:

points=5
off=np.linspace(0,h_coarse/2,points)*0.95
matrix_L2_error_Peac_off_interp=np.zeros((points, points))
matrix_L2_error_Peac_off_no_interp=np.zeros((points, points))

for no_interp in np.array([0,1]):
    ci=0
    for i in off:
        cj=0
        for j in off:
            pos_s=np.array([[0.5,0.5]])*L+np.array([i,j])
            FV_q, SS_q=full_L2_comarison(pos_s, Rv, h_coarse, x_coarse, y_coarse, K_eff, D, directness, C0, ratio, L, no_interp)
            if no_interp:
                matrix_L2_error_Peac_off_no_interp[cj,ci]=FV_q-SS_q
            else:
                matrix_L2_error_Peac_off_interp[cj,ci]=FV_q-SS_q
            cj+=1
        ci+=1

matrix_L2_error_Peac_off_no_interp=np.abs(matrix_L2_error_Peac_off_no_interp)

matrix_L2_error_Peac_off_interp=np.abs(matrix_L2_error_Peac_off_interp)

#%%

plt.imshow(matrix_L2_error_Peac_off_no_interp, extent=[0,0.5,0,0.5],origin='lower'); plt.colorbar()
plt.title("Absolute error $\mathcal{I}^{simple}_{\phi}$")
plt.xlabel("off-centering x ($\mu m$)")
plt.ylabel("off-centering y ($\mu m$)")
plt.show()

plt.imshow(matrix_L2_error_Peac_off_no_interp/base_q, extent=[0,0.5,0,0.5],origin='lower'); plt.colorbar()
plt.title("Relative error $\mathcal{I}^{simple}_{\phi}$")
plt.xlabel("off-centering x ($\mu m$)")
plt.ylabel("off-centering y ($\mu m$)")
plt.show()

plt.imshow(matrix_L2_error_Peac_off_interp, extent=[0,0.5,0,0.5],origin='lower'); plt.colorbar()
plt.title("Absolute error $\mathcal{I}_{\phi}$")
plt.xlabel("off-centering x ($\mu m$)")
plt.ylabel("off-centering y ($\mu m$)")
plt.show()

plt.imshow(matrix_L2_error_Peac_off_interp/base_q, extent=[0,0.5,0,0.5],origin='lower'); plt.colorbar()
plt.xlabel("off-centering x ($\mu m$)")
plt.ylabel("off-centering y ($\mu m$)")
plt.title("Relative error $\mathcal{I}_{\phi}$")
plt.show()

plt.plot(np.arange(len(off)), matrix_L2_error_Peac_off_no_interp[np.arange(len(off)), np.arange(len(off))]/base_q,'-o', label='No interpolation')
plt.plot(np.arange(len(off)), matrix_L2_error_Peac_off_interp[np.arange(len(off)), np.arange(len(off))]/base_q,'-o', label='Interpolation $\mathcal{I}_{\phi}$')
plt.xlabel("Position along the diagonal")
plt.ylabel("Relative error flux")
plt.title("Relative errors along the diagonal")
plt.legend()


# In[118]:
import pdb
points=5
off=np.linspace(0,h_coarse/2,points)*0.95
matrix_L2_error_Peac_off_interp=np.zeros((points, points))
matrix_L2_error_Peac_off_no_interp=np.zeros((points, points))

for no_interp in np.array([0,1]):
    ci=0
    for i in off:
        cj=0
        for j in off:
            pos_s=np.array([[0.5,0.5]])*L+np.array([i,j])
            FV_q, SS_q=full_L2_comarison(pos_s, Rv, h_coarse, x_coarse, y_coarse, K_eff, D, directness, C0, ratio, L, no_interp,M,phi_0)
            if no_interp:
                matrix_L2_error_Peac_off_no_interp[cj,ci]=FV_q-SS_q
            else:
                matrix_L2_error_Peac_off_interp[cj,ci]=FV_q-SS_q
            cj+=1
        ci+=1

matrix_L2_error_Peac_off_no_interp_metab=np.abs(matrix_L2_error_Peac_off_no_interp)

matrix_L2_error_Peac_off_interp_metab=np.abs(matrix_L2_error_Peac_off_interp)

#%%

