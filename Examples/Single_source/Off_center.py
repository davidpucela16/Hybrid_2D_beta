# -*- coding: utf-8 -*-

import os 
directory='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Updated_BCs/Code'
os.chdir(directory)

import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos


#0-Set up the sources
#1-Set up the domain
D=1
L=5
cells=5
h_ss=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=20
#Rv=np.exp(-2*np.pi)*h_ss

C0=2*np.pi


x_ss=np.linspace(h_ss/2, L-h_ss/2, int(L//h_ss))
y_ss=x_ss
directness=2


pos_s=np.array([[0.5,0.5]])*L
S=len(pos_s)
Rv=0.0001+np.zeros(S)

K_eff=C0/(np.pi*Rv**2)

print(pos_s)
print(x_ss)

C_v_array=np.ones(S)
# Assembly and resolution of the coupling model 

# In[ ]:


t=assemble_SS_2D_FD(pos_s, Rv, h_ss,L, K_eff, D, directness) #Initialises the variables of the object
t.solve_problem(np.array(['Dirichlet', 'Dirichlet','Dirichlet','Dirichlet']),np.array([0,0,0,0]), C_v_array)

s_FV=t.s_FV.reshape(len(t.x), len(t.y))
q=t.q

# Validation for large scale discrepancies between the diameter of the vessel and the domain. This is necessary since the vessel is approximated as a delta function

# In[ ]:


#Reconstruction microscopic field
#pdb.set_trace()
a=reconstruction_sans_flux(np.concatenate((np.ndarray.flatten(s_FV), t.q)), t, L,ratio, directness)
p=a.reconstruction()   
a.reconstruction_boundaries_short(np.array(['Dirichlet','Dirichlet','Dirichlet','Dirichlet']), np.array([0,0,0,0]))
a.rec_corners()
plt.imshow(a.rec_final,extent=[0,L,0,L], origin='lower'); plt.colorbar();
plt.title("reconstructed coupling model")
plt.show()


#%% - Validation

FV=FV_validation(L, ratio*cells, pos_s, C_v_array, D, K_eff, Rv, np.array([0,0,0,0]))
FV_solution=FV.solve_linear_system().reshape(ratio*cells, ratio*cells)
plt.imshow(FV_solution.reshape(ratio*cells, ratio*cells),extent=[0,L,0,L], origin='lower'); plt.colorbar();
plt.title("Reference Peaceman model")
plt.show()

#%%
plt.imshow(a.rec_final-FV_solution,extent=[0,L,0,L], origin='lower'); plt.colorbar();
plt.title("Validation - coupling model")
plt.show()

base_q=FV.get_q_linear()

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
        
        


get_plots_through_sources(a.rec_final, FV_solution,pos_s, FV.x, FV.y)




# =============================================================================
# pos=coord_to_pos(SS.x, SS.y, pos_s[0])
# pos_x=int(pos%len(SS.x))
# sol_FV.reshape(len_x_FV, len_y_FV)[49,pos_x]=0
# plt.plot(SS.y, a.rec_final[:,pos_x], label="coupling")
# plt.plot(SS.y, phi_SS[:,pos_x], label="validation")
# plt.plot(SS.y, sol_FV.reshape(len_x_FV, len_y_FV)[:,pos_x], label="Peaceman")
# 
# 
# =============================================================================
# OFF CENTERING



def full_L2_comarison(pos_s, Rv, h_ss, x_ss, y_ss, K_eff, D, directness, C0, ratio, L, no_interpolation):

    t=assemble_SS_2D_FD(pos_s, Rv, h_ss,L, K_eff, D, directness) #Initialises the variables of the object
    t.pos_arrays() #Creates the arrays that contain the information about the sources 
    t.initialize_matrices() 
    
    t.no_interpolation=no_interpolation #if ==0 there won't be interpolation
    
    M=t.assembly_sol_split_problem(np.array(['Dirichlet','Dirichlet','Dirichlet','Dirichlet']),np.array([0,0,0,0])) #The argument is the value of the Dirichlet BCs

    t.solve_problem(np.array(['Dirichlet', 'Dirichlet','Dirichlet','Dirichlet']),np.array([0,0,0,0]), C_v_array)

    s_FV=np.ndarray.flatten(t.s_FV)
    q=t.q

    #Reconstruction microscopic field
    #pdb.set_trace()
    a=reconstruction_sans_flux(np.concatenate((s_FV, q)), t, L,ratio, directness)
    p=a.reconstruction()   
    a.reconstruction_boundaries_short(np.array(['Dirichlet','Dirichlet','Dirichlet','Dirichlet']),np.array([0,0,0,0]))
    a.rec_corners()
    
    plt.imshow(a.rec_final,extent=[0,L,0,L], origin='lower'); plt.colorbar();
    plt.title("reconstruction of the coupling model")
    plt.show()
    
    FV=FV_validation(L, ratio*cells, pos_s, C_v_array, D, K_eff, Rv, np.array([0,0,0,0]))
    peac=FV.solve_linear_system()
    sol_FV=peac.reshape(cells*ratio, cells*ratio)
    FV_q_array=FV.get_q_linear()

    plt.imshow(a.rec_final-sol_FV,extent=[0,L,0,L], origin='lower'); plt.colorbar();
    plt.title("Validation - coupling model")
    plt.show()

    get_plots_through_sources(a.rec_final, sol_FV,  pos_s, FV.x, FV.y)

    return(FV_q_array, q)


# Let's evaluate the off-centering

# In[118]:


points=5
off=np.linspace(0,h_ss/2,points)*0.95
matrix_L2_error_Peac_off_interp=np.zeros((points, points))
matrix_L2_error_Peac_off_no_interp=np.zeros((points, points))

for no_interp in np.array([0,1]):
    ci=0
    for i in off:
        cj=0
        for j in off:
            
            pos_s=np.array([[0.5,0.5]])*L+np.array([i,j])
            FV_q, SS_q=full_L2_comarison(pos_s, Rv, h_ss, x_ss, y_ss, K_eff, D, directness, C0, 10, L, no_interp)
            if no_interp:
                matrix_L2_error_Peac_off_no_interp[cj,ci]=FV_q-SS_q
            else:
                matrix_L2_error_Peac_off_interp[cj,ci]=FV_q-SS_q
            cj+=1
        ci+=1
        
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

plt.plot(np.arange(len(off)), matrix_L2_error_Peac_off_no_interp[np.arange(len(off)), np.arange(len(off))]/base_q, label='Simple interpolation $\mathcal{I}^{simple}_{\phi}$')
plt.plot(np.arange(len(off)), matrix_L2_error_Peac_off_interp[np.arange(len(off)), np.arange(len(off))]/base_q, label='Complex interpolation $\mathcal{I}_{\phi}$')
plt.xlabel("Position along the diagonal")
plt.ylabel("Relative error flux")
plt.title("Relative errors along the diagonal")
plt.legend()
#%%

fig, axs = plt.subplots(2,2, figsize=(15,15))
fig.tight_layout(pad=4.0)
axs[1,0].plot(SS.y, a.rec_final[:,pos_x], label="coupling")
axs[1,0].scatter(SS.y, p_sol_mat[:,pos_x], label="Peaceman", c='r')
axs[1,0].set_title("absolute error of the flux \n estimation for ratio={}".format(ratio))

axs[1,0].set_ylabel("absolute value [$kg m^{-1} s^{-1}$]")
axs[1,0].set_xlabel("source ID")
axs[1,0].legend()

d=axs[1,1].scatter(np.arange(len(p_q)),(1e3)*np.abs(p_q-q)/np.abs(p_q))
axs[1,1].set_title("relative error * $10^{3}$")
axs[1,1].set_ylabel("relative err")
axs[1,1].set_xlabel("source ID")

b=axs[0,1].imshow(p_sol_mat, extent=[0,L,0,L],origin='lower')
axs[0,1].set_xlabel("$\mu$m")
axs[0,1].set_ylabel("$\mu$m")
axs[0,1].set_title("validation reconstruction")
divider = make_axes_locatable(axs[0,1])
cax = divider.append_axes('right', size='10%', pad=0.05)
fig.colorbar(b, cax=cax,orientation='vertical')

c=axs[0,0].imshow((a.rec_final-p_sol_mat)*1e3, extent=[0,L,0,L], origin='lower')
axs[0,0].set_xlabel("$\mu$m")
axs[0,0].set_ylabel("$\mu$m")
axs[0,0].set_title("absolute error of the reconstructed $\phi$ \n multiplied by $10^3$")
divider = make_axes_locatable(axs[0,0])
cax = divider.append_axes('right', size='10%', pad=0.05)
fig.colorbar(c, cax=cax,orientation='vertical')

