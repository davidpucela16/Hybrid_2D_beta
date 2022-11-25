# -*- coding: utf-8 -*-

#Define Computer
import os
Malphigui=1
if Malphigui:
    directory='/home/pdavid/Bureau/Hybrid_2D_beta/Code' #Malpighi
    directory_script='/home/pdavid/Bureau/Hybrid_2D_beta/Figures_and_Tests/off_center_COMSOL'
    csv_directory='/home/pdavid/Bureau/Hybrid_2D_beta/Figures_and_Tests/off_center_COMSOL/csv_outputs'
else: #Auto_58
    directory='/home/pdavid/Bureau/Code/Updated_BCs_2/Code/'
    directory_script='/home/pdavid/Bureau/Code/Updated_BCs_2/Figures_and_Tests/oFff_center'
    csv_directory='/home/pdavid/Bureau/Code/Updated_BCs_2/Figures_and_Tests/off_center/csv_outputs'
os.chdir(directory)



import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Testing import Testing
from Module_Coupling import assemble_SS_2D_FD
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos, plot_sketch, get_MRE

import pandas as pd
#0-Set up the sources
#1-Set up the domain
D=1
L=240
cells=5
h_coarse=L/cells
#ratio=int(np.max((h_ss/0.1,6)))
ratio=10
#Rv=np.exp(-2*np.pi)*h_ss

C0=1

Da_t=10
#Metabolism Parameters
M=Da_t*D/L**2
phi_0=0.4
conver_residual=5e-5
stabilization=0.5

x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, int(L//h_coarse))
y_coarse=x_coarse
directness=2


pos_s=np.array([[0.5,0.5]])*L
S=len(pos_s)
alpha=50
Rv=L/alpha+np.zeros(1)

K_eff=C0/(np.pi*Rv**2)

print(pos_s)
print(x_coarse)

C_v_array=np.ones(S)

BC_type=np.array(["Periodic", "Periodic", "Neumann", "Dirichlet"])
BC_value=np.array([0,0,0,0.2])

# =============================================================================
# BC_value=np.array([0,0,0,0.2])
# BC_type=np.array(['Neumann', 'Neumann', 'Neumann', 'Dirichlet'])
# =============================================================================


#What comparisons are we making 
COMSOL_reference=1
non_linear=0
Peaceman_reference=0
coarse_reference=1



#%%
plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)

q_FEM=pd.read_fwf(directory_script + '/q.txt').columns.astype(float).to_numpy().reshape(5,5)

field_file=directory_script + '/00' + '/contour.txt'
df=pd.read_fwf(field_file, infer_nrows=500)
ref_data=np.array(df).T #reference 2D data from COMSOL
FEM_x=ref_data[0]
FEM_y=ref_data[1]
FEM_phi=ref_data[2]

#%%

t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)

s_Multi_cart_linear, q_Multi_linear=t.Multi()

Multi_rec_linear,_,_=t.Reconstruct_Multi(0,1, FEM_x, FEM_y)


#%%
plt.tripcolor(FEM_x, FEM_y, FEM_phi - Multi_rec_linear) 
plt.colorbar()
plt.title("Validation - coupling model")
plt.show()
print("MRE q base= ", get_MRE(q_FEM[0], q_Multi_linear))

#%%
FV_noPeac_FV, FV_noPeac_q=t.Linear_FV_Peaceman(0)
print("MRE q no Peaceman Model= ", get_MRE(q_FEM[0], FV_noPeac_q))



#%%
points=5
off=np.linspace(0,h_coarse/2-h_coarse/10,points)

pd.DataFrame(off/h_coarse).to_csv(csv_directory + '/off.csv', sep=',', index=None)


q_Multi_int=np.zeros((points, points))
q_Multi_no_int=np.zeros((points, points))

for no_interp in np.array([0,1]):
    c_tot=0
    ci=0
    for i in off:
        cj=0
        for j in off:
            pos_s=np.array([[0.5,0.5]])*L+np.array([j,i])
            t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, 1, C_v_array, BC_type, BC_value)
            if no_interp:
                t.no_interpolation=1 #if ==1 there won't be interpolation
                _, q = t.Multi()
                q_Multi_no_int[ci, cj]=q
            else:
                t.no_interpolation=0 #if ==1 there won't be interpolation
                _, q = t.Multi()
                q_Multi_int[ci, cj]=q
            cj+=1
            c_tot+=1
        ci+=1
    
#%%
b=pd.DataFrame(np.ndarray.flatten(q_Multi_no_int))
b.columns=['q no interpolation']
b.to_csv(csv_directory + '/no_interp.csv', sep=',', index=None)

b=pd.DataFrame(np.ndarray.flatten(q_Multi_int))
b.columns=['q interpolation']
b.to_csv(csv_directory + '/interp.csv', sep=',', index=None)

b=pd.DataFrame(np.ndarray.flatten(q_FEM))
b.columns=['q FEM']
b.to_csv(csv_directory + '/q_FEM.csv', sep=',', index=None)

        #%%
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
extent=[-0.1,0.9,-0.1,0.9]    
    
plt.imshow(np.abs((q_Multi_int-q_FEM)/q_FEM), extent=extent,origin='lower')
plt.xticks(off*2/h_coarse)
plt.yticks(off*2/h_coarse)
plt.xlabel('$\dfrac{off \, centering_x}{h}$')
plt.ylabel('$\dfrac{off \, centering_y}{h}$')
cbar = plt.colorbar(shrink=0.6, pad=0.1)
plt.title('Relative error $\mathcal{I}_{\phi}$')
cbar.formatter.set_powerlimits((0, 0))
plt.savefig(directory_script + '/Interpolation_matrix.pdf')
plt.show()


plt.imshow(np.abs((q_Multi_no_int-q_FEM)/q_FEM), extent=extent,origin='lower')
plt.xticks(off*2/h_coarse)
plt.yticks(off*2/h_coarse)
plt.xlabel('$\dfrac{off \, centering_x}{h}$')
plt.ylabel('$\dfrac{off \, centering_y}{h}$')
cbar = plt.colorbar(shrink=0.6, pad=0.1)
plt.title('Relative error no interpolation')
cbar.formatter.set_powerlimits((0, 0))
plt.savefig(directory_script + '/No_interpolation_matrix.pdf')
plt.show()


