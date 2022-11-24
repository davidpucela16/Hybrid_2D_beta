



        
import numpy as np
from Green import Green, kernel_green_neigh
from Module_Coupling import assemble_SS_2D_FD, get_neighbourhood, get_boundary_vector, pos_to_coords, coord_to_pos
from Reconstruction_functions import coeffs_bilinear_int, green_to_block,NN_interpolation,get_boundar_adj_blocks,single_value_bilinear_interpolation, get_unk_same_reference, bilinear_interpolation, get_sub_x_y, modify_matrix, set_boundary_values, rotate_180, rotate_clockwise, rotate_counterclock, reduced_half_direction, separate_unk
from Neighbourhood import get_multiple_neigh, get_uncommon
from Small_functions import get_4_blocks, pos_to_coords
from reconst_and_test_module import real_NN_rec
import pdb
from numba import jit
import numba as nb
import matplotlib.pyplot as plt
from Testing import Testing
from Green import Green

def Local_extended_rec(L,cells, point, directness, s_FV, q_array, pos_s, s_blocks, Rv):
    s_FV=np.ndarray.flatten(s_FV)
    #For the non boundary nodes
    h_coarse=np.around(L/cells)
    x_dual=np.linspace(0,L,cells+1)
    y_dual=np.linspace(0,L,cells+1)
    
    xlen=len(x_dual)
    ylen=len(y_dual)
    
    x_pos=np.argmin((point[0]-x_dual)**2)
    y_pos=np.argmin((point[1]-y_dual)**2)
    
    x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, cells)
    y_coarse=np.linspace(h_coarse/2, L-h_coarse/2, cells)
    
    n,m,p,q=get_4_blocks(point, x_coarse, y_coarse, h_coarse)
    w=coeffs_bilinear_int(point,pos_to_coords(x_coarse, y_coarse, np.array([n,m,p,q])).T)
    #Rapid term values:
    r=np.zeros(0, dtype=float)
    
    ens_neigh=get_multiple_neigh(directness, len(x_coarse), np.array([n,m,p,q]))
    #if point[0]<3.5 and point[0]>2.5: pdb.set_trace()
    for k in np.array((n,m,p,q)):
        
        neigh=get_neighbourhood(directness, len(x_coarse), k)
        kron_delt=get_uncommon(ens_neigh, neigh)

        P_ji_delta=kernel_green_neigh(pos_to_coords(x_coarse, y_coarse, k), 
                                      kron_delt, pos_s, s_blocks, Rv).dot(q_array)
        #neigh=get_multiple_neigh(directness, len(x_coarse), np.array([n,m,p,q]))
        
        value=kernel_green_neigh(point, ens_neigh, pos_s, s_blocks, Rv).dot(q_array)-P_ji_delta
        r=np.append(r,value)
# =============================================================================
#     k=coord_to_pos(x_coarse, y_coarse, point)
#     neigh=get_neighbourhood(directness, len(x_coarse), k)
#     value=kernel_green_neigh(point, neigh, pos_s, s_blocks, Rv).dot(q_array)
#     r[[n,m,m,q]==k]=value
# =============================================================================

    rec_s=w.dot(s_FV[[n,m,p,q]]) 
    rec_r=w.dot(r)
    return(rec_s, rec_r)
    
    #0-Set up the sources
#1-Set up the domain
alpha=50

Da_t=10
D=1
K0=1
L=5

cells=5
h_coarse=L/cells


#Definition of the Cartesian Grid
x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, int(np.around(L/h_coarse)))
y_coarse=x_coarse

#V-chapeau definition
directness=1
print("directness=", directness)


#pos_s=(1-np.array([[0.5,0.05+1/alpha/2]]))*L

pos_s=np.array([[x_coarse[1],y_coarse[1]]])-L/alpha
S=len(pos_s)
Rv=L/alpha+np.zeros(S)
#ratio=int(40/cells)*2
ratio=10

print("h coarse:",h_coarse)
K_eff=K0/(np.pi*Rv**2)



C_v_array=np.ones(S) 

BC_value=np.array([0,0,0,0])
BC_type=np.array(['Dirichlet','Dirichlet','Dirichlet','Dirichlet'])

from Testing import Testing
t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)
Multi_FV_linear, Multi_q_linear=t.Multi()


def rec_Local_extended_fully(x_array, y_array, L, cells,  directness, s_FV, q, pos_s, s_blocks, Rv):
    rec_s=np.zeros(len(x_array))
    rec_r=np.zeros(len(x_array))
    for i in range(len(x_array)):
        pos=np.array([x_array[i], y_array[i]])
        rec_s[i], rec_r[i]=Local_extended_rec(L,cells, pos, directness, s_FV, q, pos_s, s_blocks, Rv)
    return(rec_s,rec_r)


x=np.linspace(h_coarse/2, L-h_coarse/2, cells*5)
y=np.linspace(h_coarse/2, L-h_coarse/2, cells*5)




source_plot_old,_,_=t.Reconstruct_Multi(0,0,  np.zeros(len(y))+pos_s[0,0], y)


b,c=rec_Local_extended_fully(np.zeros(len(y))+pos_s[0,0], y, L, cells, directness, Multi_FV_linear, Multi_q_linear, pos_s, t.s_blocks, Rv)
plt.show()

#%%



def rec_Local_extended_fully(x_array, y_array, L, cells,  directness, s_FV, q, pos_s, s_blocks, Rv):
    rec_s=np.zeros(len(x_array))
    rec_r=np.zeros(len(x_array))
    for i in range(len(x_array)):
        pos=np.array([x_array[i], y_array[i]])
        rec_s[i], rec_r[i]=Local_extended_rec(L,cells, pos, directness, s_FV, q, pos_s, s_blocks, Rv)
    return(rec_s,rec_r)


x=np.linspace(h_coarse/2, L-h_coarse/2, cells*5)
y=np.linspace(h_coarse/2, L-h_coarse/2, cells*5)

X,Y=np.meshgrid(x,y)

rec_s,rec_r=rec_Local_extended_fully(np.ndarray.flatten(X),np.ndarray.flatten(Y), L, cells, directness, Multi_FV_linear, Multi_q_linear, pos_s, t.s_blocks, Rv)
import matplotlib.pyplot as plt
plt.tricontourf(np.ndarray.flatten(X), np.ndarray.flatten(Y), rec_s+rec_r, levels=100); plt.colorbar()


a,_,_=t.Reconstruct_Multi(0, 0, np.ndarray.flatten(X), np.ndarray.flatten(Y))


source_plot_old,_,_=t.Reconstruct_Multi(0,0,  np.zeros(len(y))+pos_s[0,0], y)


b,c=rec_Local_extended_fully(np.zeros(len(y))+pos_s[0,0], y, L, cells, directness, Multi_FV_linear, Multi_q_linear, pos_s, t.s_blocks, Rv)
plt.show()

#%%
def get_dual(L,cells):
    h_coarse=np.around(L/cells)
    x_dual=np.linspace(0,L,cells+1)
    y_dual=np.linspace(0,L,cells+1)
    return(x_dual, y_dual)

x_dual, y_dual=get_dual(L,cells)

plt.plot(y,b+c)
plt.plot(y,source_plot_old)
plt.scatter(y_coarse, Multi_FV_linear[1])


plt.vlines(x_coarse, 0, np.max(source_plot_old), color='red', linestyle='--')
#plt.vlines(x_dual, 0, np.max(source_plot_old), color='black', linestyle='--')
plt.xlim((0,L))
plt.show()


#%% - 1D test

def get_real_phi(phi_bar, Rv, rel_h, x, value, dist_value):
    A=(phi_bar-value)/np.log(Rv/(dist_value))
    B=t.phi_bar-A*np.log(Rv)
    real_phi=A*np.log(rel_h+x)+B
    return(real_phi)

def get_inf_phi(phi_bar, Rv, h_coarse, x,q):
    return(-q*np.log((x+h_coarse)/Rv)/(2*np.pi)+phi_bar)

def get_singular_along_x(x_array, pos_y, pos_s, q, phi_bar,Rv):
    arr=np.zeros(0)
    x_array=np.squeeze(x_array)
    c=0
    for s in pos_s:
        for i in x_array:
            x=np.array([i, pos_y])
            arr=np.append(arr, q[c]*Green(s, x, Rv)+phi_bar[c])
        c+=1
    return(arr)

rel_h=np.linalg.norm([Rv+2*h_coarse,Rv])/2

x=np.linspace(0,rel_h,100)
#x=np.linspace(-h_coarse, h_coarse, 200)


L=np.linalg.norm(pos_to_coords(x_coarse, y_coarse, 1*5+3)-pos_s[0])

A=(t.phi_bar-Multi_FV_linear[1,3])/np.log(Rv/(2*rel_h))
B=t.phi_bar-A*np.log(Rv)


real_phi=get_real_phi(t.phi_bar, Rv,rel_h, x, Multi_FV_linear[1,3], np.linalg.norm([Rv+2*h_coarse,Rv]))
inf_phi=get_inf_phi(t.phi_bar, Rv, h_coarse, x, Multi_q_linear)+t.phi_bar

plt.plot(real_phi, label='real_phi')
plt.plot(inf_phi, label='inf_phi')
real_s=real_phi-inf_phi
plt.plot(real_s, label='real_s')
plt.legend()


#%%
calc_s=np.concatenate((np.zeros(50)+Multi_FV_linear[1,2], np.zeros(50) + Multi_FV_linear[1,3]))
plt.plot(real_s); plt.plot(calc_s)
plt.ylim((0,0.3))
sf_2=x/h_coarse
sf_1=1-x/h_coarse
#%%
real_r=np.concatenate((inf_phi[0:50], np.zeros(50)))
real_s=real_phi-real_r

plt.plot(x,real_s[0]*sf_1 + real_s[-1]*sf_2+inf_phi*sf_1, label='rec')
plt.plot(x, inf_phi + real_s[0]*sf_1 + real_s[-1]*sf_2, label='new_rec')
plt.plot(x, real_phi, label='real')
plt.legend()

#%%
plt.plot((inf_phi+s[0])*sf_1+(s[-1]+inf_phi)*sf_2, label='reconstructed')
plt.plot(real_phi, label='real')
plt.legend()

#%%

NN=real_NN_rec(x_coarse, y_coarse, Multi_FV_linear, pos_s, t.s_blocks, Multi_q_linear, 5, h_coarse, directness, t.Rv, K_eff*np.pi*t.Rv**2, C_v_array)
sing=NN.add_singular()

slow=NN.rec

phi_bar=slow-NN.rec_plus_phibar


plt.plot(y,source_plot_new)
plt.plot(y,source_plot_old)
plt.scatter(NN.r_y, slow[5]-phi_bar[5])
#plt.plot(NN.r_y, sing[5]+phi_bar[5])
plt.plot(NN.r_y,NN.rec[5]+NN.add_singular()[5])

#%%


X2,Y2=np.meshgrid(x_coarse, y_coarse)

plt.tricontourf(np.ndarray.flatten(X), np.ndarray.flatten(Y), phi-a, levels=100)
plt.vlines(x_coarse, 0, L, color='red', linestyle='--')
plt.hlines(y_coarse, 0,L, color='red', linestyle='--')
plt.vlines(x_dual, 0, L, color='black', linestyle='--')
plt.hlines(y_dual, 0,L, color='black', linestyle='--')
plt.scatter(np.ndarray.flatten(X2), np.ndarray.flatten(Y2), marker='x', s=200, color='black')
plt.colorbar()

plt.show()
# =============================================================================
# if y_pos==0:
#     #south
#     if x_pos==0:
#         #south west
#         print("southwest")
#     elif x_pos==xlen-1:
#         #south east
#         print("southeast")
#     else:
#         #south
#         print("south")
# elif y_pos==ylen-1:
#     #north
#     if x_pos==0:
#         #north west
#         print("northwest")
#     elif x_pos==xlen-1:
#         #north east
#         print("northeast")
#     else: 
#         #norht
#         print("north")
# 
# elif x_pos==0:
#     #West
#     print("West")
# elif x_pos==xlen-1:
#     #East
#     print("East")
# 
# =============================================================================









