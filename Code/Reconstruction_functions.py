# -*- coding: utf-8 -*-

import numpy as np 
from Green import kernel_green_neigh, Green
from Neighbourhood import get_neighbourhood, get_multiple_neigh, get_uncommon
from Small_functions import pos_to_coords, get_boundary_vector


def set_boundary_values(BC_type, BC_value, phi_FV, phi_q, x, y, boundary_vector,h,D, ass_object):
    """Returns the averaged values at the boundary surfaces. The return values 
    are the values of the slow potential with reference of the cell that contains it
    
    basically retuns s_{km} for \partial V_{km} \in \partial \Omega"""
    b_s_values=np.zeros((4, len(x)))
    b=0
    array_opposite=np.array([1,0,3,2])
    for bc in BC_type:
        #b represents each of the boundaries (goes from 0 to 3)
        #bc represents the type of BC (string)
        #k goes through each of the cells of each boundary k \in (0,len(x))
        normal=np.array([[0,1],[0,-1],[1,0],[-1,0]])[b]   

        c=0
        for k in boundary_vector[b,:]:
            #Loop goes through each of the boundary cells
            m=boundary_vector[array_opposite[b],c]
            r_k_face_kernel, r_k_grad_face_kernel, r_m_face_kernel ,r_m_grad_face_kernel= ass_object.get_interface_kernels(k,normal,m)
            
            if bc=='Dirichlet':
                s_km = BC_value[b] - np.dot(r_k_face_kernel,phi_q)/h 
            if bc=='Neumann':
                s_km = phi_FV[k] - np.dot(r_k_grad_face_kernel, phi_q)/2 - BC_value[b]*h/D
            if bc=='Periodic':
                
                #The division by h is because the kernel calculates the integral, what we 
                #need is an average value per full cell
                
                jump_m=0.5*np.dot(r_m_grad_face_kernel, phi_q) + np.dot(r_m_face_kernel, phi_q)/h
                jump_k=-0.5*np.dot(r_k_grad_face_kernel, phi_q) - np.dot(r_k_face_kernel, phi_q)/h
                
                s_km=0.5*(phi_FV[k] + phi_FV[m] + jump_m + jump_k)
            
            b_s_values[b,c] = s_km #the slow term is registered
            c+=1    
                    
        b+=1
    return(b_s_values)

def separate_unk(ass_object, solution):
    """gets three arrays with the FV solution, reg terms, and sources"""
    xy=len(ass_object.x)*len(ass_object.y)
    s=len(ass_object.s_blocks)
    return(solution[:xy],solution[-s:]) 

def get_sub_x_y(orig_x, orig_y, orig_h, ratio):
    """returns the subgrid for that ratio and those originals"""
    h=orig_h/ratio 
    L=orig_x[-1]+orig_h/2
    num=int(L/h)
    h=L/num
    x=np.linspace(h/2, L-h/2, num)
    y=x
    return(x,y)

def modify_matrix(original, pos_x, pos_y, value_matrix):
    #pdb.set_trace()
    if not value_matrix.shape==(len(pos_y), len(pos_x)):
        raise TypeError("input shape not appropriate to include in the matrix")
    x_micro_len=original.shape[1]
    array_pos=np.array([], dtype=int)
    for j in pos_y:
        array_pos=np.concatenate((array_pos, pos_x+j*x_micro_len))
    original_flat=np.ndarray.flatten(original)
    original_flat[array_pos]=np.ndarray.flatten(value_matrix)
    return(original_flat.reshape(original.shape))


    
def bilinear_interpolation(corner_values, ratio):
    """The corner values must be given in the form of an np.array, in the following order:
        (0,0), (0,1), (1,0), (1,1)"""
    rec_block=np.zeros((ratio, ratio))
    A=np.array([[1,-1,-1,1],[0,0,1,-1],[0,1,0,-1],[0,0,0,1]])
    c=0
    for i in np.arange(0,1,1/ratio)+1/(2*ratio):
        d=0
        for j in np.arange(0,1,1/ratio)+1/(2*ratio):
            weights=A.dot(np.array([1,i,j,i*j]))
            rec_block[d, c]=weights.dot(corner_values)
            d+=1
        c+=1
# =============================================================================
#     for i in np.linspace(0,1,ratio):
#         d=0
#         for j in  np.linspace(0,1,ratio):
#             weights=A.dot(np.array([1,i,j,i*j]))
#             rec_block[d, c]=weights.dot(corner_values)
#             d+=1
#         c+=1
# =============================================================================
    return(rec_block)

def single_value_bilinear_interpolation(coordinates_point, coordinates_vertices, vert_values):
    """Everything must be given in the form of an np.array, in the following order:
    (0,0), (0,1), (1,0), (1,1)"""
    A=np.array([[1,-1,-1,1],[0,0,1,-1],[0,1,0,-1],[0,0,0,1]])
    Lx=coordinates_vertices[2,0]-coordinates_vertices[0,0]
    Ly=coordinates_vertices[1,1]-coordinates_vertices[0,1]
    i= (coordinates_point[0]-coordinates_vertices[0,0])/Lx#relative x position
    j= (coordinates_point[1]-coordinates_vertices[0,1])/Ly#relative y position
    weights=A.dot(np.array([1,i,j,i*j]))
    return(np.dot(vert_values, weights))
    
def coeffs_bilinear_int(coordinates_point, coordinates_vertices):
    A=np.array([[1,-1,-1,1],[0,0,1,-1],[0,1,0,-1],[0,0,0,1]])
    Lx=coordinates_vertices[2,0]-coordinates_vertices[0,0]
    Ly=coordinates_vertices[1,1]-coordinates_vertices[0,1]
    i= (coordinates_point[0]-coordinates_vertices[0,0])/Lx#relative x position
    j= (coordinates_point[1]-coordinates_vertices[0,1])/Ly#relative y position
    weights=A.dot(np.array([1,i,j,i*j]))
    return(weights)
    
def reduced_half_direction(phi_k, axis):
    """Function made to reduce by half the size of the matrix phi_k in one direction
    given by axis.
    A priori, axis=0 is for the y direction and axis=1 for the x direction ,just like np.concatenate"""
    ss=phi_k.shape
    if axis==1:
        #This would mean we reduce in the x direction therefore it is north or south boundary:
        st=int(ss[1]/2)
        aa=np.arange(st, dtype=int)
        to_return=(phi_k[:,2*aa]+phi_k[:,2*aa+1])/2
    elif axis==0:
        #This would mean we reduce in the y direction therefore it is north or south boundary:
        st=int(ss[0]/2)
        aa=np.arange(st, dtype=int)
        to_return=(phi_k[2*aa,:]+phi_k[2*aa+1,:])/2
    return(to_return)
    
def NN_interpolation(corner_values, ratio):
        """The corner values must be given in the form of an np.array, in the following order:
        (0,0), (0,1), (1,0), (1,1)"""
        rec_block=np.zeros((ratio, ratio))
        pos_corners=np.array([[0,0],[0,1],[1,0],[1,1]])
        c=0
        for i in np.arange(0,1,1/ratio)+1/(2*ratio):
            d=0
            for j in np.arange(0,1,1/ratio)+1/(2*ratio):
                dist=np.zeros(4)
                for k in range(4):
                    dist[k]=np.linalg.norm(pos_corners[k,:]-np.array([j,i]))
                
                rec_block[d, c]=corner_values[np.argmin(dist)] #takes the value of the closest one
                d+=1
            c+=1
        return(rec_block)


def green_to_block(phi_q, pos_center, original_h, ratio, neigh_blocks, s_blocks, pos_s,Rv):
    """This function will add the contribution from the given sources to each of 
    the sub-discretized block"""
    rec_block=np.zeros((ratio, ratio))
    c=0
    h=original_h
    for i in h*(np.arange(0,1,1/ratio)+1/(2*ratio)-1/2):
        d=0
        for j in h*(np.arange(0,1,1/ratio)+1/(2*ratio)-1/2):
            w=kernel_green_neigh(pos_center+np.array([i,j]), neigh_blocks, pos_s,s_blocks, Rv)
            rec_block[d, c]=w.dot(phi_q)
            d+=1
        c+=1
    return(rec_block)
    
    
def tool_piece_wise_constant_ratio(ratio, matrix):
    """This function is made to return an array that mimicks a FV with 
    piece wise constant shape functions. 
    In imshow it's easy to plot a FV"""
    A=np.zeros(np.array(matrix.shape)*ratio)
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            init_j=j*ratio
            end_j=(j+1)*ratio
            
            init_i=i*ratio
            end_i=(i+1)*ratio
            
            A[init_j:end_j,init_i:end_i]=matrix[j,i]
    return(A)


def rotate_clockwise(matrix):
    sh=matrix.shape
    new_matrix=np.zeros((sh[1], sh[0]))
    for i in range(sh[0]):        
        for j in range(sh[1]):
            row=j
            col=sh[0]-i-1
            new_matrix[row, col]=matrix[i,j]
    return(new_matrix)


def rotate_180(matrix):
    new=rotate_clockwise(matrix)
    new=rotate_clockwise(new)
    return(new)

def rotate_counterclock(matrix):
    new=rotate_clockwise(matrix)
    new=rotate_clockwise(new)
    new=rotate_clockwise(new)
    return(new)



def coarse_cell_center_rec(x, y, phi_FV, pos_s, s_blocks, phi_q, directness, Rv):
        """PREVIOUSLY IT WAS CALLED coarse_NN_rec!!!!!!!!!!!!!!!!!!!!!!
        
        phi_FV is given in matrix form
        This funcion does a coarse mesh reconstruction with average u values on the cells plus
        singular term values at the cell's center"""
        
        if len(np.array(phi_FV.shape))==1:
            phi_FV=phi_FV.reshape(len(y),len(x))
        
        rec=np.zeros((len(y), len(x)))
        
        for i in range(len(y)):
            for j in range(len(x)):
                neigh=get_neighbourhood(directness, len(x), i*len(x)+j)
                Ens=np.arange(len(s_blocks))[np.in1d(s_blocks, neigh)]#sources in the neighbourhood
                rec[i,j]=phi_FV[i, j]
                for k in Ens:
                    rec[i,j]+=Green(pos_s[k], np.array([x[j],y[i]]), Rv[k])*phi_q[k]
        return(rec)
    
def get_block_reference(block_ID, ref_neighbourhood, directness, xlen, s_blocks, 
                        uni_s_blocks,phi_FV, phi_q, pos_s, Rv, x, y):
    """gets a single block in reference to the singular term in ref_neighourhood"""
    phi=phi_FV
    #get the uncommon blocks
    a=get_uncommon(ref_neighbourhood, get_neighbourhood(directness, xlen, block_ID))
    #get the value of the uncommon singular term at the given block
    unc_sing_array=kernel_green_neigh(pos_to_coords(x, y, block_ID), a, pos_s, s_blocks, Rv)
    value=unc_sing_array.dot(phi_q)
    
    #substract the uncommon sources from the unknown
    return(phi[block_ID]-value)
    
def get_unk_same_reference(blocks, directness, xlen, s_blocks, uni_s_blocks, phi_FV, phi_q, pos_s, Rv, x, y):
    """Designed to get a given set the unknowns in the same frame of reference
    regarding the singular term.
    The frame of reference will the that one accounting for influence of all the 
    sources in the ensemble of all the neighbourhoods of the blocks given to the function"""
    #pdb.set_trace()
    ens_neigh=get_multiple_neigh(directness, xlen, blocks)
    total_sources=np.in1d(s_blocks, ens_neigh)
    
    unk_extended=np.zeros((len(blocks)))
    c=0
    for i in blocks:
        unk_extended[c]=get_block_reference(i, ens_neigh, directness, xlen, s_blocks, 
                                            uni_s_blocks,phi_FV,  phi_q, pos_s, Rv, x, y)
        c+=1
    return(unk_extended)
        


    
def get_boundar_adj_blocks(x,y,coords):
    """This function is meant to find the adjacent boundary blocks
    
    Returns the 2 closes elements of the boundary regardless of distance"""
    
    dist_x=x-coords[0]
    dist_y=y-coords[1]
    
    boundary_vector=np.unique(np.ndarray.flatten(get_boundary_vector(len(x), len(y))))
    
    
    X,Y=np.meshgrid(dist_x, dist_y)
    
    arr=(X**2+Y**2)**0.5
    arr=np.ndarray.flatten(arr) #array of distances
    
    arr_bound=arr[boundary_vector]
    ind_sort=arr_bound.argsort()
    
    ordered_indices=boundary_vector[ind_sort]   
    
    return(ordered_indices[:2])



def get_average_rapid(phi_FV, phi_q, directness, pos_s,x, y, s_blocks, Rv, h, C_v_array, K_eff):
    """Calculates the average per FV cell of the local rapid potential
    
        - One value per FV cell for the rapid potential which is the average value in each cell
        - The value of the slow potential is the standard average value per FV cell"""
    w_i=np.array([1,4,1,4,16,4,1,4,1])/36
    corr=np.array([[-1,-1,],
                   [0,-1],
                   [1,-1],
                   [-1,0],
                   [0,0],
                   [1,0],
                   [-1,1],
                   [0,1],
                   [1,1]])*h/2
    slow=np.ndarray.flatten(phi_FV)
    rec_sing=np.zeros(phi_FV.size)
    for k in range(phi_FV.size):
        #pdb.set_trace()
        neigh=get_neighbourhood(directness, len(x), k)
        ids=np.in1d(s_blocks, neigh)
        phi_bar=np.sum((C_v_array-phi_q/(K_eff*np.pi*Rv**2))[ids])
        slow[k]-=phi_bar
        rec_sing[k]+=phi_bar
        for i in range(len(w_i)):
            
            pos_xy=pos_to_coords(x, y, k)+corr[i]
            kernel_G=kernel_green_neigh(pos_xy,neigh ,
                                        pos_s, s_blocks, Rv)
            rapid=np.dot(kernel_G, phi_q)
            rec_sing[k]+=w_i[i]*rapid
        
    return(rec_sing.reshape(len(y), len(x)), slow.reshape(len(y), len(x)))


