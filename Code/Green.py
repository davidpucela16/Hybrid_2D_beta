# -*- coding: utf-8 -*-

import numpy as np 
from Neighbourhood import get_multiple_neigh, get_uncommon, get_neighbourhood
print("IMPORTED")

#
def Green(q_pos, x_coord, Rv):
    """returns the value of the green function at the point x_coord, with respect to
    the source location (which is q_pos)"""
    #pdb.set_trace()
    if np.linalg.norm(q_pos-x_coord)>Rv:
        g=np.log(Rv/np.linalg.norm(q_pos-x_coord))/(2*np.pi)
    else:
        g=0
    return(g)

#
def grad_Green(q_pos, x_coord, normal, Rv):
    er=(x_coord-q_pos)/np.linalg.norm(q_pos-x_coord).astype(float)
    
    if np.linalg.norm(q_pos-x_coord)>Rv:
        g=-1/(2*np.pi*np.linalg.norm(q_pos-x_coord))
    else:
        g=1/(2*np.pi*Rv)
    return(g*(er.dot(normal.astype(float))))

def grad_Green_norm(normal, pos1, pos2):
    """normal gradient of the 2D Green's function from a delta located at pos_2
    and evaluated at pos_1"""
    r=pos1-pos2
    sc=np.dot(r, normal)
    return(sc/(2*np.pi*np.linalg.norm(r)**2))

#
def Sampson_grad_Green(a,b, q_pos,normal, Rv):
    """Integrates through the Samson Rule the normal gradient of the
    2D Green's function along a line running between a and b"""
    f0=grad_Green(q_pos, a, normal,Rv)
    f1=grad_Green(q_pos, a+(b-a)/2, normal,Rv)
    f2=grad_Green(q_pos, b, normal,Rv)
    L=np.linalg.norm(b-a)
    return(L*(f0+4*f1+f2)/6)

#
def Sampson_Green(a,b, q_pos, Rv):
    """Integrates through the Samson Rule the 2D Green's function along 
    a line running between a and b"""
    f0=Green(q_pos, a, Rv)
    f1=Green(q_pos, a+(b-a)/2, Rv)
    f2=Green(q_pos, b, Rv)
    L=np.linalg.norm(b-a)
    return(L*(f0+4*f1+f2)/6)




def block_grad_Green_norm_array(pos_s,s_blocks,  block_ID, pos_calcul, norm):
    """returns the array that will multiply the sources of the block_ID to obtain
    the gradient of the given block's singular term evaluated at pos_calcul
    pos_s -> array containing the position of the sources
    s_blocks -> Ordered array with the source containing blocks
    block_ID -> the source containing block
    pos_calcul -> point the gradient is evaluated
    norm -> normal vector"""
    s_IDs=np.where(s_blocks==block_ID)[0]
    grad_green=np.zeros(pos_s.shape[0])
    for i in s_IDs:
        grad_green[i]=grad_Green_norm(norm, pos_s[i], pos_calcul)
    return(grad_green)

#block_grad_Green_norm_array(t.pos_s, t.s_blocks, 21, np.array([0,1])+t.h, np.array([0,1]))  
                      
def kernel_green_neigh(position, neigh, pos_s, s_blocks, Rv):
    """Gets the value of the green's function at a given position with a given neighbourhood
    
    Returns the array to multiply the array of source_fluxes that will calculate
    the value of the singular term of the sources in the given neigh at the given point p_x
    
    $\sum_{j \in neigh} G(x_j, p_x)$"""
    IDs=np.arange(len(pos_s))[np.in1d(s_blocks, neigh)]
    array=np.zeros(len(pos_s))
    for i in IDs:
        array[i]=Green(position, pos_s[i], Rv[i])
    return(array)


    
def from_pos_get_green(x, y, pos_s, Rv ,corr, directness, s_blocks):
    """Get the value of the green's function for a given positions dictated by x and y
    array_pos contains the positions to evaluate. It's shape therefore: (:,2)
    x_c is the center of the Green's function
    Rv is the radius of the source
    
    Returns the array to multiply the value of the sources so it is equal to an array 
    with the value of the local singular term at each of the cell's centers applied a correction"""
    arr=np.zeros([len(x)*len(y), len(pos_s)])
    
    for j in range(len(y)):
        for i in range(len(x)):
            #pos=i+len(x)*j
            neigh=get_neighbourhood(directness, len(x), i+j*len(x))
            G_sub=kernel_green_neigh(np.array([x[i], y[j]])+corr, neigh, pos_s, s_blocks, Rv)
            arr[j*len(x)+i, :]=G_sub
    return(arr)
    
def kernel_integral_grad_Green_face(pos_s, set_of_IDs, a,b, normal, Rv):
    """Will return the kernel to multiply the array of {q}.
    It integrates the Green's function
    through the Sampson's rule over the line that links a and b
    
    Rv must be given as an array of radii"""
    kernel=np.zeros((len(pos_s)))
    for i in set_of_IDs: 
        #Loop that goes through each of the sources being integrated
        kernel[i]=Sampson_grad_Green(a, b, pos_s[i], normal,Rv[i])
    return(kernel)


def kernel_integral_Green_face(pos_s, set_of_IDs, a,b, Rv):
    """Will return the kernel to multiply the array of {q}.
    It integrates the Green's function
    through the Sampson's rule over the line that links a and b
    
    Rv must be given as an array of radii"""
    #pdb.set_trace()
    kernel=np.zeros(len(pos_s))
    for i in set_of_IDs: 
        #Loop that goes through each of the sources being integrated
        kernel[i]=Sampson_Green(a, b, pos_s[i], Rv[i])
    return(kernel)


