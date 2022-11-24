# -*- coding: utf-8 -*-

import numpy as np 

def get_neighbourhood(directness, xlen, block_ID):
    """Will return the neighbourhood of a given block for a given directness 
    in a mesh made of square cells
    
    It will assume xlen=ylen"""
    dr=directness
    pad_x=np.concatenate((np.zeros((dr))-1,np.arange(xlen), np.zeros((dr))-1))
    pad_y=np.concatenate((np.zeros((dr))-1,np.arange(xlen), np.zeros((dr))-1))
    
    pos_x, pos_y=block_ID%xlen, block_ID//xlen
    
    loc_x=pad_x[pos_x:pos_x+2*dr+1]
    loc_x=loc_x[np.where(loc_x>=0)]
    loc_y=pad_y[pos_y:pos_y+2*dr+1]
    loc_y=loc_y[np.where(loc_y>=0)]
    
    square=np.zeros((len(loc_y), len(loc_x)), dtype=int)
    c=0
    for i in loc_y:
        square[c,:]=loc_x+i*xlen
        c+=1
    #print("the neighbourhood", square)
    return(np.ndarray.flatten(square))


def get_multiple_neigh(directness, xlen, array_of_blocks):
    """This function will call the get_neighbourhood function for multiple blocks to 
    return the ensemble of the neighbourhood for all the blocks"""
    full_neigh=set()
    for i in array_of_blocks:
        full_neigh=full_neigh | set(get_neighbourhood(directness, xlen, i))
    return(np.array(list(full_neigh), dtype=int))



def get_uncommon(k_neigh, n_neigh):
    """returns the cells of the first neighbourhood that has not in common with the
    second neighbourhood"""
    
    neigh_k_unc=k_neigh[np.invert(np.in1d(k_neigh, n_neigh))]
    return(neigh_k_unc)
