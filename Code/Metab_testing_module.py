#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:15:43 2022

@author: pdavid
"""

import numpy as np 
import matplotlib.pyplot as plt 
import random 
import os 
from Testing import Testing
from Small_functions import plot_sketch
from Reconstruction_extended_space import reconstruction_extended_space

def position_sources(dens, L, cyl_rad):
    """dens -> density in source/square milimeter
       L -> side length of the domain
       cyl_rad -> radius of the capillary free region
       """
    pos_s=np.zeros((0,2))
    elem_square=1/(dens*1e-6)
    cells=np.around(L/np.sqrt(elem_square)).astype(int)   
    h=L/cells
    grid_x=np.linspace(h/2, L-h/2,cells) 
    grid_y=grid_x
    
    center=np.array([L/2, L/2])
    
    for i in grid_x: 
        for j in grid_y:
            temp_s=(np.random.rand(1,2)-1/2)*h*0.8+np.array([[i,j]])
            if np.linalg.norm(temp_s-center) > cyl_rad:
                pos_s=np.concatenate((pos_s, temp_s), axis=0)
                
    return(pos_s)

def metab_simulation(mean, std_dev, simulations, density, L,  cyl_rad, R_art, R_cap, K_eff, directness, CMRO2_max, phi_0):
    avg_phi_array=np.zeros((simulations,100))
    theta_arr=np.linspace(0,2*np.pi,20)
    
    counter=0
    for k in range(simulations):
        
        pos_s=np.array([[0.5,0.5]])*L
        pos_s=np.concatenate((pos_s, position_sources(density, L, L/4)), axis=0)
        S=len(pos_s)
        C_v_array=np.array([1])
        Rv=np.array([R_art])
        for i in range(S-1):
            C_v_array=np.append(C_v_array, random.gauss(mean,std_dev))
            Rv=np.append(Rv,R_cap)
        if np.any(pos_s>L) or np.any(pos_s<0): print("ERROR IN THE POSITIONING")
        
        cells=12
        D=1
        ratio=10
        BC_type=np.array(['Periodic','Periodic','Periodic','Periodic'])
        BC_value=np.zeros(4)
        h_coarse=L/cells
        x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, int(np.around(L/h_coarse)))
        y_coarse=x_coarse
        t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)

        t.stabilization=0.5
        print(t.stabilization)
        
        plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, os.getcwd())
        C_v_array[C_v_array>1]=1
        C_v_array[C_v_array<0]=0

        Multi_FV_metab, Multi_q_metab=t.Multi(CMRO2_max,phi_0)
        
        if np.sum(Multi_q_metab)>0:
            counter+=1
            REC_phi_array=np.zeros((0,100))
            REC_x_array=np.zeros((0,100))
            REC_y_array=np.zeros((0,100))
            for i in theta_arr:
                REC_x=np.linspace(0,L*np.cos(i)/2,100)+L/2
                REC_y=np.linspace(0,L*np.sin(i)/2,100)+L/2
                r=reconstruction_extended_space(pos_s, Rv, h_coarse, L,K_eff, D, directness)
                r.s_FV=Multi_FV_metab
                r.q=Multi_q_metab
                r.set_up_manual_reconstruction_space(REC_x,REC_y)
                r.full_rec(C_v_array, BC_value, BC_type)
                REC_phi=r.s+r.SL+r.DL
    
                REC_x_array=np.concatenate((REC_x_array, [REC_x]), axis=0)
                REC_y_array=np.concatenate((REC_y_array, [REC_y]), axis=0)
                REC_phi_array=np.concatenate((REC_phi_array, [REC_phi]), axis=0)
            
            avg_phi_REC=np.sum(REC_phi_array, axis=0)/20
            avg_phi_array[k]=avg_phi_REC
            plt.plot(np.linspace(0,L/2,100),avg_phi_REC)
            plt.xlabel('$\   m$')
            plt.title("average of the average")
            plt.show()    
        
    return(np.sum(avg_phi_array, axis=0)/(counter))

def get_met_plateau(b, L,cap_free_length):
    pos_0=np.around(len(b)*L/cap_free_length)
    return(np.sum(b[pos_0:])/(len(b)-pos_0))