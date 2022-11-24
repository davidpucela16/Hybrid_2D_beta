#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 08:14:58 2022

@author: pdavid

TESTING MUDULE

"""

import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos,pos_to_coords,get_MAE, get_MRE, get_position_cartesian_sources
from Reconstruction_extended_space import reconstruction_extended_space

import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
import pandas as pd 
import copy

import pdb

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6,6 ),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

class Testing():
    def __init__(self,pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value):
        """The solution to each case is stored as the solution on the FV grid 
        and an array of the estimation of the vessel tissue exchanges (some type 
        of q array)
        
        It further stores the concentration field on straight vertical and horizontal lines 
        passing through each of the centers of the circular sources"""
        self.h_coarse=L/cells
        self.L=L
        self.ratio=ratio
        self.directness=directness
        self.Rv=Rv
        self.pos_s=pos_s
        self.K0=np.pi*Rv**2*K_eff
        self.D=D
        self.C_v_array=C_v_array
        self.BC_type=BC_type
        self.BC_value=BC_value
        self.cells=cells
        self.K_eff=K_eff
        
        
        #Metabolism Parameters by default
        self.conver_residual=5e-5
        self.stabilization=0.5
        
        #Definition of the Cartesian Grid
        self.x_coarse=np.linspace(self.h_coarse/2, L-self.h_coarse/2, int(np.around(L/self.h_coarse)))
        self.y_coarse=self.x_coarse.copy()
        
        self.x_fine=np.linspace(self.h_coarse/(2*ratio), L-self.h_coarse/(2*ratio), int(np.around(L*ratio/self.h_coarse)))
        self.y_fine=self.x_fine.copy()
        
        self.no_interpolation=0
            
    def Linear_FV_Peaceman(self, Peaceman):
        """Performs the simulation for a refined Peaceman"""
        
        L=self.L
        pos_s=self.pos_s
        cells=self.cells
        C_v_array=self.C_v_array
        Rv=self.Rv
        BC_type=self.BC_type
        BC_value=self.BC_value 
        K_eff=self.K0/(np.pi*self.Rv**2)
        
        FV=FV_validation(L, cells*self.ratio, pos_s, C_v_array, self.D, K_eff, Rv,BC_type, BC_value, Peaceman)
        #####################################
        #####  CORR ARRAY!!
        #####################################
        phi_FV_linear=FV.solve_linear_system()
        q_FV_linear=FV.get_q_linear()
        phi_FV_linear_matrix=phi_FV_linear.reshape(cells*self.ratio, cells*self.ratio)
        
        plt.imshow(phi_FV_linear_matrix, origin='lower')
        plt.colorbar()
        plt.title("FV Peaceman solution, linear system\n mesh:{}x{}".format(self.ratio*cells, self.ratio*cells))
        plt.show()
        
        array_phi_field_x_linear=np.zeros((len(pos_s), len(self.x_fine)))
        array_phi_field_y_linear=np.zeros((len(pos_s), len(self.y_fine)))
        
        #The following chunk of code is to plot the vertical line that goes through 
        #the center of each source. I commented since it takes quite a bit of time to 
        #plot when there are many sources. It is functional on 22nd-Nov-2022
# =============================================================================
#         c=0
#         for i in pos_s:
#             pos=coord_to_pos(FV.x, FV.y, i)
#             
#             plt.plot(self.x_fine, mat_linear[pos//len(FV.x),:], label="FV")
#             plt.xlabel("x $\mu m$")
#             plt.legend()
#             plt.title("Linear Peaceman solution")
#             plt.show()
#             
#             array_phi_field_x_linear[c]=mat_linear[int(pos//len(FV.x)),:]
#             array_phi_field_y_linear[c]=mat_linear[:,int(pos%len(FV.x))]
#             c+=1
# =============================================================================
        self.array_phi_field_x_linear_Peaceman=array_phi_field_x_linear
        self.array_phi_field_y_linear_Peaceman=array_phi_field_y_linear           
        if Peaceman:
            self.q_Peaceman_linear_Peaceman=q_FV_linear
            self.phi_Peaceman_linear_Peaceman=phi_FV_linear+FV.get_corr_array()
        else:
            self.q_FV_linear=q_FV_linear
            self.phi_FV_linear=phi_FV_linear            
        
        return(phi_FV_linear,q_FV_linear)
    
    def Metab_FV_Peaceman(self, M, phi_0, Peaceman):
        """Peaceman=1 if Peaceman coupling model
        Peaceman=0 if no coupling"""
        L=self.L
        pos_s=self.pos_s
        cells=self.cells
        C_v_array=self.C_v_array
        Rv=self.Rv
        K_eff=self.K0/(np.pi*self.Rv**2)
        BC_type=self.BC_type
        BC_value=self.BC_value 
        x_fine, y_fine=self.x_fine, self.y_fine #Values of the original mesh taking into account the ratio
        
        #Standard FV model object:
        FV=FV_validation(L, cells*self.ratio, pos_s, C_v_array, self.D, K_eff, Rv,BC_type, BC_value, Peaceman)
        
        phi_FV_metab=FV.solve_non_linear_system(phi_0,M, self.stabilization) #We solve the non linear model
        phi_FV_metab=(FV.phi_metab[-1]+FV.Corr_array).reshape(cells*self.ratio, cells*self.ratio) #We reverse the Peaceman
        #correction so the phi value represents the average value in each cell 
        q_FV_metab=FV.get_q_metab()
        phi_FV_metab_matrix=phi_FV_metab.reshape(cells*self.ratio, cells*self.ratio)

        plt.imshow(phi_FV_metab_matrix, origin='lower', vmax=np.max(phi_FV_metab))
        plt.title("FV metab reference")
        plt.colorbar(); plt.show()
        
        array_phi_field_x_metab=np.zeros((len(pos_s), len(self.x_fine)))
        array_phi_field_y_metab=np.zeros((len(pos_s), len(self.y_fine)))
        c=0
        
        #The following chunk of code is to plot the vertical line that goes through 
        #the center of each source. I commented since it takes quite a bit of time to 
        #plot when there are many sources. It is functional on 22nd-Nov-2022
# =============================================================================
#         for i in pos_s:
#             pos=coord_to_pos(x_fine, y_fine, i)
#             plt.plot(phi_FV_metab_matrix[pos//len(FV.x),:], label="Peac metab")
#             plt.legend()
#             plt.show()
#             array_phi_field_x_metab[c]=mat_metab[pos//len(FV.x),:]
#             array_phi_field_y_metab[c]=mat_metab[:,int(pos%len(FV.x))]
#             c+=1
# =============================================================================
            
        if Peaceman:
            self.array_phi_field_x_metab_Peaceman=array_phi_field_x_metab
            self.array_phi_field_y_metab_Peaceman=array_phi_field_y_metab
            self.q_Peaceman_metab=q_FV_metab
            self.phi_Peaceman_metab=np.ndarray.flatten(phi_FV_metab)+FV.get_corr_array(1)
            
        else:
            self.array_phi_field_x_metab_noPeaceman=array_phi_field_x_metab
            self.array_phi_field_y_metab_noPeaceman=array_phi_field_y_metab
            self.q_FV_metab=q_FV_metab
            self.phi_FV_metab=phi_FV_metab  
        
        return(phi_FV_metab, q_FV_metab)
    
    def Multi(self, *Metab):
        cells=self.cells
        n=non_linear_metab(self.pos_s, self.Rv, self.h_coarse, self.L, self.K_eff, self.D, self.directness)
        if self.no_interpolation: n.no_interpolation=1
        n.solve_linear_prob(self.BC_type, self.BC_value, self.C_v_array)
        s_Multi_cart_linear=n.s_FV_linear
        q_Multi_linear=n.q_linear
        self.q_Multi_linear=q_Multi_linear
        self.s_Multi_cart_linear=s_Multi_cart_linear
        self.phi_bar=n.phi_bar
        self.phi_bar2=n.phi_bar2
        
        self.s_blocks=n.s_blocks
        n.phi_0, n.M=1,1
# =============================================================================
#         
#         n.assemble_it_matrices_Sampson(np.ndarray.flatten(s_Multi_cart_coarse_linear), q_Multi_linear)
#         plt.imshow(n.rec_sing.reshape(cells,cells)+s_Multi_cart_coarse_linear, origin='lower', extent=[0,self.L, 0, self.L])
#         plt.title("Average value reconstruction Multi model")
#         plt.colorbar(); plt.show()
#         
#         #self.Multi_linear_object.rec_sing for the potentials averaged per FV cell
# =============================================================================
        self.Multi_linear_object=copy.deepcopy(n)
        #self.Multi_linear_object.rec_sing for the potentials averaged per FV cell
        if Metab:
            M, phi_0=Metab
            n.Full_Newton(np.ndarray.flatten(s_Multi_cart_linear) , np.ndarray.flatten(n.q_linear), 
                          self.conver_residual, M, phi_0)
            
            self.Multi_metab_object=copy.deepcopy(n)
            s_Multi_cart_metab=n.s_FV_metab
            q_Multi_metab=n.q_metab
            
            n.assemble_it_matrices_Sampson(n.s_FV_metab, n.q_metab)
            plt.imshow((n.rec_sing+s_Multi_cart_metab).reshape(cells,cells), origin='lower', extent=[0,self.L, 0, self.L])
            plt.title("Average value reconstruction Multi model Metabolism")
            plt.colorbar(); plt.show()

            self.q_Multi_metab=q_Multi_metab
            self.s_Multi_cart_metab=s_Multi_cart_metab
            self.residual=n.residual
            return(s_Multi_cart_metab,q_Multi_metab)
        else:
            return(s_Multi_cart_linear, q_Multi_linear)
    
    def Reconstruct_Multi(self, non_linear, plot_sources,*FEM_args):
        """If non_linear the reconstruction will be made on the latest non-linear
        simulation (for the arrays self.Multi_q_metab, and self.s_Multi_cart_coarse_metab)
        
        Inside FEM_args there are the FEM_x, FEM_y, arrays where to reconstruct the 
        concentration field
        
        IMPORTANT to have provided the proper value of the ratio"""
        if non_linear:
            obj=self.Multi_metab_object
            s_FV=obj.s_FV_metab
            q=obj.q_metab
        else:
            obj=self.Multi_linear_object
            s_FV=np.ndarray.flatten(obj.s_FV_linear)
            q=obj.q_linear
        
        if not FEM_args: #Cartesian reconstruction:
            print("Have you updated the value of the ratio?? \n right now -> ratio={}".format(self.ratio))
            a=reconstruction_sans_flux(np.concatenate((s_FV, q)),obj,obj.L,self.ratio,obj.directness)
            a.reconstruction()   
            a.reconstruction_boundaries_short(self.BC_type, self.BC_value)
            a.rec_corners()
            plt.imshow(a.rec_final, origin='lower', vmax=np.max(a.rec_final))
            plt.title("bilinear reconstruction \n coupling model Metabolism")
            plt.colorbar(); plt.show()
            
            self.Multi_rec=a.rec_final
            
            
            toreturn=a.rec_final, a.rec_potentials, a.rec_s_FV
        
        
        if FEM_args:
            FEM_x=FEM_args[0]
            FEM_y=FEM_args[1]
            b=reconstruction_extended_space(self.pos_s, self.Rv, self.h_coarse, self.L, 
                                            self.K_eff, self.D, self.directness)
            b.s_FV=s_FV
            b.q=q
            b.set_up_manual_reconstruction_space(FEM_x, FEM_y)
            b.full_rec(self.C_v_array, self.BC_value, self.BC_type)
            if plot_sources:
                plt.tricontourf(b.FEM_x, b.FEM_y, b.s, levels=100)
                plt.colorbar()
                plt.title("s FEM rec")    
                plt.show()
                plt.tricontourf(b.FEM_x, b.FEM_y, b.SL, levels=100)
                plt.colorbar()
                plt.title("SL FEM rec")
                plt.show()
                plt.tricontourf(b.FEM_x, b.FEM_y, b.DL, levels=100)
                plt.colorbar()
                plt.title("DL FEM rec")
                plt.show()
                plt.tricontourf(b.FEM_x, b.FEM_y, b.s+ b.SL+b.DL, levels=100)
                plt.colorbar()
                plt.title("Full FEM rec")
                plt.show()    
            
            self.SL=b.SL
            self.DL=b.DL
            self.s=b.s
            
            #We return the phi-field, the single layer field and the smooth field
            toreturn=b.s+ b.SL+b.DL, b.SL, b.s
        
        
        if plot_sources:
            array_phi_field_x=np.zeros((len(self.pos_s), len(self.x_fine)))
            array_phi_field_y=np.zeros((len(self.pos_s), len(self.y_fine)))   
            c=0
            for i in self.pos_s:
                r=reconstruction_extended_space(self.pos_s, self.Rv, self.h_coarse, self.L, 
                self.K_eff, self.D, self.directness)
                r.s_FV=s_FV
                r.q=q
                r.set_up_manual_reconstruction_space(i[0]+np.zeros(len(self.x_fine)), self.y_fine)
                r.full_rec(self.C_v_array, self.BC_value, self.BC_type)
                array_phi_field_y[c]=r.s+r.SL+r.DL
                
                r.set_up_manual_reconstruction_space(self.x_fine, i[1]+np.zeros(len(self.y_fine)))
                r.full_rec(self.C_v_array, self.BC_value, self.BC_type)
                array_phi_field_x[c]=r.s+r.SL+r.DL
                
                plt.plot(self.x_fine, array_phi_field_x[c], label='Multi_x')
                plt.legend()
                plt.title("Plot through source center")
                plt.xlabel("x")
                plt.ylabel("$\phi$")
                plt.show()
                
                plt.plot(self.y_fine, array_phi_field_y[c], label='Multi_y')
                plt.legend()
                plt.title("Plot through source center")
                plt.xlabel("y")
                plt.ylabel("$\phi$")
                plt.show()

            c+=1
            self.array_phi_field_x_Multi=array_phi_field_x
            self.array_phi_field_y_Multi=array_phi_field_y
        #We return the phi-field, the single layer field and the smooth field
        return(toreturn)

def extract_COMSOL_data(directory_COMSOL, args):
    
    """args corresponds to which files need to be extracted"""
    toreturn=[]
    if args[0]:
        #Vessel_tissue exchanges reference    
        q_file=directory_COMSOL+ '/q.txt'
        q=np.array(pd.read_fwf(q_file, infer_rows=500).columns.astype(float))
        toreturn.append(q)
    
    if args[1]:
        #Concentration field data for the linear problem
        field_file=directory_COMSOL + '/contour.txt'
        df=pd.read_fwf(field_file, infer_nrows=500)
        ref_data=np.array(df).T #reference 2D data from COMSOL
        FEM_x=ref_data[0]*10**6 #in micrometers
        FEM_y=ref_data[1]*10**6
        FEM_phi=ref_data[2]
        toreturn.append(FEM_phi)
        toreturn.append(FEM_x)
        toreturn.append(FEM_y)
        
        
    if args[2]:
      #Plots of the concentration field along a horizontal and vertial lines passing through the center of the source
      x_file=directory_COMSOL + "/plot_x.txt"
      y_file=directory_COMSOL+ '/plot_y.txt'
      
      FEM_x_1D=np.array(pd.read_fwf(x_file, infer_rows=500)).T[1]
      x_1D=np.array(pd.read_fwf(x_file, infer_rows=500)).T[0]*10**6
      FEM_y_1D=np.array(pd.read_fwf(y_file, infer_rows=500)).T[1]
      y_1D=np.array(pd.read_fwf(y_file, infer_rows=500)).T[0]*10**6
      
      toreturn.append(FEM_x_1D)
      toreturn.append(FEM_y_1D)
      toreturn.append(x_1D)
      toreturn.append(y_1D)
    
    return(toreturn)

def FEM_to_Cartesian(FEM_x, FEM_y, FEM_phi, x_c, y_c):
    phi_Cart=np.zeros((len(y_c), len(x_c)))
    for i in range(len(y_c)):
        for j in range(len(x_c)):
            dist=(FEM_x-x_c[j])**2+(FEM_y-y_c[i])**2
            phi_Cart[i,j]=FEM_phi[np.argmin(dist)]
    return(phi_Cart)            

import os
import csv


def write_parameters_COMSOL(pos_s, L, alpha, K_0, M):
    """Writes the parameter file for COMSOL"""
    rows=[["L", L], ["alpha", alpha],["R","L/alpha"],["K_com","K_0/(2*pi*R)"],["M",M],["phi_0",0.4]]
    
    for i in range(len(pos_s)):
        rows.append(["x_{}".format(i), np.around(pos_s[i,0], decimals=4)])
        rows.append(["y_{}".format(i), np.around(pos_s[i,1], decimals=4)])
    with open('Parameters.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for i in rows:
            writer.writerow(i)
            

def save_csv(path, name_columns, data_phi):
    b=pd.DataFrame(data_phi.T)
    b.columns=name_columns
    b.to_csv(path, sep=',', index=None)
    return(b)

def save_var_name_csv(directory, variable):
    variable_name= f'{variable=}'.split('=')[0] 
    path=directory +'/' + variable_name + '.csv'
    save_csv(path, [variable_name], variable)
    return(path)
    
def load_var_name(directory, variable):
    variable_name= f'{variable=}'.split('=')[0] 
    path=directory +'/' + variable_name + '.csv'
    return(pd.read_csv(path).to_numpy())