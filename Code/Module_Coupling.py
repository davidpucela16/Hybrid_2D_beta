#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:59:14 2021

@author: pdavid

MAIN MODULE FOR THE SOLUTION SPLIT COUPLING MODEL IN 2D!

This is the first coupling module that I manage to succeed with some type of coupling

the coupling with the negihbouring FV works quite well.

The problem arises when coupling two contiguous source blocks. Since there is no 
continuity enforced explicitly the solution does not respec C1 nor C0 continuity.

Furthermore, 
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
import math
import pdb
from scipy import sparse
from scipy.sparse.linalg import spsolve


#from Green import  grad_Green_norm, grad_Green, Green, Sampson_grad_Green, Sampson_Green, block_grad_Green_norm_array,kernel_green_neigh, from_pos_get_green,kernel_integral_grad_green_face,kernel_integral_Green_face
from Green import *
from Small_functions import get_boundary_vector, pos_to_coords, coord_to_pos, get_4_blocks, v_linear_interpolation
from Neighbourhood import get_neighbourhood, get_multiple_neigh, get_uncommon
from Assembly_diffusion import A_assembly
    
import os 
print("MODULE")
print(os.getcwd())


class assemble_SS_2D_FD():
    """Create a local, potential based linear system"""
    def __init__(self, pos_s, Rv, h,L, K_eff, D,directness):   
        """Class that will assemble the localized operator splitting approach.
        Its arguments are:
            - pos_s: an array containing the position of each sources
            - Rv: an array containing the radius of each source
            - h: the size of the cartesian mesh
            - L: the size of the full square domain
            - K_eff: an array with the effective permeability of each source
            - D: the diffusion coefficient in the parenchyma 
            - directness: how many levels of adjacent FV cells are considered for the splitting"""
        x=np.linspace(h/2, L-h/2, int(np.around(L/h)))
        y=x.copy()
        self.x=x
        self.y=y
        self.K_0=K_eff*np.pi*Rv**2
        self.pos_s=pos_s #exact position of each source
        self.L=L
        self.n_sources=self.pos_s.shape[0]
        self.Rv=Rv
        self.h=h
        self.D=D
        self.boundary=get_boundary_vector(len(x), len(y))
        self.directness=directness

        self.no_interpolation=0 #The variable is set to 0 by default so the complex interpolation
        #function will be used for the estimation of the wall concentration for q
    
    def solve_problem(self, BC_types, BC_values, C_v_array):
        """Function that after initialisation solves the entire problem"""
        self.pos_arrays() #Creates the arrays with the geometrical position of the sources 
        self.initialize_matrices() #Creates the matrices (Jacobian, indep term....)
        
        #Assembly of the Laplacian and other arrays for the linear system
        LIN_MAT=self.assembly_sol_split_problem(BC_types, BC_values) 
        self.set_intravascular_terms(C_v_array) #Sets up the intravascular concentration as BC 
        sol=np.linalg.solve(LIN_MAT, -self.H0)
        s_FV=sol[:-self.S].reshape(len(self.x), len(self.y))
        q=sol[-self.S:]
        
        #initial guesses -> solution of the linear system 
        self.s_FV=s_FV
        self.q_linear=q
    
        self.set_phi_bar_linear(C_v_array)
        self.C_v_array=C_v_array
        return(self.q_linear)
    
    def set_phi_bar_linear(self, C_v_array):
        self.phi_bar=C_v_array-self.q_linear/self.K_0
        self.C_v_array=C_v_array
        self.phi_bar2=q_linear/(K0) - np.dot(self.Down, sol)
        
        if np.abs(self.phi_bar-self.phi_bar2)<10**-4:
            print('ERROR ERROR ERROR ERRROR')
            print('ERROR ERROR ERROR ERRROR')
            print('ERROR ERROR ERROR ERRROR')
            
        return(self.phi_bar)
    
    def pos_arrays(self):
        """This function is the pre processing step. It is meant to create the s_blocks
        and uni_s_blocks arrays which will be used extensively throughout. s_blocks represents
        the block where each source is located, uni_s_blocks contains all the source blocks
        in a given order that will be respected throughout the resolution"""
        #pos_s will dictate the ID of the sources by the order they are kept in it!
        source_FV=np.array([]).astype(int)
        uni_s_blocks=np.array([], dtype=int)
        for u in self.pos_s:
            r=np.argmin(np.abs(self.x-u[0]))
            c=np.argmin(np.abs(self.y-u[1]))
            source_FV=np.append(source_FV, c*len(self.x)+r) #block where the source u is located
            if c*len(self.x)+r not in uni_s_blocks:
                uni_s_blocks=np.append(uni_s_blocks, c*len(self.x)+r)
            
        self.FV_DoF=np.arange(len(self.x)*len(self.y))
        self.s_blocks=source_FV #for each entry it shows the block of that source
        self.uni_s_blocks=uni_s_blocks
        
        total_sb=len(np.unique(self.s_blocks)) #total amount of source blocks
        self.total_sb=total_sb
    
        

    def initialize_matrices(self):
        self.A_matrix=np.zeros((len(self.FV_DoF), len(self.FV_DoF)))
        self.b_matrix=np.zeros((len(self.FV_DoF), len(self.s_blocks)))
        self.c_matrix=np.zeros((len(self.s_blocks), len(self.FV_DoF)))
        self.d_matrix=np.zeros((len(self.s_blocks), len(self.s_blocks)))
        self.H0=np.zeros(len(self.s_blocks)+len(self.FV_DoF))
    
    def set_intravascular_terms(self, C_v_array):
        """Be careful, it operates over the class variable H0 which is the independent term
        Must be called only once!!"""
        self.S=len(C_v_array)
        self.H0[-len(C_v_array):]-=C_v_array
        self.C_v_array=C_v_array
    
# =============================================================================
#     def set_Dirichlet(self, values):
#         """Sets the Dirichlet BC translated to Neumann BC on the solution 
#         splitting operator
#         
#         We could adimensionalize all the positions but it is not worth the hassel I think"""
#         north, sout, east, west=self.boundary
#         v_n, v_s, v_e, v_w=values
#         c=0
#         #pdb.set_trace()
#         for b in self.boundary:
#             for k in b:
#                 normal=np.array([[0,1],[0,-1],[1,0],[-1,0]])[c]
# 
#                 pos_k=pos_to_coords(self.x, self.y, k)
#                 pos_bound=pos_k+normal*self.h/2
#                 
#                 perp_dir=np.array([[0,-1],[1,0]]).dot(normal)
#                 
#                 pos_a=pos_k+(perp_dir+normal)*self.h/2
#                 pos_b=pos_k+(-perp_dir+normal)*self.h/2
#                 #pdb.set_trace()
#                 N_k=get_neighbourhood(self.directness, len(self.x), k)
#                 s_IDs=np.arange(len(self.pos_s))[np.in1d(self.s_blocks, N_k)]
# 
#                 #The division by h is because the kernel calculates the integral, what we 
#                 #need is an average value per full cell
#                 kernel=kernel_integral_Green_face(self.pos_s, s_IDs, pos_a,pos_b,self.Rv)/self.h
#                 
#                 self.A_matrix[k,k]-=2*self.D
#                 self.b_matrix[k,:]-=kernel*2*self.D
#                 self.H0[k]+=values[c]*2*self.D
#             
#             c+=1'
# =============================================================================
            
    def set_BCs(self, BC_type, BC_values):
        array_opposite=np.array([1,0,3,2])
        for i in range(4):
        #Goes through each boundary {north, south, east, west} = {0,1,2,3,4}
            c=0
            for k in self.boundary[i,:]:
            #Goes through each of the cells in the boundary i
                normal=np.array([[0,1],[0,-1],[1,0],[-1,0]])[i]
                m=self.boundary[array_opposite[i],c]
                #The division by h is because the kernel calculates the integral, what we 
                #need is an average value per full cell
                r_k_face_kernel, r_k_grad_face_kernel, r_m_face_kernel ,  r_m_grad_face_kernel= self.get_interface_kernels(k,normal,m)
                
                if BC_type[i] == 'Dirichlet':
                    self.A_matrix[k,k]-=2*self.D
                    self.b_matrix[k,:]-=r_k_face_kernel*2*self.D/self.h
                    self.H0[k]+=BC_values[i]*2*self.D
                elif BC_type[i] == 'Neumann':
                    self.b_matrix[k,:]-=r_k_grad_face_kernel*self.D
                    self.H0[k]-=BC_values[i]*self.h
                elif BC_type[i] == 'Periodic':
 
                    jump_r_km = - r_k_grad_face_kernel/2 - r_k_face_kernel/self.h
                    jump_r_mk=r_m_grad_face_kernel/2 + r_m_face_kernel/self.h #The negative sign is because th normal is opposite

                    self.b_matrix[k,:]+=(jump_r_mk + jump_r_km)
                    self.A_matrix[k,k]-=1
                    self.A_matrix[k,m]+=1
                else:
                    print("Error in the assignation of the BC_type")
                
                c+=1
                
    def get_interface_kernels(self, block_ID, normal, neighbour):
        """Complicated function that is meant to ease the coding of the b_matrix
        The jump in flux and concentration across different interfaces is dependent on the 
        average values of the gradient and value of the rapid potential on each side of the 
        interface (side k and side m)
            Returns:
            - the kernel for the Sampson integral over the face of the rapid potential
            - the kernel of the Sampson integral over the face of the normal gradient of the rapid potential
            - The kernel for the jump between neighbours"""
        k=block_ID #the block in questino for reference
        perp_dir=np.array([[0,-1],[1,0]]).dot(normal)
        #First the kernels for r_k
        
        N_k=get_neighbourhood(self.directness, len(self.x), k)
        
        pos_k=pos_to_coords(self.x, self.y, k) #cell center
        pos_a=pos_k+(perp_dir+normal)*self.h/2
        pos_b=pos_k+(-perp_dir+normal)*self.h/2
        
        pos_m=pos_to_coords(self.x, self.y, neighbour)
        pos_a_m=pos_a-normal*self.L
        pos_b_m=pos_b-normal*self.L
        m_neigh=get_neighbourhood(self.directness, len(self.x), neighbour)
        if np.all((pos_m-pos_k)/self.h == normal):
        #This is a real neighbour and they have common sources
            unc_k_m=get_uncommon(N_k, m_neigh)
            unc_m_k=get_uncommon(m_neigh, N_k)
            
        #sources in the neighbourhood of m that are not in the neigh of k 
            Em=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks,unc_m_k)]
            #sources in the neighbourhood of k that are not in the neigh of m
            Ek=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks,unc_k_m)]
    
            #pdb.set_trace()
            r_m_grad_face_kernel=kernel_integral_grad_Green_face(self.pos_s, Em, pos_a,pos_b,normal, self.Rv)
            r_m_face_kernel=kernel_integral_Green_face(self.pos_s, Em, pos_a,pos_b,self.Rv)
            
            r_k_grad_face_kernel=kernel_integral_grad_Green_face(self.pos_s, Ek, pos_a,pos_b,normal, self.Rv)
            r_k_face_kernel=kernel_integral_Green_face(self.pos_s, Ek, pos_a,pos_b,self.Rv)
            
        else:
            #it is a "fake" neighbour caused by the periodic BCs
            #The division by h is because the kernel calculates the integral, what we 
            #need is an average value per full cell
            Em=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks,m_neigh)]
            #sources in the neighbourhood of k that are not in the neigh of m
            Ek=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks,N_k)]
            r_k_face_kernel=kernel_integral_Green_face(self.pos_s, Ek, pos_a,pos_b,self.Rv)
            r_k_grad_face_kernel= kernel_integral_grad_Green_face(self.pos_s, Ek, pos_a,pos_b,normal, self.Rv)
            r_m_grad_face_kernel=kernel_integral_grad_Green_face(self.pos_s, Em, pos_a_m,pos_b_m,normal, self.Rv)
            r_m_face_kernel=kernel_integral_Green_face(self.pos_s, Em, pos_a_m,pos_b_m,self.Rv)
            
        return(r_k_face_kernel, r_k_grad_face_kernel, r_m_face_kernel, r_m_grad_face_kernel)
    
    def assembly_sol_split_problem(self, BC_types, BC_values):
        """Main function that assembles the matrices one by one.
        The only functions called before this one are pos_arrays to get the relevant information
        stored in the very important arrays (such as s_blocks), and initialize matrices.
        
        The arg "no_interpolation" should be introduced to use simple interpolation
        when estimating the wall concentration at each source"""
        #First of all it is needed to remove the source_blocks from the main matrix
        #pdb.set_trace()
        self.A_matrix=A_assembly(len(self.x), len(self.y))/self.D
        #self.set_Dirichlet(values_Dirich)
        self.set_BCs(BC_types, BC_values)
        self.b_matrix=self.assemble_b_matrix_short()/self.D
        self.assemble_c_d_matrix()
        
        Up=np.concatenate((self.A_matrix, self.b_matrix), axis=1)
        Down=np.concatenate((self.c_matrix, self.d_matrix), axis=1)
        
        self.Up=Up
        self.Down=Down
        
        LIN_MAT=np.concatenate((Up,Down), axis=0)
        self.LIN_MAT=LIN_MAT
        return(LIN_MAT)
    

    def assemble_b_matrix_short(self):
        
        for k in self.FV_DoF:
            
            c=0
            neigh=np.array([len(self.x), -len(self.x), 1,-1])
            
            if k in self.boundary[0]:
                neigh=np.delete(neigh, np.where(neigh==len(self.x))[0])
            if k in self.boundary[1]:
                neigh=np.delete(neigh, np.where(neigh==-len(self.x))[0])
            if k in self.boundary[2]:
                neigh=np.delete(neigh,np.where(neigh==1)[0])
            if k in self.boundary[3]:
                neigh=np.delete(neigh, np.where(neigh==-1)[0])
            
            for i in neigh:
                m=k+i
                pos_k=pos_to_coords(self.x, self.y, k)
                pos_m=pos_to_coords(self.x, self.y, m)
                normal=(pos_m-pos_k)/self.h
                r_k_face_kernel, r_k_grad_face_kernel, r_m_face_kernel, r_m_grad_face_kernel=self.get_interface_kernels(k, normal, m)
                
                #pdb.set_trace()
                kernel_m=r_m_face_kernel/self.h + r_m_grad_face_kernel/2 

                kernel_k=r_k_face_kernel/self.h + r_k_grad_face_kernel/2 
      
                self.b_matrix[k,:]+=kernel_m
                self.b_matrix[k,:]-=kernel_k
                
                c+=1
        return(self.b_matrix)

    
    
    def assemble_c_d_matrix(self):
        no_interpolation=self.no_interpolation
        #pdb.set_trace()
        for block_ID in self.uni_s_blocks:
            k=block_ID
            loc_neigh_k=get_neighbourhood(self.directness, len(self.x), block_ID)
            sources_neigh=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks,loc_neigh_k)]
            sources_in=np.where(self.s_blocks==block_ID)[0] #sources that need to be solved in this block
            for i in sources_in:
                #other=np.delete(sources, i)
                other=np.delete(sources_neigh, np.where(sources_neigh==i))
#############################################################################################
#############################################################################################
                if no_interpolation:
                    self.c_matrix[i,k]=1 #regular term 
                    self.d_matrix[i,i]=1/self.K_0[i] 
                    print("old version flux estimation")
                    for j in other:
                        """Goes through each of the other sources (j != i)"""
                        self.d_matrix[i,j]+=Green(self.pos_s[i],self.pos_s[j], self.Rv[j])
                else:
                    #print("new version flux estimation")
                    if (self.pos_s[i,0]>self.x[-1] or self.pos_s[i,0]<self.x[0] or self.pos_s[i,1]>self.y[-1] or self.pos_s[i,1]<self.y[0]):
                        """The condition will be satisfied if the source falls within the dual boundary, that is,
                        it would need to be interpolated with the boundary values of the regular term. That is difficult, 
                        so for these sources that lie closer than h/2 from the boundary, no interpolation is performed"""
                        self.c_matrix[i,k]=1 #regular term 
                        self.d_matrix[i,i]=1/self.K_0[i] 
                        for j in other:
                            """Goes through each of the other sources (j != i)"""
                            self.d_matrix[i,j]+=Green(self.pos_s[i],self.pos_s[j], self.Rv[j])
                            
                    
                    else: 
                        """The source does not belong to the dual boundary, therefore, it can be 
                        interpolated without issues"""
                        blocks=get_4_blocks(self.pos_s[i], self.x, self.y, self.h) #gets the ID of each of the 4 blocks
                        
                        c_kernel=np.zeros(len(self.x)*len(self.y))
                        d_kernel=np.zeros(len(self.pos_s))
                        
                        neigh_base=get_neighbourhood(self.directness, len(self.x), block_ID)
                                #Gets the neighbourhood to which u is calculated 
                        #total_sources=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, neigh_base)]
                        coord_blocks=np.zeros((0,2))
                        for k in blocks:
                            
                            coord_block=pos_to_coords(self.x, self.y, k)
                            coord_blocks=np.vstack((coord_blocks, coord_block))
                        center=np.sum(coord_blocks, axis=0)/4
                        #pdb.set_trace()
                        c_kernel[blocks]=v_linear_interpolation(center, self.pos_s[i], self.h)
                        
                        for k in blocks:
                            neigh_m=get_neighbourhood(self.directness, len(self.x), k)
                            d_kernel+=kernel_green_neigh(coord_block, neigh_m, self.pos_s, self.s_blocks, self.Rv)*c_kernel[k]
                            d_kernel-=kernel_green_neigh(coord_block, neigh_base, self.pos_s, self.s_blocks, self.Rv)*c_kernel[k]
                        self.c_matrix[i,:]=c_kernel #regular term 
                        self.d_matrix[i,i]=1/self.K_0[i]
                        self.d_matrix[i]+=d_kernel
                        for j in other:
                            """Goes through each of the other sources (j != i)"""
                            self.d_matrix[i,j]+=Green(self.pos_s[i],self.pos_s[j], self.Rv[j])/self.D

        return()

        
    def get_corners(self):
        corners=np.array([0,len(self.x)-1,len(self.x)*(len(self.y)-1), len(self.x)*len(self.y)-1])
        self.corners=corners
        return(corners)
    





class non_linear_metab(assemble_SS_2D_FD):
    """Creates a non linear, potential based 
    diffusion 2D problem with circular sources"""
    def __init__(self,pos_s, Rv, h,L, K_eff, D,directness):
        assemble_SS_2D_FD.__init__(self,pos_s, Rv, h,L, K_eff, D,directness)
        self.pos_arrays()
    def solve_linear_prob(self, BC_types, BC_values,C_v_array):
        """Solves the problem without metabolism, necessary to provide an initial guess
        DIRICHLET ARRAY ARE THE VALUES OF THE CONCENTRATION VALUE AT EACH BOUNDARY, THE CODE
        IS NOT YET MADE TO CALCULATE ANYTHING OTHER THAN dirichlet_array=np.zeros(4)"""
        S=len(self.pos_s)
        self.S=S
        K0=self.K_0
        self.C_v_array=C_v_array
        self.initialize_matrices()
        LIN_MAT=self.assembly_sol_split_problem(BC_types, BC_values)
        self.H0[-S:]-=C_v_array
        #t.B[-np.random.randint(0,S,int(S/2))]=0
        sol=np.linalg.solve(LIN_MAT, -self.H0)
        s_FV_linear=sol[:-S].reshape(len(self.x), len(self.y))
        q_linear=sol[-S:]
        
        #initial guesses
        self.s_FV_linear=s_FV_linear
        self.q_linear=q_linear
        self.phi_bar=C_v_array-self.q_linear/self.K_0
        
        self.phi_bar2=q_linear/(K0) - np.dot(self.Down, sol)
        
        if np.any(np.abs(self.phi_bar-self.phi_bar2))<10**-4:
            print('ERROR ERROR ERROR ERRROR')
            print('ERROR ERROR ERROR ERRROR')
            print('ERROR ERROR ERROR ERRROR')
            pdb.set_trace()
        
    def reconstruct_field_kk(self, s_FV, q):
        G_rec=from_pos_get_green(self.x, self.y, self.pos_s, self.Rv ,np.array([0,0]), self.directness, self.s_blocks)
        return((np.dot(G_rec[:,:],q)+np.ndarray.flatten(s_FV)).reshape(len(self.y), len(self.x)))

    def assemble_it_matrices_Sampson(self, s_FV, q):
        """Assembles the important matrices for the iterative problem through
        Sampson integration within each cell. The matrices:
            - Upper Jacobian: [\partial_{u} I_m  \partial_{q} I_m]"""
        phi_0=self.phi_0
        #M=self.M
        w_i=np.array([1,4,1,4,16,4,1,4,1])/36
        corr=np.array([[-1,-1,],
                       [0,-1],
                       [1,-1],
                       [-1,0],
                       [0,0],
                       [1,0],
                       [-1,1],
                       [0,1],
                       [1,1]])*self.h/2
        I_m=np.zeros(s_FV.size)
        rec_sing=np.zeros(s_FV.size) #integral of the rapid term over the cell
        part_Im_s_FV=np.zeros(s_FV.size)  #\frac{\partial I_m}{\partial u}
        part_Im_q=np.zeros((s_FV.size, len(self.pos_s)))
        for k in range(s_FV.size):
            #pdb.set_trace()
            for i in range(len(w_i)):
                
                pos_xy=pos_to_coords(self.x, self.y, k)+corr[i]
                kernel_G=kernel_green_neigh(pos_xy, get_neighbourhood(self.directness, len(self.x), k),
                                            self.pos_s, self.s_blocks, self.Rv)
                SL=np.dot(kernel_G,q)
                rec_sing[k]+=w_i[i]*SL
                
                I_m[k]+=phi_0*w_i[i]/(phi_0+SL+s_FV[k])
                part_Im_s_FV[k]-=phi_0*w_i[i]/(phi_0+SL+s_FV[k])**2
                for l in range(len(self.pos_s)):
                    #part_Im_q[k,l]+=part_Im_u[k]*kernel_G[l]
                    part_Im_q[k,l]-=kernel_G[l]*phi_0*w_i[i]/(phi_0+SL+s_FV[k])**2
    
        self.I_m=I_m
        self.rec_sing=rec_sing #integral of the Single Layer potential over the cell
        self.part_Im_q=part_Im_q
        self.part_Im_s_FV=part_Im_s_FV
        return()
 
    def Full_Newton(self, s_linear,q_linear, rel_error,M, phi_0):
        self.M=M
        stabilization=1
        iterations=0
        if M/self.D>5e-4:
            #For very high consumptions, we set the initial guess to zero concentration in the parenchyma
            self.assemble_it_matrices_Sampson(s_linear, q_linear)
            s_linear=-self.rec_sing
            q_linear=q_linear
            stabilization=2*10**-4/M/self.D/2
            print("stabilization= ", stabilization)
        self.phi_0=phi_0
        rl=np.array([1])
        arr_unk=np.array([np.concatenate((s_linear,q_linear))]) #This is the array where the arrays of u through iterations will be kept
        S=self.S
        while (np.abs(rl[-1])>rel_error and iterations<300):
            self.assemble_it_matrices_Sampson(arr_unk[-1,:-S], arr_unk[-1,-S:])
            #average phi field
            s_field=arr_unk[-1,:-S]
            r_field=self.rec_sing
            phi=s_field+r_field
            if np.any(phi<0):
                s_field[phi<0]=-r_field[phi<0]
                self.assemble_it_matrices_Sampson(s_field, arr_unk[-1,-S:])
                phi=s_field+r_field
            
            metab=self.M*(1-self.I_m)*self.h**2
            metab[phi<0]=0
            part_FV=self.part_Im_s_FV
            part_q=self.part_Im_q
            part_FV=self.part_Im_s_FV*M*self.h**2
            part_q=self.part_Im_q*M*self.h**2
            Jacobian=np.concatenate((np.diag(part_FV)+self.A_matrix, 
                                     part_q+self.b_matrix), axis=1)
            

            Jacobian=np.concatenate((Jacobian, self.Down)) 
            #Compute the new value of u:
            F=np.dot(self.LIN_MAT, arr_unk[-1]) +self.H0 - np.pad(metab, [0,self.S])
            
            inc=np.linalg.solve(Jacobian, -F)
            inc[:-S][(inc[:-S]+phi)<0]=-phi[(inc[:-S]+phi)<0]
            arr_unk=np.concatenate((arr_unk, np.array([arr_unk[-1]+inc*stabilization])))

            rl=np.append(rl,np.sum(np.abs(inc))/len(inc))
            print("residual", np.sum(np.abs(inc))/len(inc))
            iterations+=1
        self.arr_unk_metab=arr_unk
        self.s_FV_metab=self.arr_unk_metab[-1,:-S]
        self.q_metab=self.arr_unk_metab[-1,-S:]
        
        self.residual=np.sum(np.abs(inc))/len(inc)
        if iterations < 300:
            return(self.s_FV_metab, self.q_metab)
        else:
            return(np.zeros(len(self.s_FV_metab)), np.zeros(len(self.q_metab)))
        







