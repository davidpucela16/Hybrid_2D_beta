#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 19:05:22 2022

@author: pdavid
"""
import numpy as np
from Green import Green, kernel_green_neigh
from Module_Coupling import assemble_SS_2D_FD, get_neighbourhood, get_boundary_vector, pos_to_coords, coord_to_pos
from Reconstruction_functions import green_to_block,NN_interpolation,get_boundar_adj_blocks,single_value_bilinear_interpolation, get_unk_same_reference, bilinear_interpolation, get_sub_x_y, modify_matrix, set_boundary_values, rotate_180, rotate_clockwise, rotate_counterclock, reduced_half_direction, separate_unk
from Neighbourhood import get_multiple_neigh, get_uncommon
from Small_functions import get_4_blocks
import pdb
class reconstruction_extended_space(assemble_SS_2D_FD):
    def __init__(self,pos_s, Rv, h,L, K_eff, D,directness):
        assemble_SS_2D_FD.__init__(self,pos_s, Rv, h,L, K_eff, D,directness)
        assemble_SS_2D_FD.pos_arrays(self)
        
        
    def solve_linear_prob(self, BC_type, BC_value,C_v_array):
        """Solves the problem without metabolism, necessary to provide an initial guess
        DIRICHLET ARRAY ARE THE VALUES OF THE CONCENTRATION VALUE AT EACH BOUNDARY, THE CODE
        IS NOT YET MADE TO CALCULATE ANYTHING OTHER THAN dirichlet_array=np.zeros(4)"""
        S=len(self.pos_s)
        self.C_v_array=C_v_array
        self.S=S
        K0=self.K_0
        self.pos_arrays()
        self.initialize_matrices()
        LIN_MAT=self.assembly_sol_split_problem(BC_type, BC_value)
        self.H0[-S:]-=C_v_array
        #t.B[-np.random.randint(0,S,int(S/2))]=0
        sol=np.linalg.solve(LIN_MAT, -self.H0)
        s_FV=sol[:-S].reshape(len(self.x), len(self.y))
        q=sol[-S:]
        
        #initial guesses
        self.s_FV=np.ndarray.flatten(s_FV)
        self.q=q
    
        
    def set_up_manual_reconstruction_space(self, FEM_x, FEM_y):
        
        self.FEM_x=FEM_x
        self.FEM_y=FEM_y
        self.rec_final=np.zeros(len(FEM_x))
        self.rec_s_FV=np.zeros(len(FEM_x))
        self.rec_potentials=np.zeros(len(FEM_x))
        #Get the boundary nodes:
        south=np.arange(len(FEM_y))[self.FEM_y<self.y[0]]
        north=np.arange(len(FEM_y))[self.FEM_y>self.y[-1]]
        east=np.arange(len(FEM_x))[self.FEM_x>self.x[-1]]
        west=np.arange(len(FEM_x))[self.FEM_x<self.x[0]]
        boundaries=np.concatenate((north, south, east, west))
                
        self.up_right=np.arange(len(FEM_x))[(self.FEM_y>self.y[-1]) & (self.FEM_x>self.x[-1])]
        self.up_left=np.arange(len(FEM_x))[(self.FEM_y>self.y[-1]) & (self.FEM_x<self.x[0])]
        self.down_right=np.arange(len(FEM_x))[(self.FEM_y<self.y[0]) & (self.FEM_x>self.x[-1])]
        self.down_left=np.arange(len(FEM_x))[(self.FEM_y<self.y[0]) & (self.FEM_x<self.x[0])]

        self.FEM_corners=np.concatenate([self.down_left, self.up_left, self.down_right, self.up_right])
        self.south=np.delete(south, np.arange(len(south))[np.in1d(south, self.FEM_corners)])
        self.north=np.delete(north, np.arange(len(north))[np.in1d(north, self.FEM_corners)])
        self.east=np.delete(east, np.arange(len(east))[np.in1d(east, self.FEM_corners)])
        self.west=np.delete(west, np.arange(len(west))[np.in1d(west, self.FEM_corners)])
        
        self.inner=np.delete(np.arange(len(self.FEM_x)), boundaries)
        self.boundaries=np.concatenate((self.north, self.south, self.east, self.west))
        
        self.dual_x=np.arange(0, self.L+0.01*self.h, self.h)
        self.dual_y=np.arange(0, self.L+0.01*self.h, self.h)
        return()
        
    def full_rec(self, C_v_array, BC_value, BC_type):
        self.C_v_array=C_v_array
        self.reconstruction_manual(C_v_array)
        self.reconstruction_boundaries(BC_value, BC_type)
        return()
    
    def reconstruction_manual(self, C_v_array):
        x,y=self.x, self.y
        rec_s=np.zeros(len(self.FEM_x))
        rec_SL=np.zeros(len(self.FEM_x))
        rec_DL=np.zeros(len(self.FEM_x))
        
        for k in self.inner:
            #pdb.set_trace()
            node_pos=np.array([self.FEM_x[k], self.FEM_y[k]])
            blocks=get_4_blocks(node_pos, self.x, self.y, self.h) #gets the ID of each of the 4 blocks 
            corner_values=get_unk_same_reference(blocks, self.directness, len(self.x), 
                                   self.s_blocks, self.uni_s_blocks, np.ndarray.flatten(self.s_FV), 
                                   self.q, self.pos_s, self.Rv, x,y)
            
            ens_neigh=get_multiple_neigh(self.directness, len(self.x), blocks)
            total_sources=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, ens_neigh)]
            coord_vert=np.array([pos_to_coords(x,y,blocks[0]),pos_to_coords(x,y,blocks[1]),pos_to_coords(x,y,blocks[2]),pos_to_coords(x,y,blocks[3])])
            value=single_value_bilinear_interpolation(np.array([self.FEM_x[k], self.FEM_y[k]]), coord_vert, corner_values)
            rec_SL[k]=kernel_green_neigh(node_pos, ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.q)
            rec_DL[k]=np.sum(C_v_array-self.q/self.K_0)
            rec_s[k]=value-rec_DL[k]
        
        self.SL=rec_SL
        self.s=rec_s
        self.DL=rec_DL #I call it DL but in reality is the value of the phi_bar that appear on the neighbourhood
        
        return()
            

    def reconstruction_boundaries(self, BC_value, BC_type):
        boundaries=self.boundaries
        q=self.q
        s_FV=np.ndarray.flatten(self.s_FV) #Values of the local slow term in each FV cell
        s_blocks=self.s_blocks
        pos_s=self.pos_s
        Rv=self.Rv
        FEM_x,FEM_y=self.FEM_x, self.FEM_y
        h=self.h
        
        #Get the boundary values in the DUAL mesh
        self.boundary_values=set_boundary_values(BC_type, BC_value, s_FV,
                                         q, self.x, self.y, 
                                         get_boundary_vector(len(self.x), len(self.y)),self.h,self.D, self) 
        Bn, Bs, Be, Bw=self.boundary_values
        
            
        for b in self.boundaries:
            #Get the exact node position (postition of the value to estimate)
            node_pos=np.array([self.FEM_x[b], self.FEM_y[b]])
            
            #Let's figure out in which boundary it lies
            if b in self.north: c=0
            elif b in self.south: c=1
            elif b in self.east: c=2
            elif b in self.west: c=3
            #Figure out directions:
            normal=np.array([[0,1],[0,-1],[1,0],[-1,0]])[c]
            tau=np.array([[0,1],[-1,0]]).dot(normal)
            #the unknowns (orig. coarse mesh) that are included here
            blocks=get_boundar_adj_blocks(self.x, self.y, node_pos)
            blocks=np.sort(blocks)
            #The following chunck of code is to calculate coord_vert (the coordinates)
            #of the vertices of the dual_volume 
            #The following calculates the center of the dual "boundary element" which comprises a quarter of each 
            #of the elements it touches. Furthermore, since it is a dual element, it is half of the size of 
            dual_center=(pos_to_coords(self.x, self.y, blocks[0])+pos_to_coords(self.x, self.y, blocks[1]))/2+normal*self.h/4
            
            #The following locates the coordinates of the four corners that make up the dual boundary volume
            coord_vert=np.array([[-tau/2-normal/4,-tau/2+normal/4, +tau/2-normal/4, tau/2+normal/4],
                                 [tau/2+normal/4, +tau/2-normal/4, -tau/2+normal/4,-tau/2-normal/4],
                                 [+tau/2-normal/4, -tau/2-normal/4, tau/2 +normal/4, -tau/2+normal/4],
                                 [-tau/2+normal/4, tau/2+normal/4, tau/2-normal/4, +tau/2-normal/4]])[c]*self.h
            
            coord_vert+=np.array([np.zeros(4)+dual_center[0], np.zeros(4)+dual_center[1]]).T
            
            
            #Extended neighbourhood of the dual_node            
            ens_neigh=get_multiple_neigh(self.directness, len(self.x), blocks)
            #Get the slow term in the same neighbourhood reference
            ue=get_unk_same_reference(blocks, self.directness, len(self.x), self.s_blocks, 
                                                self.uni_s_blocks, np.ndarray.flatten(self.s_FV), q, self.pos_s, Rv, self.x, self.y)
            
            #Position of each block (fist, second or d,e) within the boundary array of values
            d=np.where(self.boundary[c]==blocks[0])[0][0]
            e=np.where(self.boundary[c]==blocks[1])[0][0]
            unc_d_e=get_uncommon(get_neighbourhood(self.directness, len(self.x), blocks[0]), get_neighbourhood(self.directness, len(self.x), blocks[1]))
            unc_e_d=get_uncommon(get_neighbourhood(self.directness, len(self.x), blocks[1]), get_neighbourhood(self.directness, len(self.x), blocks[0]))

            
            NE=np.array([unc_d_e,unc_e_d], dtype=object)
            

            #set the boundary vertices on the same renference:

                
            if c==0: #north boundary
                if BC_type[c]=="Dirichlet":
                    V_d=BC_value[c] - kernel_green_neigh(coord_vert[1], get_neighbourhood(self.directness, len(self.x), blocks[0]), self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                    V_e=BC_value[c] - kernel_green_neigh(coord_vert[3], get_neighbourhood(self.directness, len(self.x), blocks[1]), self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                else:
                    V_d,V_e=self.boundary_values[c,d], self.boundary_values[c,e]
                corner_values=np.array([ue[0],V_d-kernel_green_neigh(coord_vert[1], unc_e_d, self.pos_s, self.s_blocks, self.Rv).dot(self.q), 
                                     ue[1], V_e-kernel_green_neigh(coord_vert[3], unc_d_e, self.pos_s, self.s_blocks, self.Rv).dot(self.q)], dtype=float)
            if c==1: #south boundary
                if BC_type[c]=="Dirichlet":
                    V_d=BC_value[c] - kernel_green_neigh(coord_vert[0], get_neighbourhood(self.directness, len(self.x), blocks[0]), self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                    V_e=BC_value[c] - kernel_green_neigh(coord_vert[2], get_neighbourhood(self.directness, len(self.x), blocks[1]), self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                else:
                    V_d,V_e=self.boundary_values[c,d], self.boundary_values[c,e]
                corner_values=np.array([V_d-kernel_green_neigh(coord_vert[0], unc_e_d, self.pos_s, self.s_blocks, self.Rv).dot(self.q), ue[0], 
                                     V_e-kernel_green_neigh(coord_vert[2], unc_d_e, self.pos_s, self.s_blocks, self.Rv).dot(self.q), ue[1]], dtype=float)
            if c==2: #east boundary
                if BC_type[c]=="Dirichlet":
                    V_d=BC_value[c] - kernel_green_neigh(coord_vert[2], get_neighbourhood(self.directness, len(self.x), blocks[0]), self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                    V_e=BC_value[c] - kernel_green_neigh(coord_vert[3], get_neighbourhood(self.directness, len(self.x), blocks[1]), self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                else:
                    V_d,V_e=self.boundary_values[c,d], self.boundary_values[c,e]
                corner_values=np.array([ue[0], ue[1], V_d-kernel_green_neigh(coord_vert[2], unc_e_d, self.pos_s, self.s_blocks, self.Rv).dot(self.q),
                                     V_e-kernel_green_neigh(coord_vert[3], unc_d_e, self.pos_s, self.s_blocks, self.Rv).dot(self.q)], dtype=float)
            if c==3: #west boundary
                if BC_type[c]=="Dirichlet":
                    V_d=BC_value[c] - kernel_green_neigh(coord_vert[1], get_neighbourhood(self.directness, len(self.x), blocks[0]), self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                    V_e=BC_value[c] - kernel_green_neigh(coord_vert[1], get_neighbourhood(self.directness, len(self.x), blocks[1]), self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                else:
                    V_d,V_e=self.boundary_values[c,d], self.boundary_values[c,e]
                corner_values=np.array([V_d-kernel_green_neigh(coord_vert[0], unc_e_d, self.pos_s, self.s_blocks, self.Rv).dot(self.q), 
                                     V_e-kernel_green_neigh(coord_vert[1], unc_d_e, self.pos_s, self.s_blocks, self.Rv).dot(self.q), ue[0], ue[1]], dtype=float)
                

            value=single_value_bilinear_interpolation(node_pos, coord_vert, corner_values)
            
            total_sources=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, ens_neigh)]
            self.SL[b]=kernel_green_neigh(node_pos, ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.q)
            self.DL[b]=np.sum(self.C_v_array-self.q/self.K_0)
            self.s[b]=value-self.DL[b]
        
        v=self.get_corners()
        v[1]=self.get_corners()[2]
        v[2]=self.get_corners()[1]
        for k in self.FEM_corners:
            #single block:
# =============================================================================
#             if k in self.down_left: d=0
#             elif k in self.up_left: d=1
#             elif k in self.down_right: d=2
#             elif k in self.up_right: d=3
# =============================================================================
            node_pos=np.array([self.FEM_x[k], self.FEM_y[k]])
            block=get_boundar_adj_blocks(self.x, self.y, node_pos)[0]
            
            ens_neigh=get_neighbourhood(self.directness, len(self.x), block)
            w=int(np.arange(4)[v==block])
            L=self.L
            coord_vert=np.array([[[0,0],[0,h/2],[h/2,0],[h/2,h/2]],
                                 [[0,L-h/2],[0,L],[h/2,L-h/2],[h/2,L]],
                                 [[L-h/2,0],[L-h/2,h/2],[L,0],[L,h/2]],
                                 [[L-h/2,L-h/2],[L-h/2,L],[L,L-h/2],[L,L]]])[w]
            
            total_sources=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, ens_neigh)]
            
            #Let's figure out which corner they belong to 
            #The coordinates of the corner blocks (in coarse orig mesh)
            coord_corners=pos_to_coords(self.x, self.y, self.corners).T
            dist_corners=np.zeros(0) #array to store the distance to each corner block
            for i in coord_corners:
                dist_corners=np.append(dist_corners,np.linalg.norm(i-node_pos))
            #set the boundary vertices on the same reference, for simplicity the corners I will interpolate the concentration directly:
            
                
                
            if np.argmin(dist_corners)==0: #South-West 
                #if any of the two boundaries is Dirichlet we fix the value of the corner:
                if BC_type[3]=="Dirichlet":
                    V_corner=BC_value[3]-kernel_green_neigh(coord_vert[0], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                elif BC_type[1]=="Dirichlet":
                    V_corner=BC_value[1]-kernel_green_neigh(coord_vert[0], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                else:
                    V_corner=(Bs[0]+Bw[0])/2
                corner_values=np.array([V_corner, Bw[0], Bs[0], s_FV[block]], dtype=float)
            if np.argmin(dist_corners)==1: #South-East
            #if any of the two boundaries is Dirichlet we fix the value of the corner:
                if BC_type[2]=="Dirichlet":
                    V_corner=BC_value[2]-kernel_green_neigh(coord_vert[2], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                elif BC_type[1]=="Dirichlet":
                    V_corner=BC_value[1]-kernel_green_neigh(coord_vert[2], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                else:
                    V_corner=(Bs[-1]+Be[0])/2
                corner_values=np.array([Bs[-1], s_FV[block], V_corner, Be[0]], dtype=float)
                
            if np.argmin(dist_corners)==2: #North-West
            #if any of the two boundaries is Dirichlet we fix the value of the corner:
                if BC_type[3]=="Dirichlet":
                    V_corner=BC_value[3]-kernel_green_neigh(coord_vert[1], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                elif BC_type[0]=="Dirichlet":
                    V_corner=BC_value[0]-kernel_green_neigh(coord_vert[1], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                else:
                    V_corner=(Bw[-1]+Bn[0])/2
                corner_values=np.array([Bw[-1], V_corner, s_FV[block], Bn[0]], dtype=float)
            if np.argmin(dist_corners)==3: #North-East
                #if any of the two boundaries is Dirichlet we fix the value of the corner:
                if BC_type[2]=="Dirichlet":
                    V_corner=BC_value[2]-kernel_green_neigh(coord_vert[3], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                elif BC_type[0]=="Dirichlet":
                    V_corner=BC_value[0]-kernel_green_neigh(coord_vert[3], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.q)
                else:
                    V_corner=(Bn[-1]+Be[-1])/2
                corner_values=np.array([s_FV[block], Bn[-1], Be[-1], V_corner], dtype=float)
                
            
            value=single_value_bilinear_interpolation(node_pos, coord_vert, corner_values)
            
            
            
            self.SL[k]=kernel_green_neigh(np.array([self.FEM_x[k], self.FEM_y[k]]), ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.q)
            self.DL[k]=np.sum(self.C_v_array-self.q/self.K_0)
            self.s[k]=value-self.DL[k]
        return()