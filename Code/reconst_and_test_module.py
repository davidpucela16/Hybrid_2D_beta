
import numpy as np
from Green import Green, kernel_green_neigh
from Module_Coupling import assemble_SS_2D_FD, get_neighbourhood, get_boundary_vector, pos_to_coords, coord_to_pos
from Reconstruction_functions import green_to_block,NN_interpolation,get_boundar_adj_blocks,single_value_bilinear_interpolation, get_unk_same_reference, bilinear_interpolation, get_sub_x_y, modify_matrix, set_boundary_values, rotate_180, rotate_clockwise, rotate_counterclock, reduced_half_direction, separate_unk
from Neighbourhood import get_multiple_neigh, get_uncommon
from Small_functions import get_4_blocks

import pdb

class real_NN_rec():
    def __init__(self,x, y, phi_FV, pos_s, s_blocks, phi_q, ratio, h, directness, Rv, K_0, C_v_array):
        self.x=x
        self.y=y
        self.directness=directness
        self.phi_FV=np.ndarray.flatten(phi_FV)
        self.phi_q=phi_q
        self.s_blocks=s_blocks
        self.Rv=Rv
        self.pos_s=pos_s
        r_h=h/ratio
        r_x=np.linspace(r_h/2, x[-1]+h/2-r_h/2, len(x)*ratio)
        r_y=np.linspace(r_h/2, y[-1]+h/2-r_h/2, len(y)*ratio)
        
        self.r_x=r_x
        self.r_y=r_y
        
        rec=np.zeros((len(r_y), len(r_x)))
        self.rec_plus_phibar=rec.copy()
        for i in range(len(r_y)):
            for j in range(len(r_x)):
                block=self.get_block(np.array([r_x[j],r_y[i]]))
                rec[i,j]=self.phi_FV[block]
                
                neigh=get_neighbourhood(directness, len(x), block)
                self.rec_plus_phibar[i,j]=rec[i,j]
                total_sources=np.in1d(s_blocks, neigh)
                constant=0
                for jj in np.arange(len(pos_s))[total_sources]:
                    constant+=C_v_array[jj] - phi_q[jj]/K_0[jj]
                self.rec_plus_phibar[i,j]-=constant
        self.rec=rec
        
        
        
    def get_block(self,pos):
        row=np.argmin(np.abs(self.y-pos[1]))
        col=np.argmin(np.abs(self.x-pos[0]))
        return(row*len(self.x)+col)
    
    def add_singular(self):
        #THIS FUNCTION DEFINETELY DOES NOT WORK AT THE MOMENT
        print("THIS FUNCTION DEFINETELY DOES NOT WORK AT THE MOMENT")
        directness=self.directness
        #pdb.set_trace()
        Rv=self.Rv
        r_x=self.r_x
        r_y=self.r_y
        rec=np.zeros((len(r_y), len(r_x)))
        for i in range(len(r_y)):
            for j in range(len(r_x)):
                pos=np.array([r_x[j],r_y[i]])
# =============================================================================
#                 if self.get_block(np.array([r_x[j],r_y[i]]))==36:
#                     pdb.set_trace()
# =============================================================================
                neigh=get_neighbourhood(directness, len(self.x), self.get_block(pos))
                Ens=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, neigh)]#sources in the neighbourhood
                for k in Ens:
                    """Goes through all the sources in the neighbourhood"""
                    rec[i,j]+=Green(self.pos_s[k], pos, Rv[k])*self.phi_q[k]
        return(rec)

    
        

class reconstruction_sans_flux():
    def __init__(self, solution, ass_object, L, ratio, directness):
        """The solution must be one dimensional array with the values on the 
        FV cells first and then the values of the flux"""
        self.phi_FV, self.phi_q=separate_unk(ass_object, solution)
        self.t=ass_object
        t=self.t
        self.L=L
        self.dual_x=np.arange(0, L+0.01*t.h, t.h)
        self.dual_y=np.arange(0, L+0.01*t.h, t.h)
        self.rec_final=np.zeros((len(t.x)*ratio, len(t.x)*ratio))
        self.rec_s_FV=np.zeros((len(t.x)*ratio, len(t.x)*ratio))
        self.rec_potentials=np.zeros((len(t.x)*ratio, len(t.x)*ratio))
        self.ratio=ratio
        self.directness=directness
        self.dual_boundary=get_boundary_vector(len(self.dual_x), len(self.dual_y))
        #boundary array of the original mesh
        self.orig_boundary=get_boundary_vector(len(t.x), len(t.y))
        

        
    def get_bilinear_interpolation(self, position):
        #pdb.set_trace()
        blocks=get_4_blocks(position, self.t.x, self.t.y, self.t.h) #gets the ID of each of the 4 blocks 
        corner_values=get_unk_same_reference(blocks, self.directness, len(self.t.x), 
                               self.t.s_blocks, self.t.uni_s_blocks, self.phi_FV, 
                               self.phi_q, self.t.pos_s, self.t.Rv, self.t.x, 
                               self.t.y)
        
        ens_neigh=get_multiple_neigh(self.directness, len(self.t.x), blocks)
        total_sources=np.arange(len(self.t.s_blocks))[np.in1d(self.t.s_blocks, ens_neigh)]
        phi_FV=bilinear_interpolation(corner_values, self.ratio)
        local_sol=phi_FV.copy()
        self.temp_u=phi_FV
        self.temp_pot=np.zeros(self.temp_u.shape)
        if np.any(np.in1d(ens_neigh, self.t.uni_s_blocks)):
            #The reconstruction of the SL and DL potentials in this cell
            self.temp_pot=green_to_block(self.phi_q, position,self.t.h,self.ratio, ens_neigh, 
                                self.t.s_blocks,self.t.pos_s,self.t.Rv)
            local_sol+=self.temp_pot
            

        return(local_sol)
    
    def get_NN_interpolation(self, position):
        #pdb.set_trace()
        blocks=get_4_blocks(position, self.t.x, self.t.y, self.t.h)
        corner_values=get_unk_same_reference(blocks, self.directness, len(self.t.x), 
                               self.t.s_blocks, self.t.uni_s_blocks, self.phi_FV,
                               self.phi_q, self.t.pos_s, self.t.Rv, self.t.x, self.t.y)
        
        ens_neigh=get_multiple_neigh(self.directness, len(self.t.x), blocks)
        total_sources=np.arange(len(self.t.s_blocks))[np.in1d(self.t.s_blocks, ens_neigh)]
        
        rec=NN_interpolation(corner_values, self.ratio)

        if np.any(np.in1d(ens_neigh, self.t.uni_s_blocks)):
            rec+=green_to_block(self.phi_q, position,self.t.h,self.ratio, ens_neigh, 
                                self.t.s_blocks,self.t.pos_s,self.t.Rv)
            
        return(rec) 
        
    def reconstruction(self, *rec_type):
        t=self.t
        ratio=self.ratio
        x,y=get_sub_x_y(t.x, t.y, t.h, ratio)
        self.x, self.y=x,y
        c=0
        if rec_type:
            typ="NN"
        else:
            typ="bil"
        for i in self.dual_x[1:-1]:
            d=0
            for j in self.dual_y[1:-1]:
                pos_x=((x>=i-t.h/2) & (x<i+t.h/2))
                pos_y=((y>=j-t.h/2) & (y<j+t.h/2))

                local_sol=self.get_bilinear_interpolation(np.array([i,j]))
                self.rec_s_FV=modify_matrix(self.rec_s_FV, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], self.temp_u)
                self.rec_potentials=modify_matrix(self.rec_potentials, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], self.temp_pot)
                self.rec_final=modify_matrix(self.rec_final, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], local_sol)

                d+=1
            c+=1
        return(self.rec_final)

    def reconstruction_boundaries_short(self, BC_type, BC_value):
        dual_bound=get_boundary_vector(len(self.dual_x), len(self.dual_y))
        phi_FV=self.phi_FV
        phi_q=self.phi_q
        ratio=self.ratio
        s_blocks=self.t.s_blocks
        pos_s=self.t.pos_s
        Rv=self.t.Rv
        t=self.t
        x,y=get_sub_x_y(t.x, t.y, t.h, ratio)
        h=t.h
        #The following gets the slow term value at the boundary 
        self.boundary_values=set_boundary_values(BC_type, BC_value, phi_FV,
                                         phi_q, t.x, t.y, 
                                         self.orig_boundary,self.t.h,self.t.D, t) 
        for c in range(4):
            o=0
            
            for b in self.dual_boundary[c,1:-1]:

                normal=np.array([[0,1],[0,-1],[1,0],[-1,0]])[c]
                tau=np.array([[0,1],[-1,0]]).dot(normal)
                
                #position at the center point of the boundary of this block:
                #Also can be described as the center of hte dual block 
                p_dual=pos_to_coords(self.dual_x,self.dual_y, b)
                
                pos_blocks=np.array([p_dual -(normal+tau)*h/2,p_dual+(-normal+tau)*h/2])
                
                #the unknowns that are included here
                first_block=coord_to_pos(t.x, t.y,p_dual -(normal+tau)*h/2)
                second_block=coord_to_pos(t.x, t.y, p_dual+(-normal+tau)*h/2)
                
                blocks=np.array([first_block, second_block])
                ens_neigh=get_multiple_neigh(self.directness, len(t.x), blocks)
                unk_extended=get_unk_same_reference(blocks, self.directness, len(t.x), t.s_blocks, 
                                                    t.uni_s_blocks, self.phi_FV, phi_q, t.pos_s, Rv, t.x, t.y)
                
                
                #Now we need the uncommon rapid potential to put the boundary values in the same reference
                N_k=get_neighbourhood(t.directness, len(t.x), first_block)
                N_m=get_neighbourhood(t.directness, len(t.x), second_block)
                
                unc_k_m=get_uncommon(N_k, N_m)
                unc_m_k=get_uncommon(N_m, N_k)
# =============================================================================
#                 Em=np.arange(len(t.s_blocks))[np.in1d(t.s_blocks,unc_m_k)]
#                 #sources in the neighbourhood of k that are not in the neigh of m
#                 Ek=np.arange(len(t.s_blocks))[np.in1d(t.s_blocks,unc_k_m)]
# =============================================================================
                #From here onwards it will differ depending on BCs:
                    
                NE=np.array([unc_m_k,unc_k_m])
                unk_boundaries=np.zeros(2)

                for k in range(2):
                    p = o + k if np.sum(tau) == 1 else o+1-k
                    unk_boundaries[k]=self.boundary_values[c,p]-np.dot(kernel_green_neigh(pos_blocks[k],
                                                                                   NE[k], t.pos_s, t.s_blocks, t.Rv), phi_q)
                
                
                phi_FV=bilinear_interpolation(np.array([unk_extended[0], unk_boundaries[0], unk_extended[1], unk_boundaries[1]]), ratio)
                
                pos_y=((y>=p_dual[1]-t.h/2) & (y<p_dual[1]+t.h/2))
                pos_x=((x>=p_dual[0]-t.h/2) & (x<p_dual[0]+t.h/2))
                
                if c==1:
                    phi_FV=rotate_180(phi_FV)
                elif c==3:
                    phi_FV=rotate_clockwise(phi_FV)
                elif c==2:
                    phi_FV=rotate_counterclock(phi_FV)
                
                
                #I think up to here is the Dirichlet specific condition
                phi_FV=reduced_half_direction(phi_FV,int(np.abs(normal[0])))
                local_sol=phi_FV.copy()
                SL=np.zeros(phi_FV.shape)
                for i in np.arange(np.sum(pos_y)):
                    for j in np.arange(np.sum(pos_x)):
                        #position of the point
                        p_pos=np.array([x[pos_x][j], y[pos_y][i]])
                        kernel=kernel_green_neigh(p_pos, ens_neigh, t.pos_s, t.s_blocks, t.Rv)
                        SL[i,j]+=kernel.dot(phi_q)
                        local_sol[i,j]+=kernel.dot(phi_q)
                
                self.rec_final=modify_matrix(self.rec_final, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], local_sol)
                self.rec_s_FV=modify_matrix(self.rec_s_FV, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], phi_FV)
                self.rec_potentials=modify_matrix(self.rec_potentials, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], SL)
                o+=1
                
        return()
    
   
    def rec_corners(self):
        
        phi_q=self.phi_q
        ratio=self.ratio
        s_blocks=self.t.s_blocks
        pos_s=self.t.pos_s
        Rv=self.t.Rv
        t=self.t
        bound=self.orig_boundary #boundary of the original mesh 
        
        x,y=get_sub_x_y(t.x, t.y, t.h, ratio)
        h=t.h
        Bn, Bs, Be, Bw=self.boundary_values 
        phi_FV=self.phi_FV #values of s at each cell
        
        for i in range(4):
            if i==0:
                #south-west
                
                p_dual=np.array([self.dual_x[0], self.dual_y[0]])
                block=bound[1,0]
                pos_y=(y<t.y[0])
                pos_x=(x<t.x[0])
                #This for the slow term
                array_bil=np.array([(Bs[0]+Bw[0])/2, Bw[0], Bs[0], phi_FV[block]])

            if i==1:
                #north-west
                p_dual=np.array([self.dual_x[0], self.dual_y[-1]])
                block=bound[0,0]
                pos_y=(y>t.y[-1])
                pos_x=(x<t.x[0])
                #This for the slow term
                array_bil=np.array([Bw[-1], (Bw[-1]+Bn[0])/2, phi_FV[block], Bn[0]])
                
            if i==2:
                #south-east
                p_dual=np.array([self.dual_x[-1], self.dual_y[0]])
                block=bound[1,-1]       
                pos_y=(y<t.y[0])
                pos_x=(x>t.x[-1])
                #This for the slow term
                array_bil=np.array([Bs[-1], phi_FV[block], (Bs[-1]+Be[0])/2, Be[0]])
                
                
            if i==3:
                #north-east
                p_dual=np.array([self.dual_x[-1], self.dual_y[-1]])
                block=bound[0,-1]
                pos_y=(y>t.y[-1])
                pos_x=(x>t.x[-1])
                #This for the slow term
                array_bil=np.array([phi_FV[block],Bn[-1], Be[-1], (Bn[-1]+Be[-1])/2])
            
            #Bilinear interpolation for the background slow field
            local_sol=bilinear_interpolation(array_bil, int(ratio/2))
            neigh=get_neighbourhood(self.directness, len(t.x), block)
            
            #rapid term 
            rapid_local=np.zeros(local_sol.shape)
            loc_x=x[pos_x]
            loc_y=y[pos_y]
            for i in range(len(loc_x)):
                for j in range(len(loc_y)):
                    pos=np.array([loc_x[i],loc_y[j]])
                    rapid_local[j,i]=np.dot(kernel_green_neigh(pos, neigh, 
                                                               t.pos_s, t.s_blocks, t.Rv), phi_q)
            
            #I am very tired, so the reconstruction in the corners will be done with the absolute value of the concentration
            self.rec_final=modify_matrix(self.rec_final, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], local_sol+ rapid_local)
            self.rec_s_FV=modify_matrix(self.rec_s_FV, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], local_sol)
            self.rec_potentials=modify_matrix(self.rec_potentials, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y],rapid_local)
            
    def get_u_pot(self, C_v_array): 
        """This function is designed to return the single layer and double layer potentials in one matrix and
        the positive background concentration in another"""

        phi_q=self.phi_q
        potentials=self.rec_potentials
        u=self.rec_s_FV
        s_blocks=self.t.s_blocks
        x,y=self.x, self.y
        t=self.t
        bound=self.orig_boundary
        for i in self.dual_x[1:-1]:
            for j in self.dual_y[1:-1]:
                #boolean arrays of the positions of the fine mesh occupied by 
                #the dual volume
                pos_x=((x>=i-t.h/2) & (x<i+t.h/2))
                pos_y=((y>=j-t.h/2) & (y<j+t.h/2))
                
                #Array of integer positions of the fine mesh occupied by 
                #the dual volume
                p1=np.arange(len(x))[pos_x]
                p2=np.arange(len(y))[pos_y]
                #Get the four blocks bordering the dual volume 
                blocks=get_4_blocks(np.array([i,j]), self.t.x, self.t.y, self.t.h) #gets the ID of each of the 4 blocks 
                ens_neigh=get_multiple_neigh(self.directness, len(self.t.x), blocks)
                total_sources=np.in1d(s_blocks, ens_neigh)
                
                constant=0
                for jj in np.arange(len(t.pos_s))[total_sources]:
                    constant+=C_v_array[jj] - phi_q[jj]/self.t.K_0[jj]
                #pdb.set_trace()
                u=modify_matrix(u,p1 , p2, self.rec_s_FV[p2,:][:,p1]-constant)
                potentials=modify_matrix(potentials, p1 , p2, self.rec_potentials[p2,:][:,p1]+constant)

        #Now for the (smaller) boundary volumes
        dual_bound=get_boundary_vector(len(self.dual_x), len(self.dual_y))
        
        #pdb.set_trace()
        Bn, Bs, Be, Bw=self.boundary_values   
        for i in range(4):
            for b in self.dual_boundary[i,1:-1]:

                normal=np.array([[0,1],[0,-1],[1,0],[-1,0]])[i]
                tau=np.array([[0,1],[-1,0]]).dot(normal)
                p_dual=pos_to_coords(self.dual_x,self.dual_y, b)
                pos_y=((y>=p_dual[1]-t.h/2) & (y<p_dual[1]+t.h/2))
                pos_x=((x>=p_dual[0]-t.h/2) & (x<p_dual[0]+t.h/2))
                
                p1=np.arange(len(x))[pos_x]
                p2=np.arange(len(y))[pos_y]
                
                #the unknowns that are included here
                first_block=coord_to_pos(t.x, t.y,p_dual -(normal+tau)*t.h/2)
                second_block=coord_to_pos(t.x, t.y, p_dual+(-normal+tau)*t.h/2)
                
                blocks=np.array([first_block, second_block])
                ens_neigh=get_multiple_neigh(self.directness, len(t.x), blocks)
                
                total_sources=np.in1d(s_blocks, ens_neigh)
                
                constant=0
                for jj in np.arange(len(t.pos_s))[total_sources]:
                    constant+=C_v_array[jj] - phi_q[jj]/self.t.K_0[jj]
                #pdb.set_trace()
                u=modify_matrix(u,p1 , p2, self.rec_s_FV[p2,:][:,p1]-constant)
                potentials=modify_matrix(potentials, p1 , p2, self.rec_potentials[p2,:][:,p1]+constant)
            
            if i==0:
                #south-west
                
                p_dual=np.array([self.dual_x[0], self.dual_y[0]])
                block=bound[1,0]
                pos_y=(y<t.y[0])
                pos_x=(x<t.x[0])
                #This for the slow term
    
            if i==1:
                #north-west
                p_dual=np.array([self.dual_x[0], self.dual_y[-1]])
                block=bound[0,0]
                pos_y=(y>t.y[-1])
                pos_x=(x<t.x[0])
                #This for the slow term
                
            if i==2:
                #south-east
                p_dual=np.array([self.dual_x[-1], self.dual_y[0]])
                block=bound[1,-1]       
                pos_y=(y<t.y[0])
                pos_x=(x>t.x[-1])
                #This for the slow term
                
                
            if i==3:
                #north-east
                p_dual=np.array([self.dual_x[-1], self.dual_y[-1]])
                block=bound[0,-1]
                pos_y=(y>t.y[-1])
                pos_x=(x>t.x[-1])
                #This for the slow term
            p1=np.arange(len(x))[pos_x]
            p2=np.arange(len(y))[pos_y]
            

            neigh=get_neighbourhood(self.directness, len(t.x), block)
            
            total_sources=np.in1d(s_blocks, neigh)
            constant=0
            for jj in np.arange(len(t.pos_s))[total_sources]:
                constant+=C_v_array[jj] - phi_q[jj]/self.t.K_0[jj]
            #pdb.set_trace()
            u=modify_matrix(u,p1 , p2, self.rec_s_FV[p2,:][:,p1]-constant)
            potentials=modify_matrix(potentials, p1 , p2, self.rec_potentials[p2,:][:,p1]+constant)
        
        
        return(u, potentials)

