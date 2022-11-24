
import os 
import numpy as np
from module_2D_coupling_FV_nogrid import * 

def get_errors(phi_SS, a_rec_final, noc_sol, p_sol,SS_phi_q, p_q, noc_q,ratio, phi_q):
    errors=[["coupling","SS" , ratio , get_L2(SS_phi_q, phi_q) , get_L2(phi_SS, a_rec_final) , get_MRE(SS_phi_q, phi_q) , get_MRE(phi_SS, a_rec_final)],
        ["coupling","Peaceman", ratio,get_L2(p_q, phi_q), get_L2(p_sol, np.ndarray.flatten(a_rec_final)), get_MRE(p_q, phi_q), get_MRE(p_sol, np.ndarray.flatten(a_rec_final))],
        ["FV","SS",1,get_L2(SS_phi_q, noc_q), get_L2(np.ndarray.flatten(phi_SS), noc_sol), get_MRE(SS_phi_q, phi_q), get_MRE(np.ndarray.flatten(phi_SS), noc_sol)],
        ["FV","Peaceman",1,get_L2(p_q, noc_q), get_L2(p_sol, noc_sol), get_MRE(p_q, phi_q), get_MRE(p_sol, noc_sol)],
        ["Peaceman","SS", 1,get_L2(SS_phi_q, p_q), get_L2(np.ndarray.flatten(phi_SS), p_sol), get_MRE(SS_phi_q, p_q), get_MRE(np.ndarray.flatten(phi_SS), p_sol)]]
    return(errors)

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
    

    

# =============================================================================
# class reconstruction_coupling(assemble_SS_2D_FD):
#     """
#     This class is meant to reconstruct the solution from the local solution splitting coupling method
#     provided the value of the post-localization regular term (u) and the source fluxes (q)
#     The pipeline will be as follows:
#         1- A dual mesh is obtained where the h_dual is half of the one of the original mesh
#         2- The values for the concentration ($\phi$) are retrieved at each point of the dual mesh
#            The value of the concentration is needed since the values of the regular term will change 
#            for different cell's for the same position. Therefore, the values of phi are retrieved and 
#            from there the reconstruction can be made on each cell through the singular term on each cell
#            
#            The values of the concentration are calculated as the average of the contribution by each FV cell 
#            to the concentration at that point.
#                - For the dual mesh points that fall on the interface between two cell's, the average value of the 
#                  concentration at the interface provided by each cell should be the same since phi-continuity accross
#                  interfaces has been imposed through the numerical model
#                - For the values at the dual mesh points that fall on the corner's of the original mesh it will be more 
#                  tricky since the values of the concentration for each FV at that point are not necessarily the same. 
#                  Therefore, here the averaging does make sense, and might provide a value that is different from the 
#                  4 contributions
#     """
#     def __init__(self, ratio, pos_s, Rv, h,L, K_eff, D,directness):
#         assemble_SS_2D_FD.__init__(self,pos_s, Rv, h,L, K_eff, D,directness)
#         
# 
#         self.K_0=K_eff*np.pi*Rv**2
#         
#         #For the reconstruction a dual mesh is created:
#         self.factor=1 #this number will indicate the amount of dual mesh cells that fit in a normal FV cell (normally 1 or 2)
#         #the ratio must necessarily be a multiple of this factor :
#         self.ratio=int(ratio//self.factor)*self.factor
#         
#         self.rec_u=np.zeros((len(self.y)*ratio, len(self.x)*ratio))
#         self.rec_bar_S=self.rec_u.copy()
#         
#         self.dual_h=self.h/self.factor
#         self.dual_x=np.arange(0, L+0.01*self.h, self.dual_h)
#         self.dual_y=np.arange(0, L+0.01*self.h, self.dual_h)
#         self.dual_boundary=get_boundary_vector(len(self.dual_x), len(self.dual_y))
#     
#     def solve_steady_state_problem(self, boundary_values, C_v_values):
#         self.boundary_values=boundary_values
#         
#         self.pos_arrays()
#         self.initialize_matrices()
#         self.full_M=self.assembly_sol_split_problem(np.array([0,0,0,0]))
#         self.S=len(self.pos_s)
#         
#         self.H0[-self.S:]=-C_v_values*self.K_0
#         sol=np.linalg.solve(self.full_M, -self.H0)
#         phi_FV=sol[:-self.S]
#         phi_q=sol[-self.S:]
#         self.phi_q, self.phi_FV=phi_q, phi_FV
#     
#     def retrieve_concentration_dual_mesh(self):
#         """This function executes the 2 point of the description of this class. It will go 
#         one by one through the dual mesh and it will provide a value for the concentration 
#         at each point"""
#         dual_conc=np.zeros((len(self.dual_y), len(self.dual_x)))
#         c=1
#         for i in self.dual_x[1:-1]:
#             d=1
#             for j in self.dual_y[1:-1]:
#                 blocks=get_blocks_radius(np.array([i,j]), self.h, self.x, self.y)
#                 value=0
#                 #get the real concentration at that point from each block:
#                 for k in blocks:
#                     neigh=get_neighbourhood(self.directness, len(self.x), k)
#                     #The value of the singular term will be given by:
#                     bar_S=np.dot(kernel_green_neigh(np.array([i,j]), neigh, 
#                                                        self.pos_s, self.s_blocks, self.Rv), self.phi_q)
#                     value+=self.phi_FV[k]+bar_S
#                 dual_conc[d, c]=value/len(blocks)
#                 d+=1
#             c+=1
#         #boundary values
#         dual_conc[0,:]=self.boundary_values[1]
#         dual_conc[-1,:]=self.boundary_values[0]
#         dual_conc[:,-1]=self.boundary_values[2]
#         dual_conc[:,0]=self.boundary_values[3]
#         
#         self.dual_conc=dual_conc
#         return()
# 
#     def execute_interpolation(self, phi_q, phi_FV):
#         for i in range(len(self.dual_x)-1):
#             #pdb.set_trace()
#             for j in range(len(self.dual_y)-1):
#                 #This loop goes through each "dual element", where it does the interpolation
#                 #of the four corner values (which have already been calculated). The advantage of 
#                 #this class is that every ""dual element" only lies within one original element. 
#                 #Therefore, there is no ambiguity on the singular term to use
# 
#                 #only a single block's information is necessary
#                 
#                 len_x=len(self.x)
#                 len_y=len(self.y)
#                 
#                 dual_block_center=np.array([self.dual_x[i], self.dual_y[j]])+self.dual_h/2
#                 block=coord_to_pos(self.x, self.y, dual_block_center) #the FV block whithin the dual block falls
#                 corner_pos=np.array([[i,j],[i,j+1],[i+1,j],[i+1,j+1]])
#                 corner_values=self.dual_conc[corner_pos[:,0], corner_pos[:,1]]
#                 
#                 position_corners=np.array([self.dual_x[corner_pos[:,0]], self.dual_y[corner_pos[:,1]]]).T
#                 
#                 S_c_values=np.array([])
#                 neigh=get_neighbourhood(self.directness, len_x, block)
#                 for coord_corner in position_corners:
#                     #this loop goes through the position of the corners
#                     G_sub=kernel_green_neigh(coord_corner, neigh, self.pos_s, self.s_blocks, self.Rv)
#                     S_c_values=np.append(S_c_values, np.dot(G_sub, phi_q))
#                     
#                 pos_x=np.arange(self.ratio/self.factor, dtype=int)+int(i*self.ratio/self.factor)
#                 pos_y=np.arange(self.ratio/self.factor, dtype=int)+int(j*self.ratio/self.factor)
#                 block_S_bar=green_to_block(phi_q, dual_block_center, self.dual_h, int(self.ratio/self.factor),
#                                            neigh, self.s_blocks, self.pos_s,self.Rv)
#                 
#                 
#                 self.rec_bar_S=modify_matrix(self.rec_bar_S,pos_x,pos_y, block_S_bar)
#                 
#                 local_sol=bilinear_interpolation(corner_values-S_c_values , int(self.ratio/self.factor))          
#                 self.rec_phi_FV=modify_matrix(self.rec_phi_FV, pos_x, pos_y, local_sol)
#                 
#                 
# =============================================================================
        
        
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
        self.rec_phi_FV=np.zeros((len(t.x)*ratio, len(t.x)*ratio))
        self.rec_potentials=np.zeros((len(t.x)*ratio, len(t.x)*ratio))
        self.ratio=ratio
        self.directness=directness
        self.dual_boundary=get_boundary_vector(len(self.dual_x), len(self.dual_y))
        
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
                
                if typ=="NN":
                    local_sol=self.get_NN_interpolation(np.array([i,j]))
                    print("Nearest neighbourg interpolation")
                else:
                    
                    local_sol=self.get_bilinear_interpolation(np.array([i,j]))
                    self.rec_phi_FV=modify_matrix(self.rec_phi_FV, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], self.temp_u)
                    self.rec_potentials=modify_matrix(self.rec_potentials, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], self.temp_pot)
                    self.rec_final=modify_matrix(self.rec_final, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], local_sol)
                d+=1
            c+=1
        return(self.rec_final)
    
    def reconstruction_boundaries(self, boundary_values):
        dual_bound=get_boundary_vector(len(self.dual_x), len(self.dual_y))
        self.boundary_values=boundary_values
        phi_q=self.phi_q
        ratio=self.ratio
        s_blocks=self.t.s_blocks
        pos_s=self.t.pos_s
        Rv=self.t.Rv
        t=self.t
        x,y=get_sub_x_y(t.x, t.y, t.h, ratio)
        h=t.h
        self.boundary_values=boundary_values
        
        Bn, Bs, Be, Bw=boundary_values   
        for c in range(4):
            for b in self.dual_boundary[c,1:-1]:

                normal=np.array([[0,1],[0,-1],[1,0],[-1,0]])[c]
                tau=np.array([[0,1],[-1,0]]).dot(normal)
                
                #position at the center point of the boundary of this block:
                #Also can be described as the center of hte dual block 
                p_dual=pos_to_coords(self.dual_x,self.dual_y, b)
                
                #the unknowns that are included here
                first_block=coord_to_pos(t.x, t.y,p_dual -(normal+tau)*h/2)
                second_block=coord_to_pos(t.x, t.y, p_dual+(-normal+tau)*h/2)
                
                blocks=np.array([first_block, second_block])
                ens_neigh=get_multiple_neigh(self.directness, len(t.x), blocks)
                unk_extended=get_unk_same_reference(blocks, self.directness, len(t.x), t.s_blocks, 
                                                    t.uni_s_blocks, self.phi_FV, phi_q, t.pos_s, Rv, t.x, t.y)
                
                unk_boundaries=np.zeros(2)
                position=np.zeros((2,2))
                for k in range(2):
                    #position of the boundary unknown
                    position[k,:]=pos_to_coords(t.x, t.y, blocks[k])+normal*t.h/2
                    unk_boundaries[k]=boundary_values[c]-kernel_green_neigh(position[k,:], ens_neigh, pos_s, s_blocks, Rv).dot(phi_q)
                
                phi_FV=bilinear_interpolation(np.array([unk_extended[0], unk_boundaries[0], unk_extended[1], unk_boundaries[1]]), ratio)
                
                pos_y=((y>=p_dual[1]-t.h/2) & (y<p_dual[1]+t.h/2))
                pos_x=((x>=p_dual[0]-t.h/2) & (x<p_dual[0]+t.h/2))
                
                if c==1:
                    phi_FV=rotate_180(phi_FV)
                elif c==3:
                    phi_FV=rotate_clockwise(phi_FV)
                elif c==2:
                    phi_FV=rotate_counterclock(phi_FV)
                
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
                self.rec_phi_FV=modify_matrix(self.rec_phi_FV, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], phi_FV)
                self.rec_potentials=modify_matrix(self.rec_potentials, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], SL)
                
        return()
    
    def rec_corners(self):
        
        phi_q=self.phi_q
        ratio=self.ratio
        s_blocks=self.t.s_blocks
        pos_s=self.t.pos_s
        Rv=self.t.Rv
        t=self.t
        bound=get_boundary_vector(len(t.x), len(t.y))
        
        x,y=get_sub_x_y(t.x, t.y, t.h, ratio)
        h=t.h
        Bn, Bs, Be, Bw=self.boundary_values 
        
        
        for i in range(4):
            if i==0:
                #south-west
                p_dual=np.array([self.dual_x[0], self.dual_y[0]])
                block=bound[1,0]
                pos_y=(y<t.y[0])
                pos_x=(x<t.x[0])
                
                neigh=get_neighbourhood(self.directness, len(t.x), block)
                value=self.phi_FV[block]+kernel_green_neigh(pos_to_coords(t.x, t.y, block), neigh, t.pos_s, t.s_blocks, t.Rv).dot(phi_q)
                array_bil=np.array([(Bs+Bw)/2, Bw, Bs, value])
            
            if i==1:
                #north-west
                p_dual=np.array([self.dual_x[0], self.dual_y[-1]])
                block=bound[0,0]
                pos_y=(y>t.y[-1])
                pos_x=(x<t.x[0])
                
                neigh=get_neighbourhood(self.directness, len(t.x), block)
                value=self.phi_FV[block]+kernel_green_neigh(pos_to_coords(t.x, t.y, block), neigh, t.pos_s, t.s_blocks, t.Rv).dot(phi_q)
                array_bil=np.array([Bw, (Bw+Bn)/2, value, Bn])
                
            if i==2:
                #south-east
                p_dual=np.array([self.dual_x[-1], self.dual_y[0]])
                block=bound[1,-1]       
                pos_y=(y<t.y[0])
                pos_x=(x>t.x[-1])
                neigh=get_neighbourhood(self.directness, len(t.x), block)
                value=self.phi_FV[block]+kernel_green_neigh(pos_to_coords(t.x, t.y, block), neigh, t.pos_s, t.s_blocks, t.Rv).dot(phi_q)
                array_bil=np.array([Bs, value, (Bs+Be)/2, Be])
                
                
            if i==3:
                #north-east
                p_dual=np.array([self.dual_x[0], self.dual_y[-1]])
                block=bound[0,-1]
                pos_y=(y>t.y[-1])
                pos_x=(x>t.x[-1])
                neigh=get_neighbourhood(self.directness, len(t.x), block)
                value=self.phi_FV[block]+kernel_green_neigh(pos_to_coords(t.x, t.y, block), neigh, t.pos_s, t.s_blocks, t.Rv).dot(phi_q)
                array_bil=np.array([value,Bn, Be, (Bn+Be)/2])


            local_sol=bilinear_interpolation(array_bil, int(ratio/2))
            
            #I am very tired, so the reconstruction in the corners will be done with the absolute value of the concentration
            self.rec_final=modify_matrix(self.rec_final, np.arange(len(x))[pos_x], np.arange(len(y))[pos_y], local_sol)

    def get_u_pot(self, C_v_array): 
        """This function is designed to return the single layer and double layer potentials in one matrix and
        the positive background concentration in another"""

        phi_q=self.phi_q
        potentials=self.rec_potentials
        u=self.rec_phi_FV
        s_blocks=self.t.s_blocks
        x,y=self.x, self.y
        t=self.t
        for i in self.dual_x[1:-1]:
            for j in self.dual_y[1:-1]:
                
                pos_x=((x>=i-t.h/2) & (x<i+t.h/2))
                pos_y=((y>=j-t.h/2) & (y<j+t.h/2))
                
                p1=np.arange(len(x))[pos_x]
                p2=np.arange(len(y))[pos_y]
                
                blocks=get_4_blocks(np.array([i,j]), self.t.x, self.t.y, self.t.h) #gets the ID of each of the 4 blocks 
                ens_neigh=get_multiple_neigh(self.directness, len(self.t.x), blocks)
                total_sources=np.in1d(s_blocks, ens_neigh)
                
                constant=0
                for jj in np.arange(len(t.pos_s))[total_sources]:
                    constant+=C_v_array[jj] - phi_q[jj]/self.t.K_0[jj]
                #pdb.set_trace()
                u=modify_matrix(u,p1 , p2, self.rec_phi_FV[p2,:][:,p1]-constant)
                potentials=modify_matrix(potentials, p1 , p2, self.rec_potentials[p2,:][:,p1]+constant)

            
        dual_bound=get_boundary_vector(len(self.dual_x), len(self.dual_y))
        
        #pdb.set_trace()
        Bn, Bs, Be, Bw=self.boundary_values   
        for c in range(4):
            for b in self.dual_boundary[c,1:-1]:

                normal=np.array([[0,1],[0,-1],[1,0],[-1,0]])[c]
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
                u=modify_matrix(u,p1 , p2, self.rec_phi_FV[p2,:][:,p1]-constant)
                potentials=modify_matrix(potentials, p1 , p2, self.rec_potentials[p2,:][:,p1]+constant)
                
        
        return(u, potentials)

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



def coarse_cell_center_rec(x, y, phi_FV, pos_s, s_blocks, phi_q, ratio, h, directness, Rv):
        """PREVIOUSLY IT WAS CALLED coarse_NN_rec!!!!!!!!!!!!!!!!!!!!!!
        
        phi_FV is given in matrix form
        This funcion does a coarse mesh reconstruction with average u values on the cells plus
        singular term values at the cell's center"""
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
        
# =============================================================================
# def get_4_blocks(position, tx, ty, th):
#     blocks_x=np.where(np.abs(tx-position[0]) < th*1.001)[0]
#     blocks_y=np.where(np.abs(ty-position[1]) < th*1.001)[0]*len(tx)
#     blocks=np.array([])
# # =============================================================================
# #     for i in blocks_x:
# #         for j in blocks_y:
# #             blocks=np.append(blocks, i+j)
# # =============================================================================
#     blocks=np.array([blocks_y[0]+blocks_x[0], blocks_y[1]+blocks_x[0],blocks_y[0]+blocks_x[1],blocks_y[1]+blocks_x[1]])
#     return(blocks)
# =============================================================================
    





class reconstruction_extended_space(assemble_SS_2D_FD):
    def __init__(self,pos_s, Rv, h,L, K_eff, D,directness):
        assemble_SS_2D_FD.__init__(self,pos_s, Rv, h,L, K_eff, D,directness)
    def solve_linear_prob(self, dirichlet_array,C_v_array):
        """Solves the problem without metabolism, necessary to provide an initial guess
        DIRICHLET ARRAY ARE THE VALUES OF THE CONCENTRATION VALUE AT EACH BOUNDARY, THE CODE
        IS NOT YET MADE TO CALCULATE ANYTHING OTHER THAN dirichlet_array=np.zeros(4)"""
        S=len(self.pos_s)
        self.C_v_array=C_v_array
        self.S=S
        K0=self.K_0
        self.pos_arrays()
        self.initialize_matrices()
        LIN_MAT=self.assembly_sol_split_problem(dirichlet_array)
        self.H0[-S:]-=C_v_array
        #t.B[-np.random.randint(0,S,int(S/2))]=0
        sol=np.linalg.solve(LIN_MAT, -self.H0)
        phi_FV=sol[:-S].reshape(len(self.x), len(self.y))
        phi_q=sol[-S:]
        
        #initial guesses
        self.phi_FV=np.ndarray.flatten(phi_FV)
        self.phi_q=phi_q
    
        
    def set_up_manual_reconstruction_space(self, FEM_x, FEM_y):
        
        self.dual_x=FEM_x
        self.dual_y=FEM_y
        self.rec_final=np.zeros(len(FEM_x))
        self.rec_phi_FV=np.zeros(len(FEM_x))
        self.rec_potentials=np.zeros(len(FEM_x))
        #Get the boundary nodes:
        south=np.arange(len(FEM_x))[self.dual_y<self.y[0]]
        north=np.arange(len(FEM_x))[self.dual_y>self.y[-1]]
        east=np.arange(len(FEM_x))[self.dual_x>self.x[-1]]
        west=np.arange(len(FEM_x))[self.dual_x<self.x[0]]
        boundaries=np.concatenate((north, south, east, west))
                
        self.up_right=np.arange(len(FEM_x))[(self.dual_y>self.y[-1]) & (self.dual_x>self.x[-1])]
        self.up_left=np.arange(len(FEM_x))[(self.dual_y>self.y[-1]) & (self.dual_x<self.x[0])]
        self.down_right=np.arange(len(FEM_x))[(self.dual_y<self.y[0]) & (self.dual_x>self.x[-1])]
        self.down_left=np.arange(len(FEM_x))[(self.dual_y<self.y[0]) & (self.dual_x<self.x[0])]

        self.dual_corners=np.concatenate([self.down_left, self.up_left, self.down_right, self.up_right])
        self.south=np.delete(south, np.arange(len(south))[np.in1d(south, self.dual_corners)])
        self.north=np.delete(north, np.arange(len(north))[np.in1d(north, self.dual_corners)])
        self.east=np.delete(east, np.arange(len(east))[np.in1d(east, self.dual_corners)])
        self.west=np.delete(west, np.arange(len(west))[np.in1d(west, self.dual_corners)])
        
        self.inner=np.delete(np.arange(len(self.dual_x)), boundaries)
        self.boundaries=np.concatenate((self.north, self.south, self.east, self.west))
        return()
        
    
    def reconstruction_manual(self):
        x,y=self.x, self.y
        rec_u=np.zeros(len(self.dual_x))
        rec_SL=np.zeros(len(self.dual_x))
        rec_DL=np.zeros(len(self.dual_x))
        
        for k in self.inner:
            #pdb.set_trace()
            block_pos=np.array([self.dual_x[k], self.dual_y[k]])
            blocks=get_4_blocks(block_pos, self.x, self.y, self.h) #gets the ID of each of the 4 blocks 
            corner_values=get_unk_same_reference(blocks, self.directness, len(self.x), 
                                   self.s_blocks, self.uni_s_blocks, np.ndarray.flatten(self.phi_FV), 
                                   self.phi_q, self.pos_s, self.Rv, x,y)
            
            ens_neigh=get_multiple_neigh(self.directness, len(self.x), blocks)
            total_sources=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, ens_neigh)]
            coord_vert=np.array([pos_to_coords(x,y,blocks[0]),pos_to_coords(x,y,blocks[1]),pos_to_coords(x,y,blocks[2]),pos_to_coords(x,y,blocks[3])])
            value=single_value_bilinear_interpolation(np.array([self.dual_x[k], self.dual_y[k]]), coord_vert, corner_values)
            rec_SL[k]=kernel_green_neigh(block_pos, ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q)
            rec_DL[k]=np.sum(self.C_v_array-self.phi_q/self.K_0)
            rec_u[k]=value-rec_DL[k]
        
        self.SL=rec_SL
        self.u=rec_u
        self.DL=rec_DL
        return()
            
    def reconstruction_boundaries_2(self, boundary_values):
        dual_bound=np.array([self.north, self.south, self.east, self.west])
        self.boundary_values=boundary_values
        phi_q=self.phi_q
        s_blocks=self.s_blocks
        pos_s=self.pos_s
        Rv=self.Rv
        dual_x,dual_y=self.dual_x, self.dual_y
        h=self.h
        
        Bn, Bs, Be, Bw=boundary_values
        
        for k in self.boundaries:
            block_pos=np.array([self.dual_x[k], self.dual_y[k]])
            normal=np.array([[0,-1],[0,1],[-1,0],[1,0]])[c]
            tau=np.array([[0,1],[-1,0]]).dot(normal)
            #the unknowns that are included here
            blocks=get_boundar_adj_blocks(self.x, self.y, block_pos)
            ens_neigh=get_multiple_neigh(self.directness, len(self.x), blocks)
            
            ue=get_unk_same_reference(blocks, self.directness, len(self.x), self.s_blocks, 
                                                self.uni_s_blocks, np.ndarray.flatten(self.phi_FV), phi_q, self.pos_s, Rv, self.x, self.y)
            
    def reconstruction_boundaries(self, boundary_values):
        dual_bound=np.array([self.north, self.south, self.east, self.west])
        self.boundary_values=boundary_values
        phi_q=self.phi_q
        s_blocks=self.s_blocks
        pos_s=self.pos_s
        Rv=self.Rv
        dual_x,dual_y=self.dual_x, self.dual_y
        h=self.h
        
        Bn, Bs, Be, Bw=boundary_values
    
        for b in self.boundaries:
            block_pos=np.array([self.dual_x[b], self.dual_y[b]])
            if b in self.north: c=0
            elif b in self.south: c=1
            elif b in self.east: c=2
            elif b in self.west: c=3
            
            normal=np.array([[0,-1],[0,1],[-1,0],[1,0]])[c]
            tau=np.array([[0,1],[-1,0]]).dot(normal)
            #the unknowns that are included here
            blocks=get_boundar_adj_blocks(self.x, self.y, block_pos)
            blocks=np.sort(blocks)
  
            ens_neigh=get_multiple_neigh(self.directness, len(self.x), blocks)
            
            ue=get_unk_same_reference(blocks, self.directness, len(self.x), self.s_blocks, 
                                                self.uni_s_blocks, np.ndarray.flatten(self.phi_FV), phi_q, self.pos_s, Rv, self.x, self.y)
            
            #The following calculates the center of the dual "boundary element" which comprises a quarter of each 
            #of the elements it touches. Furthermore, since it is a dual element, it is half of the size of 
            coord_center=(pos_to_coords(self.x, self.y, blocks[0])+pos_to_coords(self.x, self.y, blocks[1]))/2-normal*self.h/4
            
            #The following locates the coordinates of the four corners that make up the dual boundary element
            coord_vert=np.array([[tau/2+normal/4,tau/2-normal/4, -tau/2+normal/4, -tau/2-normal/4],
                                 [-tau/2-normal/4, -tau/2+normal/4, tau/2-normal/4,tau/2+normal/4],
                                 [-tau/2+normal/4, tau/2+normal/4, -tau/2 -normal/4, tau/2-normal/4],
                                 [tau/2-normal/4, -tau/2-normal/4, -tau/2+normal/4, -tau/2+normal/4]])[c]*self.h
            
            coord_vert+=np.array([np.zeros(4)+coord_center[0], np.zeros(4)+coord_center[1]]).T
            
            #set the boundary vertices on the same reference:
            corner_values=np.array([[ue[0],Bn-kernel_green_neigh(coord_vert[1], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q), ue[1], Bn-kernel_green_neigh(coord_vert[3], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q)],
                                    [Bs-kernel_green_neigh(coord_vert[0], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q), ue[0], Bs-kernel_green_neigh(coord_vert[2], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q), ue[1]],
                                    [ue[0], ue[1], Be-kernel_green_neigh(coord_vert[2], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q),Be-kernel_green_neigh(coord_vert[3], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q)],
                                    [Bw-kernel_green_neigh(coord_vert[0], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q), Bw-kernel_green_neigh(coord_vert[1], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q), ue[0], ue[1]]])[c]

            
            value=single_value_bilinear_interpolation(block_pos, coord_vert, corner_values)
            
            total_sources=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, ens_neigh)]
            self.SL[b]=kernel_green_neigh(block_pos, ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q)
            self.DL[b]=np.sum(self.C_v_array-self.phi_q/self.K_0)
            self.u[b]=value-self.DL[b]

        v=self.get_corners()
        v[1]=self.get_corners()[2]
        v[2]=self.get_corners()[1]
        for k in self.dual_corners:
            #single block:
# =============================================================================
#             if k in self.down_left: d=0
#             elif k in self.up_left: d=1
#             elif k in self.down_right: d=2
#             elif k in self.up_right: d=3
# =============================================================================
            block_pos=np.array([self.dual_x[k], self.dual_y[k]])
            block=get_boundar_adj_blocks(self.x, self.y, block_pos)[0]
            
            ens_neigh=get_neighbourhood(self.directness, len(self.x), block)
            w=int(np.arange(4)[v==block])
            L=self.L
            coord_vert=np.array([[[0,0],[0,h],[h,0],[h,h]],
                                 [[0,L-h],[0,L],[h,L-h],[h,L]],
                                 [[L-h,0],[L-h,h],[L,0],[L,h]],
                                 [[L-h,L-h],[L-h,L],[L,L-h],[L,L]]])[w]
            
            
            
            #set the boundary vertices on the same reference, for simplicity the corners I will interpolate the concentration directly:
            corner_values=np.array([[(Bs+Bw)/2, Bw, Bs, self.phi_FV[block]+kernel_green_neigh(coord_vert[3], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q)],
                                    [Bw, (Bw+Bn)/2,self.phi_FV[block]+kernel_green_neigh(coord_vert[2], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q), Bn ],
                                    [Bs, self.phi_FV[block]+kernel_green_neigh(coord_vert[1], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q), (Bs+Be)/2, Be],
                                    [self.phi_FV[block]+kernel_green_neigh(coord_vert[0], ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q), Bn, Be, (Bn+Be)/2]])[w]
            
            value=single_value_bilinear_interpolation(block_pos, coord_vert, corner_values)
            
            total_sources=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, ens_neigh)]
            
            self.SL[k]=kernel_green_neigh(np.array([self.dual_x[k], self.dual_y[k]]), ens_neigh, self.pos_s, self.s_blocks, self.Rv).dot(self.phi_q)
            self.DL[k]=np.sum(self.C_v_array-self.phi_q/self.K_0)
            self.u[k]=value-self.DL[k]-self.SL[k]
        return()
    


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
    
