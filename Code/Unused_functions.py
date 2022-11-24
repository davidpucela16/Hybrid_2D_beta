# -*- coding: utf-8 -*-

import numyp as np 
from module_2D_coupling_FV_nogrid import *
from Small_functions import *

def get_side(i, neigh, x):
    """It will return the side (north=0, south=1, east=2, west=3) the neighbour lies at"""
    lx=len(x)
    c=-5
    if i//lx == neigh//lx: #the neighbour lies in the same horizontal
        if neigh<i:
            c=3 #west
        else:
            c=2
    else: #the neighbours do not belong to the same horizontal
        if neigh>i:
            c=0
        else:
            c=1
    return(c)

def get_blocks_radius(point, radius, x, y):
    """This function will return the blocks whose center lies 
    within the circle centered at <point>, with radius <R>"""
    #get array of distances:
    distances=np.array([])
    
    for j in y:
        y_dis=point[1]-j
        x_dis=point[0]-x
        d=x_dis**2+y_dis**2
        d=np.sqrt(d)
        distances=np.concatenate((distances,d))
    return(np.where(distances<radius)[0])
        

def uni_vector(v0, vF):
    norm=np.sqrt(np.sum((vF-v0)**2))
    return((vF-v0)/norm)

class full_ss():
    """Creates a non local potential based linear problem"""
    def __init__(self, pos_s, Rv, h, K_eff, D,L):          
        #x=np.linspace(-h/2, L+h/2, int(L//h)+2)
        
        x=np.linspace(h/2,L-h/2, int(L/h))
        y=x
        self.x, self.y=x,y
        self.xlen, self.ylen=len(x), len(y)
        self.K_0=K_eff*np.pi*Rv**2
        self.pos_s=pos_s #exact position of each source
        self.h=h
        self.n_sources=self.pos_s.shape[0]
        self.Rv=Rv
        self.h=h
        self.D=D
        self.boundary=get_boundary_vector(len(x), len(y))
        
        source_FV=np.array([], dtype=int)
        for u in self.pos_s:
            r=np.argmin(np.abs(self.x-u[0]))
            c=np.argmin(np.abs(self.y-u[1]))
            source_FV=np.append(source_FV, c*len(self.x)+r) #block where the source u is located
        self.s_blocks=source_FV
    
    def solve_problem(self,B_q):
        self.setup_problem(B_q)
        v=np.linalg.solve(self.A, -self.H0)
        self.phi_q=v[-len(self.pos_s):]
        print(self.phi_q)
        v_mat=v[:-len(B_q)].reshape((len(self.x), len(self.y)))
        #plt.imshow(v_mat, origin='lower'); plt.colorbar(); plt.title("regular term")
        self.v=np.ndarray.flatten(v_mat)
        return(v_mat)
        
    def setup_problem(self, B_q):
        len_prob=self.xlen*self.ylen+len(self.pos_s)
        A=A_assembly(self.xlen, self.ylen)*self.D/self.h**2
        
        H0=np.zeros(len_prob)
        A=np.hstack((A, np.zeros((A.shape[0], len(self.pos_s)))))
        A=np.vstack((A,np.zeros((len(self.pos_s),A.shape[1]))))
        A=self.setup_boundary_zero_Dirich(A)
        self.A=A

        H0[-len(self.s_blocks):]=B_q
        A[-len(self.s_blocks),:]=0
        #pdb.set_trace()
        pos_s=np.arange(len(self.x)*len(self.y), A.shape[0])
        A[pos_s,pos_s]=1/self.K_0
        A[pos_s,self.s_blocks]=1

        
        c=0
        for i in self.pos_s:
            arr=np.delete(np.arange(len(self.pos_s)),c)
            d=0
            pos_s0=len(self.x)*len(self.y)
            for j in arr:
                self.A[ pos_s0+c,pos_s0+j] += Green(self.pos_s[j], i, self.Rv)
                d+=1
            c+=1
        
        self.H0=H0
        self.A=A
                
    def setup_boundary(self,A):
        for i in np.ndarray.flatten(self.boundary):
            cord=pos_to_coords(self.x, self.y, i)
            A[i,:]=0
            A[i,i]=1
            for c in range(len(self.pos_s)):
                A[i, -len(self.s_blocks)+c]=Green(cord, self.pos_s[c], self.Rv) 
        return(A)
    
    def setup_boundary_zero_Dirich(self, A):
        """Translates the zero Dirich into a Neuman BC for the SS problem"""
        for i in np.ndarray.flatten(self.boundary):
            cord=pos_to_coords(self.x, self.y, i)
            for c in range(len(self.pos_s)):
                A[i, -len(self.s_blocks)+c]-=2*Green(cord, self.pos_s[c], self.Rv)*self.D/self.h**2 
            A[i,i]-=2*self.D/self.h**2 
        return(A)
    
    def reconstruct(self, v_sol, phi_q):
        
        x,y=self.x, self.y
        phi=np.zeros(len(x)*len(y))
        for i in range(len(x)):
            for j in range(len(y)):
                dis_pos=j*len(x)+i
                cords=np.array([x[i], y[j]])
                g=0
                for c in range(len(self.pos_s)):
                    s=self.pos_s[c]
                    g+=Green(s,cords, self.Rv)*phi_q[c] if np.linalg.norm(s-cords) > self.Rv else 0
                phi[dis_pos]=v_sol[dis_pos]+g
        self.phi=phi
        return(phi.reshape(len(y), len(x)))
    
    def reconstruct_inf(self, phi_q, ratio):
        h=self.h/ratio
        L=self.x[-1]+self.h/2
        num=int(L/h)
        h=L/num
        x=np.linspace(h/2, L-h/2, num)
        y=x
        phi=np.zeros(len(x)*len(y))
        #pdb.set_trace()
        for i in range(len(x)):
            for j in range(len(y)):
                dis_pos=j*len(x)+i
                cords=np.array([x[i], y[j]])
                g=0
                for c in range(len(self.pos_s)):
                    s=self.pos_s[c]
                    g+=Green(s,cords, self.Rv)*phi_q[c] if np.linalg.norm(s-cords) > self.Rv else 0
                    
                phi[dis_pos]=g
        self.phi_inf=phi
        return(phi.reshape(len(y), len(x)))
    
    def get_corners(self):
        corners=np.array([0,len(self.x)-1,len(self.x)*(len(self.y)-1), len(self.x)*len(self.y)-1])
        self.corners=corners
        return(corners)
                    


def get_trans(hx, hy, pos):
    """Computes the transmissibility from the cell's center to the surface
    ARE WE CONSIDERING THE DIFFUSION COEFFICIENT IN THE GREEN'S FUNCTION
    
    WHAT HAPPENS IF THE SOURCE FALLS RIGHT ON THE BORDER OF THE CELL"""
    #pos=position of the source relative to the cell's center
    a=np.array([-hx/2,hy/2]) #north west corner
    b=np.array([hx/2,hy/2]) #north east corner
    c=np.array([hx/2,-hy/2]) #south east corner
    d=np.array([-hx/2,-hy/2]) #south west corner
    theta=np.zeros((4,2))
    for i in range(4):
        if i==0: #north
            en=np.array([0,1]) #normal to north surface
            c1=a
            c2=b
        if i==1: #south
            en=np.array([0,-1])
            c1=d
            c2=c
        if i==2: #east
            en=np.array([1,0])
            c1=c
            c2=b
        if i==3: #west
            en=np.array([-1,0])
            c1=d
            c2=a
            
        theta[i,0]=np.arccos(np.dot(en, (c1-pos)/np.linalg.norm(c1-pos)))
        theta[i,1]=np.arccos(np.dot(en, (c2-pos)/np.linalg.norm(c2-pos)))
    return((-theta[:,0]-theta[:,1])/(2*np.pi))


def assemble_array_block_trans(s_blocks, pos_s, block_ID, hx, hy,x,y):
    """This function will return the to multiply the q unknown in order to assemble the gradient 
    produced by the sources in this block
    Therefore in the final array, there will be a zero in the sources that do not lie in this block and 
    the value of the transmissibility (for the given surface) for the sources that do lie within
    
    There are four lines in the output array, comme d'habitude the first one is north, then south, east
    and west"""
    sources=np.where(s_blocks==block_ID)[0]
    p_s=pos_s[sources]
    cord_block=pos_to_coords(x,y,block_ID)
    trans_array=np.zeros((4, len(s_blocks)))
    for i in range(len(sources)):
        a=get_trans(hx, hy, p_s[i]-cord_block)
        trans_array[:,sources[i]]=a
    return(trans_array)


def get_errors(phi_SS, a_rec_final, noc_sol, p_sol,SS_phi_q, p_q, noc_q,ratio, phi_q):
    errors=[["coupling","SS" , ratio , get_L2(SS_phi_q, phi_q) , get_L2(phi_SS, a_rec_final) , get_MRE(SS_phi_q, phi_q) , get_MRE(phi_SS, a_rec_final)],
        ["coupling","Peaceman", ratio,get_L2(p_q, phi_q), get_L2(p_sol, np.ndarray.flatten(a_rec_final)), get_MRE(p_q, phi_q), get_MRE(p_sol, np.ndarray.flatten(a_rec_final))],
        ["FV","SS",1,get_L2(SS_phi_q, noc_q), get_L2(np.ndarray.flatten(phi_SS), noc_sol), get_MRE(SS_phi_q, phi_q), get_MRE(np.ndarray.flatten(phi_SS), noc_sol)],
        ["FV","Peaceman",1,get_L2(p_q, noc_q), get_L2(p_sol, noc_sol), get_MRE(p_q, phi_q), get_MRE(p_sol, noc_sol)],
        ["Peaceman","SS", 1,get_L2(SS_phi_q, p_q), get_L2(np.ndarray.flatten(phi_SS), p_sol), get_MRE(SS_phi_q, p_q), get_MRE(np.ndarray.flatten(phi_SS), p_sol)]]
    return(errors)


def Green_2D_brute_integral(a, b, normal, q_pos, t, Rv):
    """t=="G" for the gradient
    t==C for the normal G-function"""
    s=np.linspace(a, b, 100)
    L=np.linalg.norm(b-a)
    ds=L/100
    integral=0
    for i in s: #i is the point
        r=np.linalg.norm(i-q_pos)
        if r>Rv:
            if t=="G":
                er=(i-q_pos)/r
                integral-=(np.dot(er, normal))/(2*np.pi*r)*ds
            elif t=="C":
                integral+=np.log(Rv/r)/(2*np.pi)*ds
            else:
                print("WRONG FUNCTION ENTERED")
        else:
            if t=="G":
                er=(i-q_pos)/r
                integral-=(np.dot(er, normal))/(2*np.pi*Rv)*ds
            if t=="C":
                integral+=0
    return(integral)


def get_SS_validation(pos_s, Rv, h_ss, ratio, K_eff, D, L, B_q):
    SS=full_ss(pos_s, Rv, h_ss/ratio, K_eff, D, L)
    v_SS=SS.solve_problem(B_q)
    return(v_SS, SS.phi_q)

def FD_linear_interp(x_pos_relative, h):
    """returns the positions within the element to interpolate (pos) and the coefficients
    Bilinear interpolation"""
    x,y=x_pos_relative
    if x>=0:
        x1=0; x2=h/2
        if y>=0:
            y1=0; y2=h/2
            pos=np.array([4,5,1,2])
        if y<0:
            y1=-h/2; y2=0
            pos=np.array([7,8,4,5])
    elif x<0:
        x1=-h/2; x2=0
        if y>=0:
            y1=0; y2=h/2
            pos=np.array([3,4,0,1])
        if y<0:
            y1=-h/2; y2=0
            pos=np.array([6,7,3,4])
    r=np.array([(x2-x)*(y2-y),(x-x1)*(y2-y),(x2-x)*(y-y1),(x-x1)*(y-y1)])*4/h**2
    return(pos,r) 


class reconstruction_coupling(assemble_SS_2D_FD):
    """
    This class is meant to reconstruct the solution from the local solution splitting coupling method
    provided the value of the post-localization regular term (u) and the source fluxes (q)
    The pipeline will be as follows:
        1- A dual mesh is obtained where the h_dual is half of the one of the original mesh
        2- The values for the concentration ($\phi$) are retrieved at each point of the dual mesh
           The value of the concentration is needed since the values of the regular term will change 
           for different cell's for the same position. Therefore, the values of phi are retrieved and 
           from there the reconstruction can be made on each cell through the singular term on each cell
           
           The values of the concentration are calculated as the average of the contribution by each FV cell 
           to the concentration at that point.
               - For the dual mesh points that fall on the interface between two cell's, the average value of the 
                 concentration at the interface provided by each cell should be the same since phi-continuity accross
                 interfaces has been imposed through the numerical model
               - For the values at the dual mesh points that fall on the corner's of the original mesh it will be more 
                 tricky since the values of the concentration for each FV at that point are not necessarily the same. 
                 Therefore, here the averaging does make sense, and might provide a value that is different from the 
                 4 contributions
    """
    def __init__(self, ratio, pos_s, Rv, h,L, K_eff, D,directness):
        assemble_SS_2D_FD.__init__(self,pos_s, Rv, h,L, K_eff, D,directness)
        

        self.K_0=K_eff*np.pi*Rv**2
        
        #For the reconstruction a dual mesh is created:
        self.factor=1 #this number will indicate the amount of dual mesh cells that fit in a normal FV cell (normally 1 or 2)
        #the ratio must necessarily be a multiple of this factor :
        self.ratio=int(ratio//self.factor)*self.factor
        
        self.rec_u=np.zeros((len(self.y)*ratio, len(self.x)*ratio))
        self.rec_bar_S=self.rec_u.copy()
        
        self.dual_h=self.h/self.factor
        self.dual_x=np.arange(0, L+0.01*self.h, self.dual_h)
        self.dual_y=np.arange(0, L+0.01*self.h, self.dual_h)
        self.dual_boundary=get_boundary_vector(len(self.dual_x), len(self.dual_y))
    
    def solve_steady_state_problem(self, boundary_values, C_v_values):
        self.boundary_values=boundary_values
        
        self.pos_arrays()
        self.initialize_matrices()
        self.full_M=self.assembly_sol_split_problem(np.array([0,0,0,0]))
        self.S=len(self.pos_s)
        
        self.H0[-self.S:]=-C_v_values*self.K_0
        sol=np.linalg.solve(self.full_M, -self.H0)
        phi_FV=sol[:-self.S]
        phi_q=sol[-self.S:]
        self.phi_q, self.phi_FV=phi_q, phi_FV
    
    def retrieve_concentration_dual_mesh(self):
        """This function executes the 2 point of the description of this class. It will go 
        one by one through the dual mesh and it will provide a value for the concentration 
        at each point"""
        dual_conc=np.zeros((len(self.dual_y), len(self.dual_x)))
        c=1
        for i in self.dual_x[1:-1]:
            d=1
            for j in self.dual_y[1:-1]:
                blocks=get_blocks_radius(np.array([i,j]), self.h, self.x, self.y)
                value=0
                #get the real concentration at that point from each block:
                for k in blocks:
                    neigh=get_neighbourhood(self.directness, len(self.x), k)
                    #The value of the singular term will be given by:
                    bar_S=np.dot(kernel_green_neigh(np.array([i,j]), neigh, 
                                                       self.pos_s, self.s_blocks, self.Rv), self.phi_q)
                    value+=self.phi_FV[k]+bar_S
                dual_conc[d, c]=value/len(blocks)
                d+=1
            c+=1
        #boundary values
        dual_conc[0,:]=self.boundary_values[1]
        dual_conc[-1,:]=self.boundary_values[0]
        dual_conc[:,-1]=self.boundary_values[2]
        dual_conc[:,0]=self.boundary_values[3]
        
        self.dual_conc=dual_conc
        return()

    def execute_interpolation(self, phi_q, phi_FV):
        for i in range(len(self.dual_x)-1):
            #pdb.set_trace()
            for j in range(len(self.dual_y)-1):
                #This loop goes through each "dual element", where it does the interpolation
                #of the four corner values (which have already been calculated). The advantage of 
                #this class is that every ""dual element" only lies within one original element. 
                #Therefore, there is no ambiguity on the singular term to use

                #only a single block's information is necessary
                
                len_x=len(self.x)
                len_y=len(self.y)
                
                dual_block_center=np.array([self.dual_x[i], self.dual_y[j]])+self.dual_h/2
                block=coord_to_pos(self.x, self.y, dual_block_center) #the FV block whithin the dual block falls
                corner_pos=np.array([[i,j],[i,j+1],[i+1,j],[i+1,j+1]])
                corner_values=self.dual_conc[corner_pos[:,0], corner_pos[:,1]]
                
                position_corners=np.array([self.dual_x[corner_pos[:,0]], self.dual_y[corner_pos[:,1]]]).T
                
                S_c_values=np.array([])
                neigh=get_neighbourhood(self.directness, len_x, block)
                for coord_corner in position_corners:
                    #this loop goes through the position of the corners
                    G_sub=kernel_green_neigh(coord_corner, neigh, self.pos_s, self.s_blocks, self.Rv)
                    S_c_values=np.append(S_c_values, np.dot(G_sub, phi_q))
                    
                pos_x=np.arange(self.ratio/self.factor, dtype=int)+int(i*self.ratio/self.factor)
                pos_y=np.arange(self.ratio/self.factor, dtype=int)+int(j*self.ratio/self.factor)
                block_S_bar=green_to_block(phi_q, dual_block_center, self.dual_h, int(self.ratio/self.factor),
                                           neigh, self.s_blocks, self.pos_s,self.Rv)
                
                
                self.rec_bar_S=modify_matrix(self.rec_bar_S,pos_x,pos_y, block_S_bar)
                
                local_sol=bilinear_interpolation(corner_values-S_c_values , int(self.ratio/self.factor))          
                self.rec_phi_FV=modify_matrix(self.rec_phi_FV, pos_x, pos_y, local_sol)