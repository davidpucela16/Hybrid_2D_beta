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


##############################################################################
##############################################################################
# GREEN'S FUNCTIONS:

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

#
def Sampson_grad_Green(a,b, q_pos,normal, Rv):
    f0=grad_Green(q_pos, a, normal,Rv)
    f1=grad_Green(q_pos, a+(b-a)/2, normal,Rv)
    f2=grad_Green(q_pos, b, normal,Rv)
    L=np.linalg.norm(b-a)
    return(L*(f0+4*f1+f2)/6)

#
def Sampson_Green(a,b, q_pos, Rv):
    f0=Green(q_pos, a, Rv)
    f1=Green(q_pos, a+(b-a)/2, Rv)
    f2=Green(q_pos, b, Rv)
    L=np.linalg.norm(b-a)
    return(L*(f0+4*f1+f2)/6)


def grad_Green_norm(normal, pos1, pos2):
    """From the relative position of the source (with respect to the center of the block)
    rel_calc_point is the point where the gradient is calculated"""
    r=pos1-pos2
    sc=np.dot(r, normal)
    return(sc/(2*np.pi*np.linalg.norm(r)**2))

def block_grad_Green_norm_array(pos_s,s_blocks,  block_ID, pos_calcul, norm):
    """returns the array that will multiply the sources of the block_ID to obtain
    the gradient of the given block's singular term"""
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
    
#
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

#
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


##############################################################################
##############################################################################

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
    
def pos_to_coords(x, y, ID):
    xpos=ID%len(x)
    ypos=ID//len(x)
    return(np.array([x[xpos], y[ypos]]))

def coord_to_pos(x,y, coord):
    """Returns the block_ID closest to the coordinates"""
    pos_x=np.argmin((coord[0]-x)**2)
    pos_y=np.argmin((coord[1]-y)**2)
    return(int(pos_x+pos_y*len(x)))

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





def v_linear_interpolation(cell_center, x_pos, h):
    """this function is designed to give the coefficients that will multiply the values on the faces 
    of the cell to obtain a linear interpolation"""
    d=np.zeros(4)
    for i in range(4):
        if i==0:#a
            e=cell_center+np.array([-h/2,-h/2])
        if i==1:#b
            e=cell_center+np.array([-h/2,h/2])
        if i==2:#c
            e=cell_center+np.array([h/2,-h/2])
        if i==3:#d
            e=cell_center+np.array([h/2,h/2])
        d[i]=np.linalg.norm(x_pos-e)

    if np.all(d):
        alpha=1/np.sum(1/d)
        return(alpha/d)
    else: 
        ar=np.zeros(4)
        ar[np.where(d==0)]=1
        return(ar)
        


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





def get_boundary_vector(xlen, ylen):
    #3- Set up the boundary arrays
    north=np.arange(xlen* (ylen-1), xlen* ylen)
    south=np.arange(xlen)
    west=np.arange(0,xlen* ylen, xlen)
    east=np.arange(xlen-1, xlen* ylen,xlen)
    return(np.array([north, south, east, west]))
def set_TPFA_Dirichlet(Dirichlet,operator,  D, boundary_array, H0):
    """Translates properly the Dirichlet BC to a Neumann BC in the Laplacian operator
    Does this work properly with the solution splitting algorithm??"""
    c=0
    for i in boundary_array:
        C=2*D
        
        operator[i,i]-=C
        H0[i]+=C*Dirichlet[c]

        c+=1
    return(H0, operator)
    


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
            
    
    
    
class full_ss():
    """Class to solve the solution split problem with point sources in a 2D domain"""
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
                    



def get_L2(validation, phi):
    """Relative L2 norm"""
    L2=np.sum(((validation-phi)/validation)**2)**0.5
    return(L2)

def get_L1(validation, phi):
    L1=np.sum(np.abs(validation-phi))/(np.sum(np.abs(validation))*len(phi))
    return(L1)    

def get_MRE(validation, phi):
    MRE=np.sum((np.abs(validation-phi))/np.abs(validation))/len(phi)
    return(MRE)

def get_MAE(validation, phi):
    """Mean absolute error"""
    MRE=np.sum(np.abs(validation-phi))/len(phi)
    return(MRE)

def A_assembly_Dirich(cxlen, cylen):
    """This function assembles the laplacian operator for cartesian coordinates in 2D
    - It does not take into account the discretization size h, therefore, it must be homogeneous
    - For the same reason, if the discretization size is different than 1 there will be a factor to 
      multiply/divide the operator
    - As well, there is no diffusion coefficient considered
    
    INPUT -> the x and y length
    OUTPUT -> FD Laplacian operator
    """
    A=np.zeros([cxlen* cylen, cxlen*cylen])
    north=np.arange(cxlen* (cylen-1), cxlen* cylen)
    south=np.arange(cxlen)
    west=np.arange(0,cxlen* cylen, cxlen)
    east=np.arange(cxlen-1, cxlen* cylen,cxlen)
    
    boundary=np.concatenate([north, south, east, west])
    
    corners=np.array([0,cxlen-1,cxlen*(cylen-1), cxlen*cylen-1])
    
    for i in range(cxlen*cylen):
        if i not in boundary:
            A[i,i]-=4
            A[i,i+1]+=1
            A[i,i-1]+=1
            A[i,i+cxlen]+=1
            A[i,i-cxlen]+=1
            
        else:        
            if i in north:
                others=[1,-1,-cxlen]
            if i in south:
                others=[1,-1,cxlen]
            if i in east:
                others=[cxlen, -cxlen, -1]
            if i in west:
                others=[cxlen, -cxlen, 1]
                
            if i==0:
                #corner sudwest
                others=[1,cxlen]
            if i==cxlen-1:
                #sud east
                others=[-1,cxlen]
            if i==cxlen*(cylen-1):
                #north west
                others=[1,-cxlen]
            if i==cxlen*cylen-1:
                others=[-1,-cxlen]
            
            A[i,i]=-4
            for n in others:
                A[i,i+n]=1
        
    return(A)


def A_assembly(cxlen, cylen):
    """This function assembles the laplacian operator for cartesian coordinates in 2D
    - It does not take into account the discretization size h, therefore, it must be homogeneous
    - For the same reason, if the discretization size is different than 1 there will be a factor to 
      multiply/divide the operator
    - As well, there is no diffusion coefficient considered
    
    INPUT -> the x and y length
    OUTPUT -> FD Laplacian operator
    """
    A=np.zeros([cxlen* cylen, cxlen*cylen])
    north=np.arange(cxlen* (cylen-1), cxlen* cylen)
    south=np.arange(cxlen)
    west=np.arange(0,cxlen* cylen, cxlen)
    east=np.arange(cxlen-1, cxlen* cylen,cxlen)
    
    boundary=np.concatenate([north, south, east, west])
    
    corners=np.array([0,cxlen-1,cxlen*(cylen-1), cxlen*cylen-1])
    
    for i in range(cxlen*cylen):
        if i not in boundary:
            A[i,i]-=4
            A[i,i+1]+=1
            A[i,i-1]+=1
            A[i,i+cxlen]+=1
            A[i,i-cxlen]+=1
            
        else:        
            if i in north:
                others=[1,-1,-cxlen]
            if i in south:
                others=[1,-1,cxlen]
            if i in east:
                others=[cxlen, -cxlen, -1]
            if i in west:
                others=[cxlen, -cxlen, 1]
                
            if i==0:
                #corner sudwest
                others=[1,cxlen]
            if i==cxlen-1:
                #sud east
                others=[-1,cxlen]
            if i==cxlen*(cylen-1):
                #north west
                others=[1,-cxlen]
            if i==cxlen*cylen-1:
                others=[-1,-cxlen]
            
            A[i,i]=-len(others)
            for n in others:
                A[i,i+n]=1
        
    return(A)

def Lapl_2D_FD_sparse(cxlen, cylen):
    """This function assembles the laplacian operator for cartesian coordinates in 2D
    - It does not take into account the discretization size h, therefore, it must be homogeneous
    - For the same reason, if the discretization size is different than 1 there will be a factor to 
      multiply/divide the operator
    - As well, there is no diffusion coefficient considered
    
    INPUT -> the x and y length
    OUTPUT -> FD Laplacian operator
    """
    north=np.arange(cxlen* (cylen-1), cxlen* cylen)
    south=np.arange(cxlen)
    west=np.arange(0,cxlen* cylen, cxlen)
    east=np.arange(cxlen-1, cxlen* cylen,cxlen)
    
    boundary=np.concatenate([north, south, east, west])
    
    corners=np.array([0,cxlen-1,cxlen*(cylen-1), cxlen*cylen-1])
    
    row=np.array([], dtype=int)
    col=np.array([], dtype=int)
    data=np.array([])
    
    for i in range(cxlen*cylen):
        if i not in boundary:
            c=np.array([0,1,-1,cxlen, -cxlen])
        else:        
            if i in north:
                c=np.array([0,1,-1,-cxlen])
            if i in south:
                c=np.array([0,1,-1,cxlen])
            if i in east:
                c=np.array([0,cxlen, -cxlen, -1])
            if i in west:
                c=np.array([0,cxlen, -cxlen, 1])
                
            if i==0:
                #corner sudwest
                c=np.array([0,1,cxlen])
            if i==cxlen-1:
                #sud east
                c=np.array([0,-1,cxlen])
            if i==cxlen*(cylen-1):
                #north west
                c=np.array([0,1,-cxlen])
            if i==cxlen*cylen-1:
                c=np.array([0,-1,-cxlen])

        d=np.ones(len(c))
        d[0]=-len(c)+1
        row=np.concatenate((row, np.zeros(len(c))+i))
        col=np.concatenate((col,c+i))
        data=np.concatenate((data, d))
    
    operator=sp.sparse.csc_matrix((data, (row, col)), shape=(cylen*cxlen, cxlen*cylen))
    return(operator)






class assemble_SS_2D_FD():
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
    
    def set_Dirichlet(self, values):
        """Sets the Dirichlet BC translated to Neumann BC on the solution 
        splitting operator
        
        We could adimensionalize all the positions but it is not worth the hassel I think"""
        north, sout, east, west=self.boundary
        v_n, v_s, v_e, v_w=values
        c=0
        #pdb.set_trace()
        for b in self.boundary:
            for k in b:
                normal=np.array([[0,1],[0,-1],[1,0],[-1,0]])[c]

                pos_k=pos_to_coords(self.x, self.y, k)
                pos_bound=pos_k+normal*self.h/2
                
                perp_dir=np.array([[0,-1],[1,0]]).dot(normal)
                
                pos_a=pos_k+(perp_dir+normal)*self.h/2
                pos_b=pos_k+(-perp_dir+normal)*self.h/2
                #pdb.set_trace()
                N_k=get_neighbourhood(self.directness, len(self.x), k)
                s_IDs=np.arange(len(self.pos_s))[np.in1d(self.s_blocks, N_k)]

                #The division by h is because the kernel calculates the integral, what we 
                #need is an average value per full cell
                kernel=kernel_integral_Green_face(self.pos_s, s_IDs, pos_a,pos_b,self.Rv)/self.h
                
                self.A_matrix[k,k]-=2*self.D
                self.b_matrix[k,:]-=kernel*2*self.D
                self.H0[k]+=values[c]*2*self.D
            
            c+=1
    
    def assembly_sol_split_problem(self, values_Dirich):
        """Main function that assembles the matrices one by one.
        The only functions called before this one are pos_arrays to get the relevant information
        stored in the very important arrays (such as s_blocks), and initialize matrices."""
        #First of all it is needed to remove the source_blocks from the main matrix
        #pdb.set_trace()
        self.A_matrix=A_assembly(len(self.x), len(self.y))/self.D
        self.set_Dirichlet(values_Dirich)
        
        self.b_matrix=self.assemble_b_matrix()/self.D
        self.assemblec_c_d_matrix()
        
        Up=np.concatenate((self.A_matrix, self.b_matrix), axis=1)
        Down=np.concatenate((self.c_matrix, self.d_matrix), axis=1)
        
        self.Up=Up
        self.Down=Down
        
        LIN_MAT=np.concatenate((Up,Down), axis=0)
        self.LIN_MAT=LIN_MAT
        return(LIN_MAT)
    
    def assemble_b_matrix(self):
        
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
            #pdb.set_trace()
            for i in neigh:
                m=k+i
                m_neigh=get_neighbourhood(self.directness, len(self.x), m)
                k_neigh=get_neighbourhood(self.directness, len(self.x), k)
                unc_k_m=get_uncommon(k_neigh, m_neigh)
                unc_m_k=get_uncommon(m_neigh, k_neigh)
                
                pos_k=pos_to_coords(self.x, self.y, k)
                pos_m=pos_to_coords(self.x, self.y, m)
                
                normal=(pos_m-pos_k)/self.h
                
                #sources in the neighbourhood of m that are not in the neigh of k 
                Em=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks,unc_m_k)]
                
                #sources in the neighbourhood of k that are not in the neigh of m
                Ek=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks,unc_k_m)]
                
                perp_dir=np.array([[0,-1],[1,0]]).dot(normal)
                
                pos_a=pos_k+(perp_dir+normal)*self.h/2
                pos_b=pos_k+(-perp_dir+normal)*self.h/2
                
                #pdb.set_trace()
                kernel_m=kernel_integral_grad_Green_face(self.pos_s, Em, pos_a,pos_b,normal, self.Rv)/2
                kernel_m+=kernel_integral_Green_face(self.pos_s, Em, pos_a,pos_b,self.Rv)/self.h
                
                kernel_k=kernel_integral_grad_Green_face(self.pos_s, Ek, pos_a,pos_b,normal, self.Rv)/2
                kernel_k+=kernel_integral_Green_face(self.pos_s, Ek, pos_a,pos_b,self.Rv)/self.h
                
                self.b_matrix[k,:]+=kernel_m
                self.b_matrix[k,:]-=kernel_k
                
                c+=1
        return(self.b_matrix)
    
    def assemblec_c_d_matrix(self):
        
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
                    #
                    c_kernel[blocks]=v_linear_interpolation(coord_blocks, self.pos_s[i], self.h)
                    self.c_matrix[i,:]=c_kernel #regular term 
                    self.d_matrix[i,i]=1/self.K_0[i]
                    self.d_matrix[i]+=d_kernel
                    for j in other:
                        """Goes through each of the other sources (j != i)"""
                        self.d_matrix[i,j]+=Green(self.pos_s[i],self.pos_s[j], self.Rv[j])/self.D
# =============================================================================
#                 self.c_matrix[i,k]=1 #regular term 
#                 self.d_matrix[i,i]=1/self.K_0[i] 
#                 print("old version flux estimation")
#                 for j in other:
#                     """Goes through each of the other sources (j != i)"""
#                     self.d_matrix[i,j]+=Green(self.pos_s[i],self.pos_s[j], self.Rv[j])
# =============================================================================
    

        return()
    def solve_problem(self, dirichlet, C_v_array):
        """Function that after initialisation solves the entire problem"""
        self.pos_arrays() #Creates the arrays with the geometrical position of the sources 
        self.initialize_matrices() #Creates the matrices (Jacobian, indep term....)
        
        #Assembly of the Laplacian and other arrays for the linear system
        LIN_MAT=self.assembly_sol_split_problem(dirichlet) 
        self.set_intravascular_terms(C_v_array) #Sets up the intravascular concentration as BC 
        sol=np.linalg.solve(LIN_MAT, -self.H0)
        phi_FV=sol[:-self.S].reshape(len(self.x), len(self.y))
        phi_q=sol[-self.S:]
        
        #initial guesses
        self.phi_FV=phi_FV
        self.phi_q=phi_q
        
        self.phi_bar=C_v_array-phi_q/self.K_0
        return()
        
    
    

    def get_corners(self):
        corners=np.array([0,len(self.x)-1,len(self.x)*(len(self.y)-1), len(self.x)*len(self.y)-1])
        self.corners=corners
        return(corners)
    


class real_NN_rec():
    def __init__(self,x, y, phi_FV, pos_s, s_blocks, phi_q, ratio, h, directness, Rv):
        self.x=x
        self.y=y
        self.phi_FV=phi_FV
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
        
        for i in range(len(r_y)):
            for j in range(len(r_x)):
                rec[i,j]=phi_FV[self.get_block(np.array([r_x[j],r_y[i]]))]
        self.rec=rec
        
    def get_block(self,pos):
        row=np.argmin(np.abs(self.y-pos[1]))
        col=np.argmin(np.abs(self.x-pos[0]))
        return(row*len(self.x)+col)
    
    def add_singular(self, directness):
        #THIS FUNCTION DEFINETELY DOES NOT WORK AT THE MOMENT
        print("THIS FUNCTION DEFINETELY DOES NOT WORK AT THE MOMENT")
        
        #pdb.set_trace()
        Rv=self.Rv
        r_x=self.r_x
        r_y=self.r_y
        rec=np.zeros((len(r_y), len(r_x)))
        for i in range(len(r_y)):
            for j in range(len(r_x)):
# =============================================================================
#                 if self.get_block(np.array([r_x[j],r_y[i]]))==36:
#                     pdb.set_trace()
# =============================================================================
                neigh=get_neighbourhood(directness, len(self.x), self.get_block(np.array([r_x[j],r_y[i]])))
                Ens=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, neigh)]#sources in the neighbourhood
                
                for k in Ens:
                    """Goes through all the sources in the neighbourhood"""
                    rec[i,j]+=Green(self.pos_s[k], np.array([r_x[j],r_y[i]]), Rv[j])*self.phi_q[k]
        return(rec)

    
    

def get_SS_validation(pos_s, Rv, h_ss, ratio, K_eff, D, L, B_q):
    SS=full_ss(pos_s, Rv, h_ss/ratio, K_eff, D, L)
    v_SS=SS.solve_problem(B_q)
    return(v_SS, SS.phi_q)
    

class FV_validation():
    def __init__(self,L, num_cells, pos_s, phi_j, D, K_eff, Rv, dirichlet_array):
        """This class is meant as validation, therefore we consider the Peaceman 
        correction directly
        num_cells represents the quantity of cells in each direction
        
        The characteristic length scale for the adimensionalization is h (the 
                                        characteristic size of the grid)"""
        h=L/num_cells
        K_0=K_eff*np.pi*Rv**2
        self.K_0=K_0
        #Peaceman correction:
        #R=1/(1/K_0+np.log(0.2*h/Rv)/(2*np.pi*D))
        R=K_0
        print("R:", R)
        x=np.linspace(h/2, L-h/2, num_cells)
        y=x
        self.D=D
        self.x=x
        self.y=y
        self.R=R
        self.phi_j=phi_j
        self.h=h
        self.Rv=Rv
        self.pos_s=pos_s
        self.set_up_system(dirichlet_array)
        self.L=L
        
    def set_up_system(self, dirichlet_array):
        x=self.x
        y=self.y
        R=self.R
        phi_j=self.phi_j
        pos_s=self.pos_s
        h=self.h
        D=self.D

        A=Lapl_2D_FD_sparse(len(x), len(y))
        A_virgin=A.copy()
        self.A_virgin=A_virgin
        #set dirichlet
        H0,A=set_TPFA_Dirichlet(dirichlet_array,A, self.D, get_boundary_vector(len(x), len(y)), np.zeros(len(x)*len(y)))
        #Set sources
        s_blocks=np.array([], dtype=int)
        c=0
        s_blocks=np.array([], dtype=int)
        for i in pos_s:
            x_pos=np.argmin(np.abs(i[0]-x))
            y_pos=np.argmin(np.abs(i[1]-y))
            
            block=y_pos*len(x)+x_pos
            A[block, block]-=R[c]
            H0[block]=phi_j[c]*R[c]
            s_blocks=np.append(s_blocks, block)
            c+=1
        self.s_blocks=s_blocks
        self.H0=H0
        self.A=A
        
    
        
    def solve_linear_system(self):
        return(spsolve(self.A, -self.H0))
    
    def get_q(self,phi):
        return((self.phi_j-phi[self.s_blocks])*self.R)
    
    #NOW THE FUNCTIONS FOR THE NON LINEAR PROBLEM
    def solve_non_linear_system(self, phi_0,M, stabilization):
        self.phi_0=phi_0
        self.M=M
        #Definition of the array that will contain the metabolism at each iteration
        self.F_2=np.zeros(len(self.x)*len(self.y))
        #initial guess
        #pdb.set_trace()
        phi=self.solve_linear_system()
        q=np.array([self.R*(self.phi_j-phi[self.s_blocks])])
        phi=np.array([phi])
        rerr_q=np.array([1])
        while rerr_q[-1]>0.00005:
            #pdb.set_trace()
            Jacobian=self.A+np.diag(self.get_partial_Im_phi(phi[-1]))
            
            inc=np.linalg.solve(Jacobian, -self.get_F(phi[-1]))*stabilization
            #inc=self.get_F(phi)
            phi=np.concatenate((phi, np.array([phi[-1]+inc])), axis=0)
            q=np.concatenate([q,[self.R*(self.phi_j-phi[-1,self.s_blocks])]], axis=0)
            rerr_q=np.append(rerr_q, np.max(np.abs(q[-1]-q[-2]))/np.max(np.abs(q[-1])))
            print("Residual q FV: ",rerr_q[-1])
        self.phi=phi
        self.q=q
        return(self.phi)
    
    def get_corr_array(self, phi):
        return(-self.get_q(phi)*np.log(0.342/0.2)/(2*np.pi))
    
    def get_partial_Im_phi(self, phi):
        """Calculates ONLY THE NON LINEAR part of the Jacobian"""
        q=self.R*(self.phi_j-phi[self.s_blocks])
        #pdb.set_trace()
        Corr_array=np.zeros(len(self.x)*len(self.y))
        Corr_array[self.s_blocks]=-q*np.log(0.342/0.2)/(2*np.pi*self.D)
        
        non_lin_Jac=-self.M*self.phi_0*(phi+Corr_array+self.phi_0)**-2
        return(non_lin_Jac)
    
    def get_F(self, phi):
        q=self.R*(self.phi_j-phi[self.s_blocks])
        Corr_array=np.zeros(len(self.x)*len(self.y))
        Corr_array[self.s_blocks]=-q*np.log(0.342/0.2)/(2*np.pi*self.D)
        
        F_2=self.M*(1-self.phi_0/(phi+Corr_array+self.phi_0))*self.h**2 #metabolism
        F_1=sp.sparse.csc_matrix.dot(self.A, phi)+self.H0 #steady state system 
                                                 #linear syst
        self.Corr_array=Corr_array
        self.F_2=np.vstack((self.F_2, F_2))
        return(F_1-F_2)
        








class non_linear_metab(assemble_SS_2D_FD):
    def __init__(self,pos_s, Rv, h,L, K_eff, D,directness):
        assemble_SS_2D_FD.__init__(self,pos_s, Rv, h,L, K_eff, D,directness)
    def solve_linear_prob(self, dirichlet_array,C_v_array):
        """Solves the problem without metabolism, necessary to provide an initial guess
        DIRICHLET ARRAY ARE THE VALUES OF THE CONCENTRATION VALUE AT EACH BOUNDARY, THE CODE
        IS NOT YET MADE TO CALCULATE ANYTHING OTHER THAN dirichlet_array=np.zeros(4)"""
        S=len(self.pos_s)
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
        self.phi_FV=phi_FV
        self.phi_q=phi_q
    
    def reconstruct_field_kk(self, phi_FV, q):
        G_rec=from_pos_get_green(self.x, self.y, self.pos_s, self.Rv ,np.array([0,0]), self.directness, self.s_blocks)
        return((np.dot(G_rec[:,:],q)+np.ndarray.flatten(phi_FV)).reshape(len(self.y), len(self.x)))

    def assemble_it_matrices_Sampson(self, phi_FV, q):
        """Assembles the important matrices for the iterative problem through
        Sampson integration within each cell. The matrices:
            - Upper Jacobian: [\partial_{u} I_m  \partial_{q} I_m]"""
        phi_0=self.phi_0
        M=self.M
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
        I_m=np.zeros(self.phi_FV.size)
        rec_sing=np.zeros(self.phi_FV.size)
        part_Im_phi_FV=np.zeros(self.phi_FV.size)  #\frac{\partial I_m}{\partial u}
        part_Im_q=np.zeros((self.phi_FV.size, len(self.pos_s)))
        for k in range(self.phi_FV.size):
            #pdb.set_trace()
            for i in range(len(w_i)):
                
                pos_xy=pos_to_coords(self.x, self.y, k)+corr[i]
                kernel_G=kernel_green_neigh(pos_xy, get_neighbourhood(self.directness, len(self.x), k),
                                            self.pos_s, self.s_blocks, self.Rv)
                SL=np.dot(kernel_G,q)
                rec_sing[k]+=w_i[i]*SL
                
                I_m[k]+=phi_0*w_i[i]/(phi_0+SL+phi_FV[k])
                part_Im_phi_FV[k]-=phi_0*w_i[i]/(phi_0+SL+phi_FV[k])**2
                for l in range(len(self.pos_s)):
                    #part_Im_q[k,l]+=part_Im_u[k]*kernel_G[l]
                    part_Im_q[k,l]-=kernel_G[l]*phi_0*w_i[i]/(phi_0+SL+phi_FV[k])**2
    
        self.I_m=I_m
        self.rec_sing=rec_sing
        self.part_Im_q=part_Im_q
        self.part_Im_phi_FV=part_Im_phi_FV
        return()
 

        

    def Full_Newton(self, u,q, rel_error,M, phi_0):
        self.M=M
        self.phi_0=phi_0
        rl=np.array([1])
        phi=np.array([np.concatenate((u,q))]) #This is the array where the arrays of u through iterations will be kept
        S=self.S
        while np.abs(rl[-1])>rel_error:
            
            self.assemble_it_matrices_Sampson(phi[-1,:-S], phi[-1,-S:])
            metab=self.M*(1-self.I_m)*self.h**2
            Jacobian=np.concatenate((np.diag(self.part_Im_phi_FV)+self.A_matrix, 
                                     self.part_Im_q+self.b_matrix), axis=1)
            

            Jacobian=np.concatenate((Jacobian, self.Down)) 
            #Compute the new value of u:
            F=np.dot(self.LIN_MAT, phi[-1]) +self.H0 - np.pad(metab, [0,self.S])
            
            inc=np.linalg.solve(Jacobian, -F)
            phi=np.concatenate((phi, np.array([phi[-1]+inc])))
            self.u=phi[-1,:-S]
            self.q=phi[-1,-S:]
            rl=np.append(rl,np.sum(np.abs(inc))/len(inc))
            print("residual", np.sum(np.abs(inc))/len(inc))
        self.phi=phi
        self.u=self.phi[-1,:-S]
        self.q=self.phi[-1,-S:]
        return(phi[-1])





#From post module:

        
def get_4_blocks(position, tx, ty, th):
    #pdb.set_trace()
    blocks_x=np.where(np.abs(tx-position[0]) < th*1.01)[0]
    blocks_y=np.where(np.abs(ty-position[1]) < th*1.01)[0]*len(tx)
    blocks=np.array([blocks_y[0]+blocks_x[0], blocks_y[1]+blocks_x[0],blocks_y[0]+blocks_x[1],blocks_y[1]+blocks_x[1]])
    
    return(blocks)






