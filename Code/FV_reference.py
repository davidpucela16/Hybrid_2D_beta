# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy.sparse.linalg import spsolve
from Module_Coupling import get_boundary_vector
from Assembly_diffusion import Lapl_2D_FD_sparse
from Small_functions import set_TPFA_Dirichlet
import pdb

class FV_validation():
    """Creates a cartesian FV model"""
    def __init__(self,L, num_cells, pos_s, phi_j, D, K_eff, Rv, BC_type, BC_value, *Peaceman):
        """This class is meant as validation, therefore we consider the Peaceman 
        correction directly
        num_cells represents the quantity of cells in each direction
        
        The characteristic length scale for the adimensionalization is h (the 
                                        characteristic size of the grid)
        
        ONLY HANDLES DIRICHLET AND NEUMANN BCS"""
        h=L/num_cells
        K_0=K_eff*np.pi*Rv**2
        self.K_0=K_0
        if np.sum(Peaceman):
            #Peaceman correction:
            R=1/(1/K_0+np.log(0.2*h/Rv)/(2*np.pi*D))
        else:
            R=K_0
        #R=K_0
        print("Peaceman Production Index PI=", R)
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
        self.set_up_system(BC_type, BC_value)
        self.L=L
        
    def set_up_system(self, BC_type, BC_value):
        x=self.x
        y=self.y
        R=self.R
        phi_j=self.phi_j
        pos_s=self.pos_s

        A=Lapl_2D_FD_sparse(len(x), len(y))
        A_virgin=A.copy()
        self.A_virgin=A_virgin
        bound_vect=get_boundary_vector(len(x), len(y))
        #set dirichlet
        H0=np.zeros(len(x)*len(y))
        for i in range(4):
            if BC_type[i]=='Dirichlet':
                H0,A=set_TPFA_Dirichlet(BC_type, BC_value,A, self.D, bound_vect, H0)
    
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
        solution=spsolve(self.A, -self.H0)
        self.unc_solution=solution.copy()
        return(solution)
    
    def get_q_linear(self):
        #for linear system
        return((self.phi_j-self.unc_solution[self.s_blocks])*self.R)
    
    def get_q_metab(self):
        return((self.phi_j-self.phi_metab[-1,self.s_blocks])*self.R)
    
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
        iterations=0
        while rerr_q[-1]>0.00005 and iterations<500:
            #pdb.set_trace()
            Jacobian=self.A+np.diag(self.get_partial_Im_phi(phi[-1]))
            
            inc=np.linalg.solve(Jacobian, -self.get_F(phi[-1]))*stabilization
            #inc=self.get_F(phi)
            phi=np.concatenate((phi, np.array([phi[-1]+inc])), axis=0)
            q=np.concatenate([q,[self.R*(self.phi_j-phi[-1,self.s_blocks])]], axis=0)
            rerr_q=np.append(rerr_q, np.max(np.abs(q[-1]-q[-2]))/np.max(np.abs(q[-1])))
            print("Residual q FV: ",rerr_q[-1])
            iterations+=1
        self.phi_metab=phi
        self.q_metab=q[-1]
        self.accumul_q=q
        return(self.phi_metab)
    
    def get_corr_array(self, *non_linear):
        """returns by default the corr array for the linear prob"""
        FV_corr_array=np.zeros(len(self.x)*len(self.y))
        if non_linear:
            corr=-self.q_metab*np.log(0.342/0.2)/(2*np.pi)
            FV_corr_array[self.s_blocks]+=corr
        else:
            corr=-self.get_q_linear()*np.log(0.342/0.2)/(2*np.pi)
            FV_corr_array[self.s_blocks]+=corr
        return(FV_corr_array)
    
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