
import numpy as np 

def pos_to_coords(x, y, ID):
    xpos=ID%len(x)
    ypos=ID//len(x)
    return(np.array([x[xpos], y[ypos]]))

def coord_to_pos(x,y, coord):
    """Returns the block_ID closest to the coordinates"""
    pos_x=np.argmin((coord[0]-x)**2)
    pos_y=np.argmin((coord[1]-y)**2)
    return(int(pos_x+pos_y*len(x)))

import matplotlib.pylab as pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas 
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from Small_functions import coord_to_pos, pos_to_coords

def plot_sketch(x, y, directness, h, pos_s, L, directory, *title):

    vline=(y[1:]+x[:-1])/2
    fig, ax=plt.subplots()
    array_of_colors=['cyan','lightgreen']
    c=0
    if len(pos_s)<4:
        for i in pos_s:
            side=(directness+0.5)*h*2
            lower_corner=pos_to_coords(x,y,coord_to_pos(x,y,i-directness*np.array([h,h])))-np.array([h,h])/2
            ax.add_patch(Rectangle(lower_corner, side, side,
                         edgecolor = array_of_colors[c],
                         facecolor = array_of_colors[c],
                         fill=True,
                         lw=5, zorder=0))
            c+=1
    plt.scatter(pos_s[:,0], pos_s[:,1], color='r')
    if title:
        plt.title(title[0])
    for xc in vline:
        plt.axvline(x=xc, color='k', linestyle='--')
    for xc in vline:
        plt.axhline(y=xc, color='k', linestyle='--')
    
    plt.xlim([0,L])
    plt.ylim([0,L])
    plt.ylabel("y ($\mu m$)")
    plt.xlabel("x ($\mu m$)")
    
    plt.savefig(directory + "/sketch.png", transparent=True)
    plt.show()



def v_linear_interpolation(cell_center, x_pos, h):
    """this function is designed to give the coefficients that will multiply the values on the vertices 
    of the cell to obtain a linear interpolation"""
    """Everything must be given in the form of an np.array, in the following order:
    (0,0), (0,1), (1,0), (1,1)"""
    a=cell_center-np.array([h,h])/2

    i= (x_pos[0]-a[0])/h#relative x position
    j= (x_pos[1]-a[1])/h#relative y position
    A=np.array([[1,-1,-1,1],[0,0,1,-1],[0,1,0,-1],[0,0,0,1]])
    weights=A.dot(np.array([1,i,j,i*j]))
    return(weights)


def get_boundary_vector(xlen, ylen):
    #3- Set up the boundary arrays
    north=np.arange(xlen* (ylen-1), xlen* ylen)
    south=np.arange(xlen)
    west=np.arange(0,xlen* ylen, xlen)
    east=np.arange(xlen-1, xlen* ylen,xlen)
    return(np.array([north, south, east, west]))

def set_TPFA_Dirichlet(BC_type, BC_value,operator,  D, boundary_array, H0):
    """Translates properly the Dirichlet BC to a Neumann BC in the Laplacian operator
    Does this work properly with the solution splitting algorithm??
    
    Nope, only works with FV method obviously"""
    c=0
    for i in boundary_array:
        if BC_type[c]=='Dirichlet':
            C=2*D
            
            operator[i,i]-=C
            H0[i]+=C*BC_value[c]
        c+=1
    return(H0, operator)


def get_L2(validation, phi):
    """Relative L2 norm"""
    L2=np.sum(((validation-phi)/validation)**2)**0.5
    return(L2)

def get_L1(validation, phi):
    L1=np.sum(np.abs(validation-phi))/(np.sum(np.abs(validation))*phi.size)
    return(L1)    

def get_MRE(validation, phi):
    MRE=np.sum((np.abs(validation-phi))/np.abs(validation))/phi.size
    return(MRE)

def get_MAE(validation, phi):
    """Mean absolute error"""
    MRE=np.sum(np.abs(validation-phi))/phi.size
    return(MRE)


def get_4_blocks(position, tx, ty, th):
    #pdb.set_trace()
    blocks_x=np.where(np.abs(tx-position[0]) < th*1.01)[0]
    blocks_y=np.where(np.abs(ty-position[1]) < th*1.01)[0]*len(tx)
    blocks=np.array([blocks_y[0]+blocks_x[0], blocks_y[1]+blocks_x[0],blocks_y[0]+blocks_x[1],blocks_y[1]+blocks_x[1]])
    
    return(blocks)

def get_position_cartesian_sources(x,y,pos_s):
    "Returns the position within the cartesian grid of the center of the source"
    p_x,p_y=np.zeros(0,dtype=int), np.zeros(0,dtype=int)
    for i in pos_s:
        p_y=np.append(p_y, np.argmin((y-i[1])**2))
        p_x=np.append(p_x, np.argmin((x-i[0])**2))
    return(p_x,p_y)
    
    
    
    
    