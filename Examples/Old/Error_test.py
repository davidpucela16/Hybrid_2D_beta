#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:36:57 2022

@author: pdavid

THIS TESTS ARE MEANT TO PROVIDE THE ERRORS FOR A CONVERGENCE GRAPH
"""

import numpy as np
import matplotlib.pyplot as plt

#For a single source:
num_cells=np.array([3,7,15])
h=2/num_cells

L2_norm_q=np.array([0.14,0.1,0.05])*10**-2

#For the multiple sources of the figure

#Rv approx 1e-3
gjerde_h=np.array([1/4,1/8,1/16,1/32])
my_h=np.array([1/3,1/6])

my_error=np.array([0.05,0.025])
gjerde_error=np.array((0.02,0.01,0.002,0.0005))

my_FV=np.array([1/6,1/12,1/24,1/48,1/60,1/90,1/120])
my_FV_error=np.array((3.2,1.76,1.26,0.6,0.6,0.6,0.06))

FEM1=np.array([1/4,1/16,1/32,1/64])
FEM_error1=np.array([0.2,0.2,0.1,0.05])

FEM_error2=np.array([0.025,0.03,0.025,0.003])



plt.plot(gjerde_h, gjerde_error, label="[Gjerde et al.,2019]", marker='>', markersize=15)
plt.plot(h, L2_norm_q, label="coupling model", marker='<', markersize=15)
plt.plot(my_FV, my_FV_error, label="Peaceman FV model", marker='o', markersize=15)
plt.plot(FEM1, FEM_error1, label="approx FEM error [Gjerde et al.,2019], R_small", marker='*', markersize=15)
plt.plot(FEM1, FEM_error2, label="approx FEM error [Gjerde et al.,2019], $R_{FEM}=6*R_v$", marker='*', markersize=15)
plt.yscale("log")
plt.xscale("log")

plt.xlabel('$h/L$')
plt.ylabel('$L^2 error$')

#plt.legend()

plt.title("$L^2$ error for the flux (q) estimation for a single source")


#For multiple sources

L2_norm_mult_q=np.array([5,4,1.6])*10**-2
L1_norm_q=np.array([0.09, 0.05 , 0.01])*10**-2

plt.figure(figsize=(9,9))
plt.plot(gjerde_h, gjerde_error, label="[Gjerde et al.,2019]", marker='>', markersize=15)
plt.plot(h, L2_norm_q, label="coupling model", marker='<', markersize=15)
#plt.plot(my_FV, my_FV_error, label="Peaceman FV model", marker='o', markersize=15)
plt.plot(FEM1, FEM_error1, label="approx FEM error [Gjerde et al.,2019], R_small", marker='*', markersize=15)
plt.plot(FEM1, FEM_error2, label="approx FEM error [Gjerde et al.,2019], $R_{FEM}=6*R_v$", marker='*', markersize=15)
plt.plot(h,L2_norm_mult_q, label="multiple L2")
plt.plot(h, L1_norm_q, label="multiple_L1")
plt.yscale("log")
plt.xscale("log")

plt.xlabel('$h/L$')
plt.ylabel('$L^2 error$')

plt.legend()

plt.title("$L^2$ error for the flux (q) estimation for a single source")