# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:37:43 2020

@author: Win10
"""
import GA_TSP_Operators as op
import numpy as np


Genome = np.array([np.random.permutation(10) for i in range(2)]).T

#print(Genome[:,0]) # kol hashurot ba'amuda ha efes
#                         =======>
#                                               ||
#                                               ||
#                                               ||
#                                               V



# =============================================================================
# =============================================================================
# index1,index2 =  op.get_indecies(0,10);
# child = op.crossover_perm(Genome[:,0], Genome[:,1],index1,index2)
# print("child sorted : ",np.sort(child))
# 
# print("parent1shape: ",Genome[:,0].shape)
# print("parent2 shape: ",Genome[:,1].shape)
# print("child shape  : ",child.shape)
# op.mutate_perm(child,0,100)
# =============================================================================
# =====================

x = np.random.permutation(10);
print(x)
print("minima value is at index: ", np.argmin(x))
print("maxima value is at index: :",np.argmax(x))


x[-2] =17
print(x)



