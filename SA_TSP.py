
# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
Simulated Annealing on a continuous domain bounded within [lb,ub]**n
"""
import numpy as np



def SimulatedAnnealing(n, max_evals, fitnessfct,individual,graph, seed=None) :
    T_init=6.0 #initial temp
    T_min=1e-4 #minimal temp
    alpha=0.99 # the factor that shrinks the temp
    max_internal_runs = 50 # for internal loop
    local_state = np.random.RandomState(seed)#??

    
    
    xbest = xmin = individual
    fmin = fitnessfct(xmin,graph)
    eval_cntr = 1
    T = T_init
    
    
    
    while ((T > T_min) and eval_cntr < max_evals) :
        for _ in range(max_internal_runs) :
            
            x = step(xmin,int(5*T),local_state)#TODO create perm step
            f_x = fitnessfct(x,graph)# evaluating new vector
            eval_cntr += 1
            
            """WEAK POINT!!!! try flip fx fmax"""
            dE =   f_x - fmin #fmin - f_x  #de is negative iff  f_x is "better" fmin ::::::
            if dE <= 0 or local_state.uniform(size=1) < np.exp(-dE/T) :# accepting disimprovment
                xmin = x #updating minimum vector
                fmin = f_x#updating minimum value
            """WEAK POINT!!!! see if it maches first week point"""    
            if dE < 0 :# saving the global best:: regardless of accepting disaprovment
                #fbest=f_x   #this is necessarily the best value we have found so far
                xbest=x     #this is necessarily the best vector we have found so far
                

# =============================================================================
#             if fbest < f_lower_bound+eps_satisfactory :
#                 T=T_min
#                 break
# =============================================================================
        T *= alpha
        
    return xbest



#################################################Step Function#####################################################
def step(perm, temperture,local_state): #TODO implement step for permutation
    #n=150
    permcpy = perm.copy()
    
    for i in range(temperture):
        
     index = local_state.randint(0,len(permcpy));
     temp= permcpy[index]
     permcpy[index] = permcpy[index-1]
     permcpy[index-1]=temp
     
    return permcpy

#################################################Step Function#####################################################
  
    def compute_tour_for_perm(perm,graph):
        
        fitness = 0
        k=len(perm)
        
        tlen = 0.0
        for i in range(k):
            tlen += graph[perm[i], perm[(i+1)%k]]
        fitness = tlen
        return fitness
   
    
###################################################################################################################
