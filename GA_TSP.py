# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:07:17 2020

@author: Win10
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import GA_TSP_Operators as op


def GA_TSP(n, max_evals, selectfct, fitnessfct, crossoverfct, mutationfct,  seed=None):
    eval_cntr = 0
    history = []
    # GA params
    mu = 100
    pc = 0.37
    pm = 2/n
    
    local_state = np.random.RandomState(seed)
    Genome = np.array([local_state.permutation(n) for i in range(mu)]).T
    fitness = fitnessfct(Genome)
    eval_cntr += mu
    fcurr_best = fmin = np.min(fitness)
    xmin = Genome[:, [np.argmin(fitness)]]
    
    history.append(fmin)
    while (eval_cntr < max_evals):
        
        #Creating next generation
        newGenome = np.empty([n, mu], dtype=int)
        # 1. sexual selection + 1-point recombination
        for i in range(int(mu/2)):
            p1 = selectfct(Genome, fitness, local_state)
            p2 = selectfct(Genome, fitness, local_state)
            if local_state.uniform() < pc:  # Crossover over permutation
                idx1,idx2 =get_indecies(0,n,seed)
                Child1 = crossoverfct(p1,p2,idx1,idx2,seed)
                Child2 = crossoverfct(p2,p1,idx1,idx2,seed)
                
            else:  # no Crossover
                Child1 = np.copy(p1)
                Child2 = np.copy(p2)
    
            if local_state.uniform() < pm:
                mutationfct(Child1,0,n,seed)
                mutationfct(Child2,0,n,seed)
  
            newGenome[:, [2*i-1]] = np.copy(Child1)#mismatch shape
            newGenome[:, [2*i]] = np.copy(Child2.T) #also mismatch shape    
            
            #TODO:The best individual of the parental population is kept
        
        Genome = np.copy(newGenome)
        fitness = fitnessfct(Genome)
        eval_cntr += mu
        
        fcurr_best = np.min(fitness)
        if fmin > fcurr_best:
            fmin = fcurr_best
            xmin = Genome[:, [np.argmin(fitness)]]
        history.append(fcurr_best)
        if np.mod(eval_cntr, int(max_evals/10)) == 0:
            print(eval_cntr, " evals: fmin=", fmin)


    return xmin, fmin, history

#################################################################################################################
#TODO : check if RWS works and if not, give simpler implementation
def select_proportional(Genome, fitness, rand_state):
    ''' RWS: Select one individual out of a population Genome with fitness values fitness using proportional selection.'''
    cumsum_f = np.cumsum(fitness)
    r = sum(fitness) * rand_state.uniform()
    idx = np.ravel(np.where(r < cumsum_f))[0]
    return Genome[:, [idx]]


##################################################################################################################
def compute_tour_length(graph):
    def compute_tour_for_perm(perm):
        tlen = 0.0
        for i in range(len(perm)):
            tlen += graph[perm[i], perm[np.mod(i+1, len(perm))]]
        return tlen
    return compute_tour_for_perm


##################################################################################################################

def create_graph(file):
    dirname = ""
    fname = os.path.join(dirname, file)
    data = []
    with open(fname) as f:
        for line in f:
            data.append(line.split())
    n = len(data)
    G = np.empty([n, n])
    for i in range(n):
        for j in range(i, n):
            G[i, j] = np.linalg.norm(np.array([float(data[i][1]), float(
                data[i][2])]) - np.array([float(data[j][1]), float(data[j][2])]))
            G[j, i] = G[i, j]
    return G


#######################################################################################################

def get_indecies (low=0, high=150,seed=None):#returns two different indecies [smaller, greater]
    
    local_state = np.random.RandomState(seed)

    index1 = local_state.randint(low,high)
    index2 = local_state.randint(low,high)

    while (index1 == index2):
        index1 = local_state.randint(low,high)
        index2 = local_state.randint(low,high)
#print("oops:indecies are equal")

    if index1 > index2:
        greater = index1
        smaller = index2
    else:
        greater = index2
        smaller = index1

    return smaller, greater

#################################################Main#############################################################
if __name__ == "__main__":
    graph = create_graph("tokyo.dat")
    n = len(graph)  
    evals = 10**6
    Nruns = 1
    fbest = []
    xbest = []
    for i in range(Nruns):

        xmin, fmin, history = GA_TSP(n, evals, select_proportional,
                                     compute_tour_length(graph), op.crossover_perm, op.mutate_perm, i+37)
        plt.semilogy(np.array(history))
        plt.show()
        # represent solution in a print
        print(i, ": minimal tour found is ", fmin, " at location ", xmin)
        fbest.append(fmin)
        xbest.append(xmin)
    print("====\n Best ever: ", max(fbest),
          "x*=", xbest[fbest.index(max(fbest))].T)
