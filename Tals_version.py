# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:48:33 2020
Tal's version of this project:
    In this version, the entire source code was changed
    in order to avoid  the syntax of columns. Means that in this
    version all population are rows and from a specific array type
    (unlike the older version where each individual was 1X15 matrix)
    in this version every individual is an array
@author: Win10
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:07:17 2020
@author: Win10
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import GA_TSP_Operators as op
import inspect


def GA_TSP_V2(n, max_evals, selectfct, fitnessfct, crossoverfct, mutationfct,  seed=None):
    eval_cntr = 0
    history = []
    mu = 20
    pc = 0.37
    pm = 0.2
    local_state = np.random.RandomState(seed)
    
    #==create & eval initial population==#
    population = np.array([local_state.permutation(n) for i in range(mu)])#the type of "population" is important for th difference between two version 
    fitness = fitnessfct(population)
    eval_cntr += mu
    fcurr_best = fmin = np.min(fitness)
    xmin = population[np.argmin(fitness)]    
    history.append(fmin)
    #==create & eval initial population==#
    
    while eval_cntr < max_evals:
        newPopulation = np.empty([mu,n], dtype=int)#why [mu,n] and not [n,mu]
        for i in range(int(mu/2)):
            #==select two strong parents& create next generation by cross and mute============#
            p1 = selectfct(population, fitness, local_state)
            p2 = selectfct(population, fitness, local_state)
            if local_state.uniform() < pc:  # Crossover over permutation
                Child1 = crossoverfct(p1,p2,local_state)
                Child2 = crossoverfct(p2,p1,local_state)
            else:  # no Crossover
                Child1 = np.copy(p1)
                Child2 = np.copy(p2)
            if local_state.uniform() < pm:
                mutationfct(Child1,local_state)
                mutationfct(Child2,local_state)
            newPopulation[2*i-1,:] = np.copy(Child1)#mismatch shape
            newPopulation[2*i,:] = np.copy(Child2) #also mismatch shape  
            #==select two strong parents& create next generation by cross and mute=============#
         
        #== eval new population==========#    
        population = newPopulation
        fitness = fitnessfct(population)
        eval_cntr += mu
        fcurr_best = np.min(fitness)
        #== eval new population==========#
        
        #==update minimum attained and best individual=====#
        if fmin > fcurr_best:
            fmin = fcurr_best
            xmin = population[np.argmin(fitness)]
        #==update minimum attained and best individual=====#
        
        """Adding the best parent back to next generation made a DRAMATIC change in the bad results we had before """
        population[0] = xmin;
        population[1] = xmin;
        history.append(fmin)
        if np.mod(eval_cntr, int(max_evals/100)) == 0:
            print(eval_cntr, " evals: fmin=", fmin)
    return xmin, fmin, history
    
##################################################SEXUAL SELECTION ROLLETE###############################################################

def select_proportional_v2(population, fitness, rand):
    ''' RWS: Select one individual out of a population Genome with fitness values fitness using proportional selection.'''
    cumsum_f = np.cumsum(fitness)
    r = sum(fitness) * rand.uniform()
    idx = np.ravel(np.where(r < cumsum_f))[0]
    return population[idx]

####################################################TOUR LENGTH###########################################################
def compute_tour_length_v2(graph):
    def compute_tour_for_genome_v2(population):
        (mu,n) = np.shape(population)
        fitness = np.empty(mu,dtype=float);
        for j in range(mu):
            tlen = 0.0
            for i in range(n):
                try:
                    tlen += graph[population[j,i-1], population[j,i]]
                except:
                    print("Wtf")
            fitness[j] = tlen
        return fitness
    return compute_tour_for_genome_v2
###################################################CREATE GRAPH###############################################################

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

#########################################################################################################################
    

#####################################################CROSSOVER###########################################################
def crossover_v2(p1,p2,rand):
    n= len(p1);
    child = np.empty(n,dtype=int);
    [index1,index2] = rand.choice(n,2,replace=True)
    if index1 > index2:
        index1,index2 = index2,index1
    smaller = index1
    larger = index2
    copied_segment=[]
    
    while smaller <= index2:#cpy segment to child from parent1
        child[smaller]=p1[smaller]
        copied_segment.append(p1[smaller])
        smaller=smaller+1    
    smaller = index1
    larger = index2+1#start copying parent2 right after the cut point
    child_index =larger;
    
    for i in range(len(p2)):        
        if  p2[larger%n] not in copied_segment:
            child[child_index%n] = p2[larger%n]
            child_index=child_index+1            
        larger=larger+1
    return child
#############################################################MUTATION#########################################################

def mutate_perm_v2(child,local_state):
    index = local_state.randint(0,len(child));
    temp= child[index]
    child[index] = child[index-1]
    child[index-1]=temp
    
#############################################################################################################################
    
#################################################CONTROL FUNCTIONS #################################################################
    
def test_for_bad_perm(perm,cnt):# CONTROL FUNCTION: Check if all permutations are valid
    if len(perm) != len(np.unique(perm)):
        print("bad perm",cnt)
      
        
def test_for_bad_genome(genome):# CONTROL FUNCTION: Check if Genome is valid
    frame = inspect.currentframe()
    outer = inspect.getouterframes(frame,2)
    for col in genome.T:
        if len(col) != len(np.unique(col)):
            print("super bad",iteration,outer[1][3]);
            return
        
##############################################################MAIN#############################################################
if __name__ == "__main__":
    iteration = 0    
    graph = create_graph("tokyo.dat")
    n = len(graph)  
    evals = 10**5
    Nruns = 1
    fbest = []
    xbest = []
    for i in range(Nruns):
        xmin,fmin,history = GA_TSP_V2(n, evals, select_proportional_v2,
                                    compute_tour_length_v2(graph), crossover_v2, mutate_perm_v2)

        
        plt.semilogy(np.array(history))
        plt.show()
        # represent solution in a print
        print(i, ": minimal tour found is ", fmin, " at location ", xmin)
        fbest.append(fmin)
        xbest.append(xmin)
    print("====\n Best ever: ", max(fbest),
          "x*=", xbest[fbest.index(max(fbest))].T)
