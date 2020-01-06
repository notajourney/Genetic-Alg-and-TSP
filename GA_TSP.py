# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:07:17 2020

@author: Win10
"""


import numpy as np
import matplotlib.pyplot as plt
import os


def GA_TSP(n, max_evals, selectfct, fitnessfct, crossoverfct, mutationfct, max_attainable=np.inf, seed=None):
    eval_cntr = 0
    history = []
    #
    # GA params
    mu = 100
    pc = 0.37
    pm = 2/n
    #    kXO = 1 # 1-point Xover
    local_state = np.random.RandomState(seed)
    # TODO 1 create mu permutations of size n
    Genome = np.array([local_state.permutation(n) for i in range(mu)]).T
    fitness = fitnessfct(Genome)
    eval_cntr += mu

    fcurr_best = fmin = np.min(fitness)

    xmin = Genome[:, [np.argmin(fitness)]]

    history.append(fmin)
    while (eval_cntr < max_evals):
        # Generate offspring population (recombination, mutation)
        newGenome = np.empty([n, mu], dtype=int)
        # 1. sexual selection + 1-point recombination
        for i in range(int(mu/2)):
            p1 = selectfct(Genome, fitness, local_state)
            p2 = selectfct(Genome, fitness, local_state)
            if local_state.uniform() < pc:  # recombination
                # TODO 2 implement crossover for TSP
                Xnew1, Xnew2 = crossoverfct(p1, p2, local_state)
            else:  # no recombination; two parents are copied as are
                Xnew1 = np.copy(p1)
                Xnew2 = np.copy(p2)
            #        2. mutation
            # TODO 2 implement mutation for TSP
            Xnew1 = mutationfct(Xnew1, local_state)
            Xnew2 = mutationfct(Xnew2, local_state)
            #
            newGenome[:, [2*i-1]] = np.copy(Xnew1)
            newGenome[:, [2*i]] = np.copy(Xnew2)
        # The best individual of the parental population is kept
        Genome = np.copy(newGenome)

        fitness = fitnessfct(Genome)
        eval_cntr += mu
        fcurr_best = np.min(fitness)
        if fmin > fcurr_best:
            fmin = fcurr_best
            xmin = Genome[:, [np.argmin(fitness)]]
        history.append(fcurr_best)
        if np.mod(eval_cntr, int(max_evals/10)) == 0:
            print(eval_cntr, " evals: fmax=", fmin)
        if fmin == max_attainable:
            print(eval_cntr, " evals: fmax=", fmin, "; done!")
            break
    return xmin, fmin, history
#

# selectfct

#TODO : check if RWS works and if not, give simpler implementation
def select_proportional(Genome, fitness, rand_state):
    ''' RWS: Select one individual out of a population Genome with fitness values fitness using proportional selection.'''
    cumsum_f = np.cumsum(fitness)
    r = sum(fitness) * rand_state.uniform()
    idx = np.ravel(np.where(r < cumsum_f))[0]
    return Genome[:, [idx]]

# fitnessfct

# higher order function
# a function that returns a function
# while saving the graph in a closure


def compute_tour_length(graph):
    def compute_tour_for_perm(perm):
        tlen = 0.0
        for i in range(len(perm)):
            tlen += graph[perm[i], perm[np.mod(i+1, len(perm))]]
        return tlen
    return compute_tour_for_perm

# TODO crossoverfct
def order_one_crossover(papa, mama, rand):

    return

# TODO mutationfct
def swap_mutation(perm, rand):
    for i in range(len(perm))
    
    return

# create graph from TSP file


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


if __name__ == "__main__":
    graph = create_graph("tokyo.dat")
    n = len(graph)  # TODO test if correct
    print(n)
    evals = 10**6
    Nruns = 1
    fbest = []
    xbest = []
    for i in range(Nruns):
        #        xmin,fmin,history = GA(n,evals,decoding_ones,select_proportional,TeleCom,n,i+37)
        xmin, fmin, history = GA_TSP(n, evals, select_proportional,
                                     compute_tour_length(graph), order_one_crossover, swap_mutation, n, i+37)
        plt.semilogy(np.array(history))
        plt.show()
        # represent solution in a print
        print(i, ": minimal tour found is ", fmin, " at location ", xmin)
        fbest.append(fmin)
        xbest.append(xmin)
    print("====\n Best ever: ", max(fbest),
          "x*=", xbest[fbest.index(max(fbest))].T)
