# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 09:35:13 2020

@author: Win10
"""
import numpy as np

#######################################################################################################

def get_indecies (low=0, high=100,seed=None):#returns two different indecies [smaller, greater]
    
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

#######################################################################################################

def crossover_perm (parent1,parent2,idx1,idx2,seed=None):#returns child after crossover with both parents
    #1:implements "order one crossover"

    #2: copies to child from parent2  the remaining permutation candidates ( that are not already included in childe)
        # in the same order the candidates apear  in parent2

    index1 = idx1
    index2=idx2
    #print("indecies are:",index1, index2)
    local_state = np.random.RandomState(seed)
    child = local_state.permutation(len(parent1))

    #print("child when created: ",child)
    smaller = index1
    larger = index2
    copied_segment=[]
    
    while (smaller <= index2):#cpy segment to child from parent1
        child[smaller]=parent1[smaller]
        copied_segment.append(parent1[smaller])
        smaller=smaller+1
    
    smaller = index1
    larger = index2+1#start copying parent2 right after the cut point
    child_index =larger;
    
    for i in range(len(parent2)):
        
        if(not parent2[larger%10] in copied_segment):
            child[child_index%10] = parent2[larger%10]
            child_index=child_index+1
            
        larger=larger+1

    return np.copy(child)
#######################################################################################################
def mutate_perm(child,lower_bound=0,upper_bound=150,seed=None):#implements a swap mutation
    index1,index2= get_indecies(lower_bound, upper_bound,seed)
    temp= child[index1]
    child[index1] = child[index2]
    child[index2]=temp
    return np.copy(child)

#######################################################################################################



















###################################################

##older version of crossover_perm (delete before submiting)
def crossover_permmmmmmm (parent1,parent2,child,local_state,lower_bound=0,upper_bound=100):#returns child after crossover with both parents
    #1:implements "order one crossover"
    #2: randomizes two different indecies, and copies the segment indicated by them from parent1 to child
    #3: copies to child from parent2  the remaining permutation candidates ( that are not already included in childe)
        # in the same order the candidates apear  in parent2
    index1,index2 = get_indecies(local_state,0,parent1.size)
    print("indecies are:",index1, index2)
    
    smaller = index1
    larger = index2
    copied_segment=[]
    
    while (smaller <= index2):#cpy segment to child from parent1
        child[smaller]=parent1[smaller]
        copied_segment.append(parent1[smaller])
        smaller=smaller+1
    
    smaller = index1
    larger = index2+1#start after the cut point
    stopat = larger-1;#end at the cutpoint
    child_index =larger;
    
    
    
    while(1):#cpy by order of parent2
    
        if(not parent2[larger%10] in copied_segment):
            child[child_index%10] = parent2[larger%10]
            child_index=child_index+1
        
        larger=larger+1;
        if(larger%10==stopat):
            break;
    
    print("parent1:",parent1)
    print("parent2:",parent2)
    print("child:  ",child)
    return child