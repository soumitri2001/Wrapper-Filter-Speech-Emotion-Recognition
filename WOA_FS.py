'''
The codes have been taken from the following repository:
https://github.com/CMATER-JUCS/Py_FS
'''

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from utils.feature_selection import *


def whale_optim_algo(num_agents, max_iter, data):
    
    short_name = 'WOA'
    agent_name = 'Whale'
    num_features = data.train_X.shape[1]
    cross_limit = 10
    trans_function = get_trans_function()
    np.random.seed(1)

    # setting up the objectives
    weight_acc = 0.99
    obj_function = compute_fitness
    obj = (obj_function, weight_acc)
    compute_accuracy = (compute_fitness, 1)

    # initialize whales and Leader (the agent with the max fitness)
    whales = initialize(num_agents, num_features)

    fitness = np.zeros(num_agents)
    accuracy = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")
    Leader_accuracy = float("-inf")

    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # rank initial population
    whales, fitness, accs = sort_agents(whales, obj, data)

    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        a = 2 - iter_no * (2/max_iter)  

        # update the position of each whale
        for i in range(num_agents):
            # update the parameters
            r = np.random.random() 
            A = (2 * a * r) - a  
            C = 2 * r  
            l = -1 + (np.random.random() * 2)  
            p = np.random.random() 
            b = 1                 
            
            if p < 0.5:
                # Shrinking Encircling mechanism
                if abs(A) >= 1:
                    rand_agent_index = np.random.randint(0, num_agents)
                    rand_agent = whales[rand_agent_index, :]
                    mod_dist_rand_agent = abs(C * rand_agent - whales[i,:]) 
                    whales[i,:] = rand_agent - (A * mod_dist_rand_agent)   # Eq. (9)
                else:
                    mod_dist_Leader = abs(C * Leader_agent - whales[i,:]) 
                    whales[i,:] = Leader_agent - (A * mod_dist_Leader)  # Eq. (2)  
            else:
                # Spiral-Shaped Attack mechanism
                dist_Leader = abs(Leader_agent - whales[i,:])
                whales[i,:] = dist_Leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + Leader_agent

            # Apply transfer function on the updated whale to map into binary
            for j in range(num_features):
                trans_value = trans_function(whales[i,j])
                if (np.random.random() < trans_value): 
                    whales[i,j] = 1
                else:
                    whales[i,j] = 0

        # update final information
        whales, fitness, accs = sort_agents(whales, obj, data)
        display(whales, fitness, accs, agent_name)
        if fitness[0]>Leader_fitness:
            Leader_agent = whales[0].copy()
            Leader_fitness = fitness[0].copy()
            Leader_accs = accs[0].copy()

    # compute final accuracy
    Leader_agent,_, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    whales,_, accuracy = sort_agents(whales, compute_accuracy, data)

    print('\n================================================================================')
    print('                                    Final Result                                  ')
    print('================================================================================\n')
    print('Leader ' + agent_name + ' Dimension : {}'.format(int(np.sum(Leader_agent))))
    print('Leader ' + agent_name + ' Fitness : {}'.format(Leader_fitness))
    print('Leader ' + agent_name + ' Classification Accuracy : {}'.format(Leader_accuracy))
    print('\n================================================================================\n')

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.final_population = whales
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    
    return solution


### Testing the FS function with sample dataset ###
if __name__ == "__main__":
    data = datasets.load_digits()
    X = np.array(data.data)
    y = np.array(data.target)
    obj = Data()
    obj.train_X, obj.val_X, obj.train_Y, obj.val_Y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    sol_WOA = whale_optim_algo(num_agents=40, max_iter=100, data=obj)
    validate_FS(X, y, sol_WOA.best_agent, save=False)