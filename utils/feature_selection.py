'''
The code structure used has been taken from the following repository:
https://github.com/CMATER-JUCS/Py_FS
'''

import os 
import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_score, classification_report, plot_confusion_matrix, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN

random.seed(1)
np.random.seed(1)


### utility classes/functions for filter-based FS ###
class Result():
    # structure of filter-based FS result
    def __init__(self):
        self.ranks = None
        self.scores = None
        self.features = None
        self.ranked_features = None 

### to normalize a vector in [lb, ub] ###
def normalize(vector, lb=0, ub=1):
    # function to normalize a numpy vector in [lb, ub]
    norm_vector = np.zeros(vector.shape[0])
    maximum = max(vector)
    minimum = min(vector)
    norm_vector = lb + ((vector - minimum)/(maximum - minimum)) * (ub - lb)

    return norm_vector


### utility classes/functions for wrapper-based FS ###
class Solution():    
    #structure of the solution 
    def __init__(self):
        self.num_features = None
        self.num_agents = None
        self.max_iter = None
        self.obj_function = None
        self.execution_time = None
        self.convergence_curve = {}
        self.best_agent = None
        self.best_fitness = None
        self.best_accuracy = None
        self.final_population = None
        self.final_fitness = None
        self.final_accuracy = None

class Data():
    # structure of the training data
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.val_X = None
        self.val_Y = None

### initializing population for wrapper-based FS ###
def initialize(num_agents, num_features):
    # define min and max number of features
    min_features = int(0.5 * num_features)
    max_features = int(0.8 * num_features)
    agents = np.zeros((num_agents, num_features))
    for agent_no in range(num_agents):
        random.seed(time.time() + agent_no)
        num = random.randint(min_features,max_features)
        pos = random.sample(range(0,num_features - 1),num)
        for idx in pos:
            agents[agent_no][idx] = 1 
    return agents

### utility function to sort agents based on fitness ###
def sort_agents(agents, obj, data):
    # sort the agents according to fitness
    train_X, val_X, train_Y, val_Y = data.train_X, data.val_X, data.train_Y, data.val_Y
    (obj_function, weight_acc) = obj
    num_agents = agents.shape[0]
    fitness = np.zeros(num_agents)
    acc = np.zeros(num_agents)
    for id, agent in enumerate(agents):
        fitness[id], acc[id] = obj_function(agent, train_X, val_X, train_Y, val_Y, weight_acc)
    idx = np.argsort(-fitness)
    sorted_agents = agents[idx].copy()
    sorted_fitness = fitness[idx].copy()
    sorted_acc = acc[idx].copy()
    return sorted_agents, sorted_fitness, sorted_acc

### utility function to display population at a time step ###
def display(agents, fitness, acc, agent_name='Agent'):
    # display the population
    print('\nNumber of agents: {}'.format(agents.shape[0]))
    print('\n------------- Best Agent ---------------')
    print('Fitness: {}'.format(fitness[0]))
    print('Accuracy: {}'.format(acc[0]))
    print('Number of Features: {}'.format(int(np.sum(agents[0]))))
    print('----------------------------------------\n')
    for id, agent in enumerate(agents):
        print('{} {} - Fitness: {},Accuracy: {}, Number of Features: {}'.format(agent_name, id+1, fitness[id], acc[id], int(np.sum(agent))))
    print('================================================================================\n')

### calculate accuracy of each agent ### 
def compute_accuracy(agent, train_X, test_X, train_Y, test_Y): 
    # compute classification accuracy of the given agents
    cols = np.flatnonzero(agent)     
    if(cols.shape[0] == 0):
        return 0    
    clf = KNN()
    train_data = train_X[:,cols]
    train_label = train_Y
    test_data = test_X[:,cols]
    test_label = test_Y
    clf.fit(train_data,train_label)
    acc = clf.score(test_data,test_label)
    return acc
        
### calculate fitness value of each agent ### 
def compute_fitness(agent, train_X, test_X, train_Y, test_Y, weight_acc=0.99,dims=None):
    # compute a basic fitness measure
    if(weight_acc == None):
        weight_acc = 0.99
    weight_feat = 1 - weight_acc
    agent = agent.reshape(-1)
    if dims != None:
        num_features = dims
    else:
        num_features = agent.shape[0]
    acc = compute_accuracy(agent, train_X, test_X, train_Y, test_Y)
    feat = (num_features - np.sum(agent))/num_features
    fitness = weight_acc * acc + weight_feat * feat
    return fitness, acc

### Transfer function to map continuous to binary ###
def sigmoid(val):
    if val < 0:
        return 1 - 1/(1 + np.exp(val))
    else:
        return 1/(1 + np.exp(-val))

def get_trans_function(shape='s'):
    return sigmoid

### validate an agent in FS ###
def validate_FS(X,y,agent,caption="conf_mat", save=False):
    cols = np.flatnonzero(agent)       
    X1 = (X[:,cols]).copy() # getting selected features
    X_train,X_test,y_train,y_test =  train_test_split(X1, y, test_size=0.2, shuffle=False, random_state=1)
    model = KNN()
    model.fit(X_train,y_train)    
    y_pred = model.predict(X_test)
    accuracy = np.sum(y_pred == y_test)/y_test.shape[0]
    print(f'Accuracy for the selection using KNN classifier: {accuracy:.6f}')
    print('-'*50)
    print(classification_report(y_test,y_pred,digits=4))
    print('-'*50)
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(model, X_test, y_test, values_format = 'd')
    if save==True:
        plt.savefig(f'{caption}.jpg',dpi=300)
    plt.show()

