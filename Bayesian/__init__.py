import numpy as np
import math

#define global variables
training_percent         = 0.7
cross_validation_percent = 0.15
testing_percent          = 0.15

samples                 = []
training                = [[]]
cross_validation        = []
testing                 = []
independent_prob_table  = []
CPD_list                = []
log_loss_training       = []
log_loss_testing        = []
BIC                     = []
finalObjList            = []
CPD_last                = []

year  = 11
grade = 3
type  = 'h'
alphaRange  = [50,25,10,5,1]
bestAlpha   = 50
num_samples = 100

t =[1,2,3]

alpha = bestAlpha



# defining functions which are general and used in more than one file

#maximum values for each node, c for cursive
def getMax(type):
    if type is 'c':
        max_vals= [4,5,3,5,4,4,4,5,3,4,4,4,4,5,3,5,4,4,4,5,3,4,4,4,4,5,3,5,4,4,4,5,3,4,4,4]
    else :
        max_vals= [6,6,4,6,3,5,6,4,3,4,4,4,6,6,4,6,3,5,6,4,3,4,4,4,6,6,4,6,3,5,6,4,3,4,4,4]#[6,6,4,6,4,5,6,4,3,4,4,4]
    return max_vals

# return empty graph
def getEmptyGraph(V):
    return np.zeros((V,V))

#return total number of rows of CPD and modifies the parents list
def getParents(i,parents_list,graph):  
    number_of_rows =1
    #print(graph)
    for j in range(0,12):
        if(graph[j][i] ==1):
            number_of_rows = number_of_rows*(getMax(type)[j]+1)
            parents_list.append(j)
    return number_of_rows  

def logP(samples,CPD_list):
    log_P=0 
    for i in range(0,samples.shape[0]):
        for obj in CPD_list:
            val = obj.getLogLoss(samples[i,:].tolist())
            log_P = log_P+math.log(val)
            #log_P = log_P+val
    return log_P


