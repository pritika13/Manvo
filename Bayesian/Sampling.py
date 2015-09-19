'''
Created on Apr 6, 2015

@author: pritika
'''
from Bayesian import *
import random


def generate(obj,val_list):
    index = obj.calculate_index_independent(val_list)
    table = obj.getTable()
    l = table[index].tolist()
    val = random.random()
    for i in range(0,len(l)):
        val=val-l[i]
        if val<=0:
            break
    return i

def iterate_Ancestral(obj,val_list):
    global finalObjList
    parent_list = obj.getParents()
    for p in parent_list:
        if val_list[p] is -1:
            iterate_Ancestral(finalObjList[p],val_list)
    if val_list[obj.getMyNodeNumber()] is -1:
        val_list[obj.getMyNodeNumber()] = generate(obj,val_list)

def getAncestral(M,n):
    global finalObjList
    samples =np.zeros((M,n))
   
    for i in range(0,M):
        val_list = [-1]*n
        for obj in finalObjList:
            iterate_Ancestral(obj,val_list)
        samples[i,:]=np.array(val_list)
    return samples

def getGibbs(M,n):
    # parameters
    size_to_Mix = 10000
    size_to_skip=1000
    #
    l1 = [0]*n
    l2 = [0]*n
    samples =np.zeros((M,n))
    count=0
    for i in range (0,n):#initialisation
        l1[i] = random.randint(0,getMax(type)[i])    ##########################################
    for t in range(0,size_to_Mix+M*size_to_skip):
        #queue to keep the top 100 values
        for i in range(0,len(finalObjList)):
            l1[i]=generate(finalObjList[i],l1)
        if t>=size_to_Mix and t%size_to_skip==size_to_skip-1:
            samples[count,:]=np.array(l1)
            count=count+1
    return samples

#################for ances

def iterate_Ancestral_test(obj,val_list):
    global finalObjList
    parent_list = obj.getParents()
    for p in parent_list:
        if p>11 and val_list[p] is -1:
            iterate_Ancestral(finalObjList[p],val_list)
    if val_list[obj.getMyNodeNumber()] is -1 and obj.getMyNodeNumber()>11 :
        val_list[obj.getMyNodeNumber()] = generate(obj,val_list)

def getAncestral_test(samples,M,n):
    global finalObjList
   # samples =np.zeros((M,n))
    modifiedCPD = finalObjList[12:36]
    samples_new = np.zeros((M,n))
    for i in range(0,38):
        print("in for")
        val_list = [-1]*n
        for j in range(0,11):
            val_list[j] = samples[i][j]
        for obj in modifiedCPD:
            iterate_Ancestral_test(obj,val_list)
        print("val list is ")
        print(val_list)    
        samples_new[i,:]=np.array(val_list)
        
    return samples_new