'''
Created on Apr 6, 2015

@author: pritika
'''
from Bayesian import *
from GenerateCPD import calCPD
import math

################# statistics #######################
def dist(samples,m,type):
    N=samples.shape[0]
    D=samples.shape[1]
    d=np.zeros(N)
    max_val=getMax(type)
    for i in range(0,N):
        for j in range(0,D):
            d[i]=d[i]+abs(samples[i,j]-m[j])/max_val[j]
        d[i]=d[i]/D
    return d

def meanComRare(samples,type):
    N=samples.shape[0]
    D=samples.shape[1]
    
    new_samples = samples
    
    #####for 12
    samples = new_samples[:,0:12]
    
    m=np.zeros((3,D))
    a1=samples.sum(axis=0)/samples.shape[0]
    d=dist(samples,a1,type)
    a2=samples[d.argmin(),:]
    a3=samples[d.argmax(),:]
    
    ####for 12:24
    samples = new_samples[:,12:24]
   
  
    b1=samples.sum(axis=0)/samples.shape[0]
    d=dist(samples,b1,type)
   
    b2=samples[d.argmin(),:]
    b3=samples[d.argmax(),:]
    
    
    ####for 25:36
    samples = new_samples[:,24:36]
    
    c1=samples.sum(axis=0)/samples.shape[0]
    d=dist(samples,c1,type)
    c2=samples[d.argmin(),:]
    c3=samples[d.argmax(),:]
    
   
    for i in range(0,36):
        if i<12:
            m[0,i] = a1[i] 
            m[1,i] = a2[i]
            m[2,i] = a3[i]
        if i>=12 and i<24:
            m[0,i] = b1[i-12] 
            m[1,i] = b2[i-12]
            m[2,i] = b3[i-12]
            
        if i >=24:   
            m[0,i] = c1[i-24] 
            m[1,i] = c2[i-24]
            m[2,i] = c3[i-24]
   
    return m



def meanComRareOld(samples,type):
    N=samples.shape[0]
    D=samples.shape[1]
    m=np.zeros((3,D))
    m[0,:]=samples.sum(axis=0)/samples.shape[0]
    d=dist(samples,m[0,:],type)
    m[1,:]=samples[d.argmin(),:]
    m[2,:]=samples[d.argmax(),:]
    return m

def entopyIndep(samples,type):
    # first bin data:
    N=samples.shape[0]
    D=samples.shape[1]
    P_table=np.zeros((D,max(getMax(type))+1))
    for i in range(0,D):
        for j in range(0,max(getMax(type))+1):
            P_table[i,j] =samples[samples[:,i]==j].shape[0]
        sum_arr = np.sum(P_table, axis=1)
       
        if sum_arr[i] == 0:
            P_table[i,:]= P_table[i,:]/36
        else:    
            P_table[i,:]= P_table[i,:]/sum_arr[i]
    entropy=np.zeros(D)
    entropy=np.zeros(D)
    for i in range(0,N):
        for j in range(0,D):
            
            m = P_table[j,samples[i,j]]
            if m == 0:
               
                m = 1
            entropy[j]=entropy[j]-math.log(m)
    entropy=entropy/N
    return entropy
    
def entopy(samples,CPD_list):
    N=samples.shape[0]
    return -logP(samples,CPD_list)/N
    
def KL(samples,CPD_list_1,CPD_list_2):
    N=samples.shape[0]
    return (-logP(samples,CPD_list_2)+logP(samples,CPD_list_1))/N
    
def mutualEntropy(samples,CPD_list,num):
    D=samples.shape[1]
    CPD_list_empty=calCPD(samples,np.zeros((D,D)),num)
    return KL(samples,CPD_list,CPD_list_empty)

