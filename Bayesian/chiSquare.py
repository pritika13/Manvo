from __future__ import division

from os import listdir
import os, sys
from os.path import isfile, join
import numpy as np
import operator
from Bayesian import *
import Prior

path = "sample";
dirs = os.listdir(path);


def getJointProb(i,j,k,l,subset_f,length):
    
    first = subset_f[:,i]
    second = subset_f[:,j]    
    times=0
    for x in range(0,length):
        if first[x]==k and second[x]==l:
            times = times+1
            
    
    return times#/length    

def getChi(content,num,independent_prob_table):
    
    n= len(content)
    print("n is "+str(n))
    
#file_n = "sample/"+f
#f_n = f
    table = independent_prob_table
    k =0
    
    print("table ")
    print(table)
    chi_test ={"-1":-1};
   
    #getdata from file
    """with open(file_n, 'r') as f:
        content = f.readlines()
    f.close()"""


    subset_f=np.zeros((len(content),num))
    i =0
    for i in range(0,n):
       # l=c.split("\t")[:num]
        for j in range(0,num):
           subset_f[i][j] = int(content[i][j])
        i = i+1

    for i in range(num-12,num-1):
        for j in range(i+1,num):
            print("i is "+str(i)+" and j is "+str(j));
            l_1 = table[i].tolist();
            l_2 = table[j].tolist()
            #l_1 = l_1[:l_1.index(0)]
            #l_2 = l_2[:l_2.index(0)]
            cal_sum =0
            for k in range(0,len(l_1)):
                for l in range(0,len(l_2)):
                    Expectation = l_1[k]*l_2[l]*n
                    original = getJointProb(i,j,k,l,subset_f,n)  
                    print("original is "+str(original)+" expected = "+str(Expectation))  
                    if Expectation > 0:
                        cal_sum = cal_sum+((original-Expectation)*(original-Expectation))/Expectation
            key = str(i)+"\t"+ str(j)        
            chi_test.update({key:cal_sum})        
            chi_test.pop("-1", None)            
            sorted_x = sorted(chi_test.items(), key=operator.itemgetter(1),reverse=True)
            name =""
            if num is 24:
                name = "chi/file24.txt"
            elif num is 36:
                 name = "chi/file36.txt"
            elif num is 12:
                name = "chi/file12.txt"     
            if num is 24 or 36 or 12:         
                with open(name, 'w') as f1:
                    for k,v in sorted_x:
                        f1.write(str(k)+"\t"+str(v)+"\n")
                    f1.write('\n')
                f1.close()        
    
    return sorted_x

