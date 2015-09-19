from Bayesian import *
import math 
import random
import Bayesian
from Crypto.Util.number import size


def loadData(n):
    global training
    global cross_validation
    global testing
    if n is 121:
        if type is 'c':
            file_sam="sample/cursive_20"+str(year)+"-20"+str(year+1)+"_"+str(grade)+".txt"
        else:
            file_sam="sample/handwritten_20"+str(year)+"-20"+str(year+1)+"_"+str(grade)+".txt"
    else:
        file_sam = "sample/data_3years.txt"         
    with open(file_sam, 'r') as f:
        content = f.readlines()
    f.close()
    random.shuffle(content)
    num_sample=len(content)
  
    samples=np.zeros((num_sample,n))

    for i in range(0,num_sample):
        l=content[i].split("\t")[:n]
        for j in range(0,n):
            samples[i][j] = int(l[j])
     
    for i in range(0,int(math.ceil(training_percent*num_sample))):        
        training.append(samples[i,:].tolist())
    
    for i in range(int((math.floor(training_percent*num_sample)+1)),int(math.floor((training_percent+cross_validation_percent)*num_sample))):    
        cross_validation.append(samples[i,:].tolist())
    
    for i in range(int(math.floor((training_percent+cross_validation_percent)*num_sample)+1),len(samples)):    
        testing.append(samples[i,:].tolist())
        
    training.pop(0)
    cross_validation.pop(0)
    testing.pop(0)
   
  
def getAllSamples(num):
    file_sam = "sample/data_3years.txt"         
    with open(file_sam, 'r') as f:
        content = f.readlines()
    f.close()
    
    num_sample=len(content)
    samples=np.zeros((num_sample,num))

    for i in range(0,num_sample):
        l=content[i].split("\t")[:num]
        for j in range(0,num):
            samples[i][j] = int(l[j])    
       
    return samples    
       
    

        
