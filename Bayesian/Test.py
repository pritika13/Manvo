import numpy as np
import operator
import functools
import math 
import random
import time

def getMax(type):
    if type is 'c':
        max_vals= [4,5,3,5,4,4,4,5,3,4,4,4]
    else :
        max_vals= [5,5,3,5,4,4,5,5,3,4,4,4]#[6,6,4,6,4,5,6,4,3,4,4,4]
    return max_vals
    
def getEmptyGraph(V):
    return np.zeros((V,V))

def getParents(i,parents_list,graph): #return total number of rows of CPD and modifies the parents list 
    number_of_rows =1
    #print(graph)
    for j in range(0,12):
        if(graph[j][i] ==1):
            number_of_rows = number_of_rows*(getMax(type)[j]+1)
            parents_list.append(j)
    return number_of_rows    
    

def getfile_chi():
    if type is 'c':
        file_chi=("chi/cursive_20"+str(year)+"-20"+str(year+1)+"_"+str(grade)+".txt")
    else:
        file_chi="chi/handwritten_20"+str(year)+"-20"+str(year+1)+"_"+str(grade)+".txt"
    return file_chi
############################# Prior ################################

# get probability for each individual node
def getIndependentProbability():
    minimumP=0.001
    independent_prob_table=np.zeros((12,6+1))
    for i in range(0,12):
        for j in range(0,6+1):
            temp =training[training[:,i]==j].shape[0]
            if temp is 0 and j <=getMax(type)[i]:
                independent_prob_table[i,j] = minimumP*training.shape[0] # if count is zero
            else:
                independent_prob_table[i,j] = temp
        sum_arr = np.sum(independent_prob_table, axis=1)
        independent_prob_table[i,:]= independent_prob_table[i,:]/sum_arr[i]
    return independent_prob_table
    
def getAlpha(node,nodeValue,parentList,parentsValue,alpha):
    Pprime = independent_prob_table[node][nodeValue]
    
    for p in range(0,len(parentList)):    
        Pprime = Pprime*independent_prob_table[parentList[p]][parentsValue[p]]
    return alpha*Pprime    

########################### Structure ################################
def calBIC(logloss,CPD_list,M):
    k=0;
    for CPD in CPD_list:
        k=k+math.log(CPD.table.size)
    return -logloss+k*math.log(M)/2
    
def makeStructure(logloss,logloss_testing,BIC,criteria): 
    with open(getfile_chi(), 'r') as f:
        content = f.readlines()
    f.close()
    i =0
    graph = getEmptyGraph(12)
    #content = content[:5]####################################################
    M=cross_validation.shape[0]
    CPD_list=calCPD(training,graph)
    l=logP(cross_validation,CPD_list)
    logloss.append(l)
    logloss_testing.append(logP(testing,CPD_list))
    BIC.append(calBIC(l,CPD_list,M))
    for c in content:
        flag1=False
        flag2=False
        line=c.split("\t")
        #forward
        a = int(line[0])
        b = int(line[1])
        #print(str(a)+" "+str(b))
        modifyGraph(a,b,'a',graph)
        if not isCyclic(graph):
            CPD_list=calCPD(training,graph)
            l1=logP(cross_validation,CPD_list)
            l1_testing=logP(testing,CPD_list)
            BIC1=calBIC(l1,CPD_list,M)
            flag1=True
        modifyGraph(a,b,'r',graph)
        modifyGraph(b,a,'a',graph)
        if not isCyclic(graph):
            CPD_list=calCPD(training,graph)
            l2=logP(cross_validation,CPD_list)
            l2_testing=logP(testing,CPD_list)
            BIC2=calBIC(l2,CPD_list,M)
            Flag2=True
        #print(str(a)+" "+str(b)+" ="+str(l1)+" , "+str(b)+" "+str(a)+" ="+str(l2))
        if criteria=='logLoss':
            if (flag1 and flag2):
                if max(l1,l2)>logloss[-1]:
                    if l2 < l1:
                        modifyGraph(b,a,'r',graph)
                        modifyGraph(a,b,'a',graph)
                        logloss_testing.append(l1_testing)
                    else:
                        logloss_testing.append(l2_testing)
                    logloss.append(max(l1,l2))
                    BIC.append(min(BIC1,BIC2))
                else:
                    modifyGraph(b,a,'r',graph)
            elif flag1:
                modifyGraph(b,a,'r',graph)
                if l1>logloss[-1]:
                    modifyGraph(a,b,'a',graph)
                    logloss_testing.append(l1_testing)
                    logloss.append(l1)
                    BIC.append(BIC1)
            elif flag2:
                if l2>logloss[-1]:
                    logloss_testing.append(l2_testing)
                    logloss.append(l2)
                    BIC.append(BIC2)
                else:
                    modifyGraph(b,a,'r',graph)
            else:
                modifyGraph(b,a,'r',graph)
        else:
            if (flag1 and flag2):
                if min(BIC1,BIC2)<BIC[-1]:
                    if l2 < l1:
                        modifyGraph(b,a,'r',graph)
                        modifyGraph(a,b,'a',graph)
                        logloss_testing.append(l1_testing)
                    else:
                        logloss_testing.append(l2_testing)
                    logloss.append(max(l1,l2))
                    BIC.append(min(BIC1,BIC2))
                else:
                    modifyGraph(b,a,'r',graph)
            elif flag1:
                modifyGraph(b,a,'r',graph)
                if BIC1<BIC[-1]:
                    modifyGraph(a,b,'a',graph)
                    logloss_testing.append(l1_testing)
                    logloss.append(l1)
                    BIC.append(BIC1)
            elif flag2:
                if BIC2<BIC[-1]:
                    logloss_testing.append(l2_testing)
                    logloss.append(l2)
                    BIC.append(BIC2)
                else:
                    modifyGraph(b,a,'r',graph)
            else:
                modifyGraph(b,a,'r',graph)
            
                
    return graph            
    
def modifyGraph(i,j,op_type,graph):
    if op_type=='a':
        graph[i][j]=1
    elif op_type =='r':
        graph[i][j]=0

########################### CPD ####################################
class CPD:  # for each node there is a CPD object
    def __init__(self,parents_num,list_of_parents,columns,my_node_num,type):
        self.parents_num = parents_num # number of all combination of parrents values
        self.columns = columns # number of values for self node
        self.table = np.zeros((self.parents_num,self.columns))
        self.list_of_parents = list_of_parents
        self.my_node_num = my_node_num
        self.type=type
    def getTable(self):
        return self.table
    def getParents(self):
        return self.list_of_parents
    def getMyNodeNumber(self):
        return self.my_node_num
    def calculate_index(self):
        list_max = []
        list_true =[]
        for parent_index in self.list_of_parents:
            list_max.append(getMax(self.type)[parent_index])
            list_true.append(self.record_value[parent_index])
        lm = []
        multiply = 1
        for i in range(0,len(list_max)):
            multiply = list_max[i] * multiply
        for i in range(0,len(list_max)):
            lm.append(multiply/list_max[i])
            multiply = lm[i]
        index = 0
        for i in range(0,len(list_max)):
            index = index+int(lm[i])*int(list_true[i])
        return index

    def calculate_index_independent(self,record):
        list_max = []
        list_true =[]
        for parent_index in self.list_of_parents:
            list_max.append(getMax(self.type)[parent_index])
            list_true.append(record[parent_index])
        lm = []
        multiply = 1;
        for i in range(0,len(list_max)):
            multiply = list_max[i] * multiply
        for i in range(0,len(list_max)):
            lm.append(multiply/list_max[i])
            multiply = lm[i]
        index = 0;    
        for i in range(0,len(list_max)):
            index = index+int(lm[i])*int(list_true[i]);
        return index
    def calculate_parent_value(self,index):
        #print("index i got is "+str(index))
        list_max = []
        lm =[]
        for parent_index in self.list_of_parents:
            list_max.append(getMax(self.type)[parent_index]+1)
        multiply = 1;
        for i in range(1,len(list_max)):
            multiply = list_max[i] * multiply
        for i in range(1,len(list_max)):
            lm.append(index//multiply)
            index=index%multiply
            multiply = multiply/list_max[i]
            #if i==len(list_max)-1:
        if len(list_max)>0:
            lm.append(index)
        
        return lm


    def update(self,r_list): # adds new data to CPD
        self.record_value = r_list
        index = self.calculate_index()
        self.table[index][int(r_list[self.my_node_num])] = self.table[index][int(r_list[self.my_node_num])] +1
    def finalize(self): # fuses prior and occurance, normalizes to calculate probability (call once only it after all updates)
        for row in range(0,self.parents_num):
            for col in range(0,self.columns):
                self.table[row,col]=self.table[row,col]+getAlpha(self.my_node_num,col,self.list_of_parents,self.calculate_parent_value(row),alpha)
            self.table[row,:]=self.table[row,:]/np.sum(self.table[row,:])
    def getLogLoss(self,rec):
        self.record_value = rec
        index = self.calculate_index()
        return self.table[index][int(rec[self.my_node_num])] 
            
def getObjList():
    return CPD_list

def calCPD(samples,graph):
    CPD_list =[]
    for i in range(0,12):
        list_of_parents =[]
        CPD_list.append(CPD(getParents(i,list_of_parents,graph),list_of_parents,getMax(type)[i]+1,i,type))
    for i in range(0,samples.shape[0]):
        for obj in CPD_list:
            obj.update(samples[i,:].tolist())
    for obj in CPD_list:
        obj.finalize()
    return CPD_list
    
def logP(samples,CPD_list):
    log_P=0 
    for i in range(0,samples.shape[0]):
        for obj in CPD_list:
            log_P = log_P+math.log(obj.getLogLoss(samples[i,:].tolist()))
    return log_P
######################### Sampleing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    parent_list = obj.getParents()
    for p in parent_list:
        if val_list[p] is -1:
            iterate_Ancestral(CPD_list[p],val_list)
    if val_list[obj.getMyNodeNumber()] is -1:
        val_list[obj.getMyNodeNumber()] = generate(obj,val_list)
        
def isCyclic(G,*l):
    if len(l) is 0:
        visited_list=[0]*G.shape[0]
        call_num=0
        for i in range(0,G.shape[0]):
            return (False or isCyclic(G,i,visited_list,call_num))
    else:
        visited_list=l[1]
        #print(visited_list)
        #print(l[2])
        if l[2]>G.shape[0]-1:
            return True
        else:
            flag=False
            for j in range(0,G.shape[0]):
                if G[l[0],j]>0:
                    if visited_list[j] is 0:
                        flag=flag or isCyclic(G,j,visited_list,l[2]+1)
            if visited_list[l[0]] is 0:
                visited_list[l[0]]=1
            return flag

            
def getAncestral(M):
    samples =np.zeros((M,12))
    for i in range(0,M):
        val_list = [-1]*12
        for obj in CPD_list:
            iterate_Ancestral(obj,val_list)
        samples[i,:]=np.array(val_list)
    return samples

def getGibbs(M):
    # parameters
    size_to_Mix = 10000
    size_to_skip=1000
    #
    l1 = [0]*12
    l2 = [0]*12
    samples =np.zeros((M,12))
    count=0
    for i in range (0,12):#initialisation
        l1[i] = random.randint(0,getMax(type)[i])    ##########################################
    for t in range(0,size_to_Mix+M*size_to_skip):
        #queue to keep the top 100 values
        for i in range(0,len(CPD_list)):
            l1[i]=generate(CPD_list[i],l1)
        if t>=size_to_Mix and t%size_to_skip==size_to_skip-1:
            samples[count,:]=np.array(l1)
            count=count+1
    return samples

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
        P_table[i,:]= P_table[i,:]/sum_arr[i]
    entropy=np.zeros(D)
    entropy=np.zeros(D)
    for i in range(0,N):
        for j in range(0,D):
            entropy[j]=entropy[j]-math.log(P_table[j,samples[i,j]])
    entropy=entropy/N
    return entropy
    
def entopy(samples,CPD_list):
    N=samples.shape[0]
    return -logP(samples,CPD_list)/N
    
def KL(samples,CPD_list_1,CPD_list_2):
    N=samples.shape[0]
    return (-logP(samples,CPD_list_2)+logP(samples,CPD_list_1))/N
    
def mutualEntropy(samples,CPD_list):
    D=samples.shape[1]
    CPD_list_empty=calCPD(samples,np.zeros((D,D)))
    return KL(samples,CPD_list,CPD_list_empty)

###############################################
training_percent=0.7
cross_validation_percent=0.15
testing_percent=0.15
###############################################
year=11
grade=3
type='c'
alphaRange=[50,25,10,5,1]
bestAlpha=50
num_samples=100
#done
############################ load data #########################
for ii in range(1,10):
    if type is 'c':
            file_sam=("sample/cursive_20"+str(year)+"-20"+str(year+1)+"_"+str(grade)+".txt")
    else:
        file_sam="sample/handwritten_20"+str(year)+"-20"+str(year+1)+"_"+str(grade)+".txt"
    with open(file_sam, 'r') as f:
        content = f.readlines()
    f.close()
    random.shuffle(content)
    num_sample=len(content)
    samples=np.zeros((num_sample,12))
    for i in range(0,num_sample):
        l=content[i].split("\t")[:12]
        for j in range(0,12):
            samples[i][j] = int(l[j])
    training=samples[0:math.floor(training_percent*num_sample),:]
    cross_validation=samples[(math.floor(training_percent*num_sample)+1):math.floor((training_percent+cross_validation_percent)*num_sample),:]
    testing=samples[(math.floor((training_percent+cross_validation_percent)*num_sample)+1):,:]
    ############################## get sat of raw data ########################################
    '''
    print("data for year=20"+str(year)+"-20"+str(year+1)+" grade="+str(grade)+" type="+type)
    m=meanComRare(samples,type)
    d=dist(m[1:,:],m[0,:],type)
    print("mean is:")
    print(m[0,:])
    with open("stats/mean.txt", 'w') as f1:
        for s in m[0,:]:
            f1.write(str(s)+'\t')
        f1.write('\n')
    f1.close()
    print("a common sample in the original data is: "+"(dist=",str(d[0])+")")
    print(m[1,:])
    with open("stats/mean.txt", 'w') as f1:
        for s in m[1,:]:
            f1.write(str(s)+'\t')
        f1.write('\n')
    f1.close()
    print("a rare sample in the original data  is: "+"(dist=",str(d[1])+")")
    print(m[2,:])
    with open("stats/mean.txt", 'w') as f1:
        for s in m[2,:]:
            f1.write(str(s)+'\t')
        f1.write('\n')
    f1.close()
    print("-----------------------------------------------------------")
    entInd=entopyIndep(samples,type)
    print ("Entropy of each feature in the original data is: ")
    print(entInd)
    with open("stats/entropy.txt", 'w') as f1:
        for s in entInd:
            f1.write(str(s)+'\t')
        f1.write('\n')
    f1.close()
    print("----------------------")
    '''
    #############################################################
    independent_prob_table=getIndependentProbability()
    '''
    ########################## study effect of alpha #############
    log_loss_effect_alpha=[]
    for alpha in alphaRange:
        log_loss=[]
        graph=makeStructure(log_loss,[],[],'logLoss')
        log_loss_effect_alpha.append(log_loss)
    print("effect of alpha:")
    print(log_loss_effect_alpha)
    with open("stats/log_loss_effect_alpha.txt", 'w') as f1:
        for s in log_loss_effect_alpha:
            for x in s:
                f1.write(str(x)+',')
            f1.write('\n')
    f1.close()
    print("----------------------")
    '''

    ############## study of overfit: compare logloss, logloss on testing, and BIC ######
    alpha=bestAlpha
    log_loss_training_overfit=[]
    log_loss_testing_overfit=[]
    BIC_overfit=[]
    print("Training in process")
    t1=time.time()
    graph=makeStructure(log_loss_training_overfit,log_loss_testing_overfit,BIC_overfit,'logLoss')
    t2=time.time()
    print("Trained graph using original method:")
    print(graph)
    CPD_list_1=calCPD(training,graph)
    '''
    with open("stats/trained_graph.txt", 'w') as f1:
        for s in graph:
            s = s.tolist()
            for x in s:
                f1.write(str(x)+',')
            f1.write('\n')
    f1.close()
    '''
    print("Training time for the original method "+str(t2-t1)+"sec")
    print("log loss on training data: (original method)")
    print(log_loss_training_overfit)
    print("log loss on testing data: (original method)")
    print(log_loss_testing_overfit)
    print("BIC: (original method)")
    print(BIC_overfit)
    '''
    with open("stats/log_loss_training_overfit.txt", 'w') as f1:
        for s in log_loss_training_overfit:
            f1.write(str(s)+'\t')
        f1.write('\n')
    f1.close()
    with open("stats/log_loss_testing_overfit.txt", 'w') as f1:
        for s in log_loss_testing_overfit:
            f1.write(str(s)+'\t')
        f1.write('\n')
    f1.close()
    with open("stats/BIC_overfit.txt", 'w') as f1:
        for s in BIC_overfit:
            f1.write(str(s)+'\t')
        f1.write('\n')
    f1.close()
    '''
    print("-----------------------------")
    ############## proposed method: compare logloss, logloss on testing, and BIC ######
    alpha=bestAlpha
    log_loss_training=[]
    log_loss_testing=[]
    BIC=[]
    print("Training in process")
    t1=time.time()
    graph=makeStructure(log_loss_training,log_loss_testing,BIC,'BIC')
    t2=time.time()
    print("Trained graph using proposed method:")
    print(graph)
    CPD_list_2=calCPD(training,graph)
    '''
    with open("stats/trained_graph_proposed.txt", 'w') as f1:
        for s in graph:
            s = s.tolist()
            for x in s:
                f1.write(str(x)+',')
            f1.write('\n')
    f1.close()
    '''
    print("Training time for the proposed method "+str(t2-t1)+"sec")
    print("log loss on training data: (proposed method)")
    print(log_loss_training)
    print("log loss on testing data: (proposed method)")
    print(log_loss_testing)
    print("BIC: (proposed method)")
    print(BIC)
    with open("stats/BICprop.txt", 'w') as f1:
        for s in BIC:
            f1.write(str(s)+'\t')
        f1.write('\n')
    f1.close()
    print("-----------------------------")

    print(KL(testing,CPD_list_1,CPD_list_2))


'''
######################### stat of original data on the trained BN  ####################################
CPD_list = calCPD(training,graph)
print("entropy:")
print(entopy(samples,CPD_list))
print("mutual entropy:")
print(mutualEntropy(samples,CPD_list))
with open("stats/ent.txt", 'w') as f1:
    f1.write(str(entopy(samples,CPD_list))+'\n')
    f1.write(str(mutualEntropy(samples,CPD_list))+'\n')
f1.close()
######################### generating samples from the BN ############################################
Ancestral_samples=getAncestral(num_samples)
print("Ancestral sample:")
print(Ancestral_samples)
with open("stats/Ancestral_samples.txt", 'w') as f1:
    for s in Ancestral_samples:
        for x in s:
            f1.write(str(x)+'\t')
        f1.write('\n')
f1.close()
An_m=meanComRare(Ancestral_samples,type)
d=dist(An_m[0:,:],m[0,:],type)
print("ancestral samples mean is:")
print(An_m[0,:])
with open("stats/mean.txt", 'a') as f1:
    for s in An_m[0,:]:
        f1.write(str(s)+'\t')
    f1.write('\n')
f1.close()
print("a common sample in the ancestral samples is: "+"(dist=",str(d[1])+")")
print(An_m[1,:])
with open("stats/mean.txt", 'a') as f1:
    for s in An_m[1,:]:
        f1.write(str(s)+'\t')
    f1.write('\n')
f1.close()
print("a rare sample in the ancestral samples is: "+"(dist=",str(d[2])+")")
print(An_m[2,:])
with open("stats/mean.txt", 'a') as f1:
    for s in An_m[2,:]:
        f1.write(str(s)+'\t')
    f1.write('\n')
f1.close()
print("-----------------------------------------------------------")
print("entropy of the ancestral samples:")
print(entopy(Ancestral_samples,CPD_list))
print("mutual entropy of the ancestral samples:")
print(mutualEntropy(Ancestral_samples,CPD_list))
print("----------------------------------------------------------------")
with open("stats/ent.txt", 'w') as f1:
    f1.write(str(entopy(Ancestral_samples,CPD_list))+'\n')
    f1.write(str(mutualEntropy(Ancestral_samples,CPD_list))+'\n')
f1.close()
Gibbs_samples=getGibbs(num_samples)
print("Gipps sample:")
print(Gibbs_samples)
with open("stats/Gibbs_sample.txt", 'w') as f1:
    for s in Gibbs_samples:
        for x in s:
            f1.write(str(x)+'\t')
        f1.write('\n')
f1.close()
Gi_m=meanComRare(Gibbs_samples,type)
d=dist(Gi_m[0:,:],m[0,:],type)
print("Gipps sample mean is:")
print(Gi_m[0,:])
with open("stats/mean.txt", 'a') as f1:
    for s in Gi_m[0,:]:
        f1.write(str(s)+'\t')
    f1.write('\n')
f1.close()
print("a common sample in the Gibb's samples is: "+"(dist=",str(d[1])+")")
print(Gi_m[1,:])
with open("stats/mean.txt", 'a') as f1:
    for s in Gi_m[1,:]:
        f1.write(str(s)+'\t')
    f1.write('\n')
f1.close()
print("a rare sample in the Gibb's samples is: "+"(dist=",str(d[2])+")")
print(Gi_m[2,:])
with open("stats/mean.txt", 'a') as f1:
    for s in Gi_m[2,:]:
        f1.write(str(s)+'\t')
    f1.write('\n')
f1.close()
print("-----------------------------------------------------------")
print("entropy of the Gibb's samples:")
print(entopy(Gibbs_samples,CPD_list))
print("mutual entropy of the Gibb's samples:")
print(mutualEntropy(Gibbs_samples,CPD_list))
print("----------------------")
with open("stats/ent.txt", 'a') as f1:
    f1.write(str(entopy(Gibbs_samples,CPD_list))+'\n')
    f1.write(str(mutualEntropy(Gibbs_samples,CPD_list))+'\n')
f1.close()
#########################

'''

'''
############################### temporal study ####################################
yearList=[11,12,13]
gradeList=[3,4,5]
type='p'
for i in range(0,3):
    year=yearList[i]
    grade=gradeList[i]
    if type is 'c':
        file_sam=("sample/cursive_20"+str(year)+"-20"+str(year+1)+"_"+str(grade)+".txt")
    else:
        file_sam="sample/handwritten_20"+str(year)+"-20"+str(year+1)+"_"+str(grade)+".txt"
    with open(file_sam, 'r') as f:
        content = f.readlines()
    f.close()
    random.shuffle(content)
    num_sample=len(content)
    samples=np.zeros((num_sample,12))
    for j in range(0,num_sample):
        l=content[j].split("\t")[:12]
        for k in range(0,12):
            samples[j][k] = int(l[k])
    training=samples[0:math.floor(training_percent*num_sample),:]
    cross_validation=samples[(math.floor(training_percent*num_sample)+1):math.floor((training_percent+cross_validation_percent)*num_sample),:]
    testing=samples[(math.floor((training_percent+cross_validation_percent)*num_sample)+1):,:]
    #
    independent_prob_table=getIndependentProbability()
    alpha=bestAlpha
    #
    print("Training in process")
    graph=makeStructure([],[],[],'BIC')
    print(graph)
    with open("tempo/graph 20"+str(yearList[i])+" grade"+str(gradeList[i])+".txt", 'w') as f1:
        for s in graph:
            s = s.tolist()
            for x in s:
                f1.write(str(x)+',')
            f1.write('\n')
    f1.close()
    #
    print ("Entropy of each feature in the original data is: ")
    entInd=entopyIndep(samples,type)
    print(entInd)
    # sample
    CPD_list = calCPD(training,graph)
    Ancestral_samples=getAncestral(num_samples)
    print ("Entropy of each feature in ancestral samples is: ")
    entInd_An=entopyIndep(Ancestral_samples,type)
    print(entInd_An)
    with open("tempo/entropy 20"+str(yearList[i])+" grade"+str(gradeList[i])+".txt", 'w') as f1:
        s = entInd.tolist()
        for x in s:
            f1.write(str(x)+',')
        f1.write('\n')
        s = entInd_An.tolist()
        for x in s:
            f1.write(str(x)+',')
        f1.write('\n')
    f1.close()
    '''