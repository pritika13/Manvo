'''
Created on Apr 6, 2015

@author: pritika
'''
from Bayesian import *
from GenerateCPD import calCPD
import Bayesian

def getfile_chi(flag):
    if flag is 0:
        if type is 'c':
            file_chi=("chi/cursive_20"+str(year)+"-20"+str(year+1)+"_"+str(grade)+".txt")
        else:
            file_chi="chi/handwritten_20"+str(year)+"-20"+str(year+1)+"_"+str(grade)+".txt"
    else:
        file_chi = "chi/file"+str(flag)+".txt"   
         
    
    return file_chi

def modifyGraph(i,j,op_type,graph):
    if op_type=='a':
        graph[i][j]=1
    elif op_type =='r':
        graph[i][j]=0
        
def calBIC(logloss,CPD_list,M):
    k=0;
    for CPD in CPD_list:
        k=k+math.log(CPD.table.size)
    return -logloss+k*math.log(M)/2

def isCyclic(G,*l):
    if len(l) is 0:
        visited_list=[0]*G.shape[0]
        call_num=0
        for i in range(0,G.shape[0]):
            return (False or isCyclic(G,i,visited_list,call_num))
    else:
        visited_list=l[1]
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

def makeStructure(logloss,logloss_testing,BIC,criteria,num,graph,objList):
    global cross_validation
    global training
    global testing
    global CPD_last
    
    CPD_total = CPD_last
    

    cross_validation1 = np.array(cross_validation)[:,:num]
    training1 = np.array(training)[:,:num]
    testing1 = np.array(testing)[:,:num]
    
    
            
    with open(getfile_chi(num), 'r') as f:
        content = f.readlines()
    f.close()
    i =0
    
    #content = content[:5]####################################################
    M = cross_validation1.shape[0]
   
    CPD_list =[]
    # for more than 12 edges
    CPD_list =calCPD(training1,graph,num)
   
    """CPD_list_got=calCPD(training,graph,num)
    if len(CPD_total) > 0 and len(CPD_list_got) >0:
        CPD_list = CPD_total
        for obj in CPD_list_got:
            CPD_list.append(obj)  
    else:
        CPD_list = CPD_list_got   """ 
    
    l=logP(cross_validation1,CPD_list)
    logloss.append(l)
    logloss_testing.append(logP(testing1,CPD_list))
    BIC.append(calBIC(l,CPD_list,M))
    s =0
    a=0
    b=0
    for c in content:
        flag1=False
        flag2=False
        line=c.split("\t")
        #print("line is" )
        #print(line)
        if len(line)<2:
            continue
        s = s+1
        if s > 20:
            break
        #forward
        a = int(line[0])
        b = int(line[1])
        l1 =0.0
        l2=0.0
        #print(str(a)+" "+str(b))
        modifyGraph(a,b,'a',graph)
        if not isCyclic(graph):
            CPD_list =calCPD(training1,graph,num)
            l1=logP(cross_validation1,CPD_list)
            l1_testing=logP(testing1,CPD_list)
            BIC1=calBIC(l1,CPD_list,M)
            flag1=True
        modifyGraph(a,b,'r',graph)
        modifyGraph(b,a,'a',graph)
        if not isCyclic(graph):
            CPD_list =calCPD(training1,graph,num)
            l2=logP(cross_validation1,CPD_list)
            l2_testing=logP(testing1,CPD_list)
            BIC2=calBIC(l2,CPD_list,M)
            flag2=True
       
        if criteria == 'logLoss':
            if (flag1 and flag2):
               
                if max(l1,l2)>=logloss[-1]:
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
                if l1>=logloss[-1]:
                    modifyGraph(a,b,'a',graph)
                    logloss_testing.append(l1_testing)
                    logloss.append(l1)
                    BIC.append(BIC1)
            elif flag2:
                
                if l2>=logloss[-1]:
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
    
    CPD_last = CPD_list;
    
    for obj in CPD_last:
        objList.append(obj)
     
   
    
    return graph            
            
             

def makeStructure24(logloss,logloss_testing,BIC,criteria,graph12,objList):
    graph24 = getEmptyGraph(24)
    global training
    for i in range(0,12):
        j=0
        for j in range(0,12):
            graph24[i][j] = graph12[i][j]
       
        graph24[i][j+i+1] = 1  
   
    graph24 = makeStructure(log_loss_training,log_loss_testing,BIC,'logLoss',24,graph24,objList)  
    return graph24

def makeStructure36(logloss,logloss_testing,BIC,criteria,graph24,objList):
    graph36 = getEmptyGraph(36)
    global training
    for i in range(0,24):
        j=0
        for j in range(0,24):
            graph36[i][j] = graph24[i][j]
    
    j = 24     
    for i in range(12,24):
        graph36[i][j] = 1
        j = j+1  
          
    
   
    graph36 = makeStructure(log_loss_training,log_loss_testing,BIC,'logLoss',36,graph36,objList)  
    return graph36       
           