from Bayesian import *
from Prior import getAlpha
from __builtin__ import str


class CPD:  # for each node there is a CPD object
    def __init__(self,parents_num,list_of_parents,columns,my_node_num,type):
        self.parents_num = parents_num # number of all combination of parents values
        self.columns = columns # number of values for self node
        #self.table = np.ones((self.parents_num,self.columns))
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
        #print("parents")
        #print(self.list_of_parents)
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
        """print("list max")
        print(list_max)
        print("lm")
        print(lm)"""
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
        """print("list of parents")
        print(self.list_of_parents)"""
        index = self.calculate_index()
        """ print(index)
        print(self.my_node_num)"""
        
        self.table[index][int(r_list[self.my_node_num])] = self.table[index][int(r_list[self.my_node_num])] +1
        
        
        
    def finalize(self): # fuses prior and occurance, normalizes to calculate probability (call once only it after all updates)
       
        for row in range(0,self.parents_num):
            for col in range(0,self.columns):
                al = getAlpha(self.my_node_num,col,self.list_of_parents,self.calculate_parent_value(row),alpha)
                self.table[row,col]=self.table[row,col]+al
            self.table[row,:]=self.table[row,:]/np.sum(self.table[row,:])
        
            
    def getLogLoss(self,rec):
        self.record_value = rec
        index = self.calculate_index()
        return self.table[index][int(rec[self.my_node_num])] 


def calCPD(samples,graph,num):
    CPD_list =[]
    for i in range(0,num):
        list_of_parents =[]
        CPD_list.append(CPD(getParents(i,list_of_parents,graph),list_of_parents,getMax(type)[i]+1,i,type))
    for i in range(0,samples.shape[0]):
        for j in range(0,num): 
           CPD_list[j].update(samples[i,:].tolist())
    
    
    for j in range(0,num):  
       CPD_list[j].finalize()
    return CPD_list

