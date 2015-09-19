from Bayesian import *

# get probability for each individual node
def getIndependentProbability(n):
    #n = 12
    global independent_prob_table
    global training
    training = np.array(training)
    minimumP=0.001
    independent_prob_table=np.zeros((n,6+1))
    for i in range(0,n):
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
    global independent_prob_table
    
    Pprime = independent_prob_table[node][nodeValue]
    
    for p in range(0,len(parentList)):    
        Pprime = Pprime*independent_prob_table[parentList[p]][parentsValue[p]]
    return alpha*Pprime    



