'''
Created on Apr 6, 2015

@author: pritika
'''
from Bayesian import *
from GenerateCPD import calCPD
from Bayesian.LoadData import *
from Prior import *
from LearnStructure import makeStructure
from LearnStructure import makeStructure24
from LearnStructure import makeStructure36
from Sampling import *
from Statistics import *
from chiSquare import getChi
import time



def CalLocalLogP(sample,CPD_list,beg,end):
        log_P=0 
        for  i in range(beg,end+1):
            val = CPD_list[i].getLogLoss(sample)
            log_P = log_P+math.log(val)
            #log_P = log_P+val
        return log_P

def main():
    num = 36
    
    #load data
    global training
    global cross_validation
    global testing
    global independent_prob_table
    global log_loss_testing
    global log_loss_training
    global BIC
    global CPD_list
    
    
    loadData(num)
  
    
    #Learn structure
   
    graph = getEmptyGraph(num)
    independent_prob_table=getIndependentProbability(num)
    #chi = getChi(training, num,independent_prob_table)
    
    objList =[]
    graph12 = makeStructure(log_loss_training,log_loss_testing,BIC,'logLoss',12,graph,objList)
    objList =[]
    graph24 = makeStructure24(log_loss_training,log_loss_testing,BIC,'logLoss',graph12,objList)
    objList =[]
    graph36 = makeStructure36(log_loss_training,log_loss_testing,BIC,'logLoss',graph24,objList)
    with open("results/abc.txt", 'w') as f1:
        for i in range(0,36):
            for j in range(0,36):
                f1.write(str(graph36[i][j])+"\t")
            f1.write('\n')
    f1.close()
    
    global finalObjList
    for obj in objList:
        finalObjList.append(obj)
    #sampling
  
   
    
    ###########normal
    samples = getAllSamples(num)
    entInd_ori=entopyIndep(samples,type) 
    entInd_ori1 = entInd_ori[0:12]
    entInd_ori2 = entInd_ori[13:24]
    entInd_ori3 = entInd_ori[25:36]
    
    entInd_ori1 = ['{:.8f}'.format(x) for x in entInd_ori1]
    entInd_ori2 = ['{:.8f}'.format(x) for x in entInd_ori2]
    entInd_ori3 = ['{:.8f}'.format(x) for x in entInd_ori3]
    
    m=meanComRare(samples,type)
    
    mean1 = m[0][0:12]
    mean2 = m[0][13:24]
    mean3 = m[0][25:36]
    mean1 = ['{:.8f}'.format(x) for x in mean1]
    mean2 = ['{:.8f}'.format(x) for x in mean2]
    mean3 = ['{:.8f}'.format(x) for x in mean3]
    
    com1  = m[1][0:12]
    com2  = m[1][0:12]
    com3  = m[1][0:12]
    com1 = ['{:.8f}'.format(x) for x in com1]
    com2 = ['{:.8f}'.format(x) for x in com2]
    com3 = ['{:.8f}'.format(x) for x in com3]
    
    rare1 = m[2][0:12]
    rare2 = m[2][13:24]
    rare3 = m[2][25:36]
    rare1 = ['{:.8f}'.format(x) for x in rare1]
    rare2 = ['{:.8f}'.format(x) for x in rare2]
    rare3 = ['{:.8f}'.format(x) for x in rare3]
    
    d=dist(m[1:,:],m[0,:],type)
    
    
    
    ###########ancestral
   
    Ancestral_samples=getAncestral_test(samples,num_samples,num)
    entInd_An=entopyIndep(Ancestral_samples,type)
    entInd_An1 = entInd_An[0:12]
    entInd_An2 = entInd_An[13:24]
    entInd_An3 = entInd_An[25:36]
    entInd_An1 = ['{:.8f}'.format(x) for x in entInd_An1]
    entInd_An2 = ['{:.8f}'.format(x) for x in entInd_An2]
    entInd_An3 = ['{:.8f}'.format(x) for x in entInd_An3]
    
    An_m=meanComRare(Ancestral_samples,type)
    
   
    meanAn1 = An_m[0][0:12]
    meanAn2 = An_m[0][13:24]
    meanAn3 = An_m[0][25:36]
    meanAn1 = ['{:.8f}'.format(x) for x in meanAn1]
    meanAn2 = ['{:.8f}'.format(x) for x in meanAn2]
    meanAn3 = ['{:.8f}'.format(x) for x in meanAn3]
    

   

    comAn1  = An_m[1][0:12]
    comAn2  = An_m[1][0:12]
    comAn3  = An_m[1][0:12]
    comAn1 = ['{:.8f}'.format(x) for x in comAn1]
    comAn2 = ['{:.8f}'.format(x) for x in comAn2]
    comAn3 = ['{:.8f}'.format(x) for x in comAn3]
    
    rareAn1 = An_m[2][0:12]
    rareAn2 = An_m[2][13:24]
    rareAn3 = An_m[2][25:36]
    rareAn1 = ['{:.8f}'.format(x) for x in rareAn1]
    rareAn2 = ['{:.8f}'.format(x) for x in rareAn2]
    rareAn3 = ['{:.8f}'.format(x) for x in rareAn3]
    
    ##############
    with open("stats/mean2An.txt", 'w') as f1:
       
        
        f1.write(",".join(meanAn1)+'\n')
        f1.write(",".join(comAn1)+'\n')
        f1.write(",".join(rareAn1)+'\n')
        
        
    f1.close()
    
    
    with open("stats/mean3An.txt", 'w') as f1:
        f1.write(",".join(meanAn2)+'\n')
        f1.write(",".join(comAn2)+'\n')
        f1.write(",".join(rareAn2)+'\n')
        
        
    f1.close()
    
    with open("stats/mean4An.txt", 'w') as f1:
       
        
        f1.write(",".join(meanAn3)+'\n')
        f1.write(",".join(comAn3)+'\n')
        f1.write(",".join(rareAn3)+'\n')
        
        
    f1.close()
    #############
    
    
    d1=dist(An_m[0:,:],m[0,:],type)
    
    
    ###########gibbs
    
    Gibbs_samples=getGibbs(num_samples,num)
    entInd_Gibb  = entopyIndep(Gibbs_samples,type)
    entInd_Gibb1 = entInd_Gibb[0:12]
    entInd_Gibb2 = entInd_Gibb[13:24]
    entInd_Gibb3 = entInd_Gibb[25:36]
    entInd_Gibb1 = ['{:.8f}'.format(x) for x in entInd_Gibb1]
    entInd_Gibb2 = ['{:.8f}'.format(x) for x in entInd_Gibb2]
    entInd_Gibb3 = ['{:.8f}'.format(x) for x in entInd_Gibb3]
    
    Gb_m = meanComRare(Gibbs_samples,type)
    
    meanGb1 = Gb_m[0][0:12]
    meanGb2 = Gb_m[0][13:24]
    meanGb3 = Gb_m[0][25:36]
    meanGb1 = ['{:.8f}'.format(x) for x in meanGb1]
    meanGb2 = ['{:.8f}'.format(x) for x in meanGb2]
    meanGb3 = ['{:.8f}'.format(x) for x in meanGb3]
    
    comGb1  = Gb_m[1][0:12]
    comGb2  = Gb_m[1][0:12]
    comGb3  = Gb_m[1][0:12]
    comGb1 = ['{:.8f}'.format(x) for x in comGb1]
    comGb2 = ['{:.8f}'.format(x) for x in comGb2]
    comGb3 = ['{:.8f}'.format(x) for x in comGb3]
    
    rareGb1 = Gb_m[2][0:12]
    rareGb2 = Gb_m[2][13:24]
    rareGb3 = Gb_m[2][25:36]
    rareGb1 = ['{:.8f}'.format(x) for x in rareGb1]
    rareGb2 = ['{:.8f}'.format(x) for x in rareGb2]
    rareGb3 = ['{:.8f}'.format(x) for x in rareGb3]
    
    d2 = dist(Gb_m[0:,:],m[0,:],type)
   
    ######write entroyp
    with open("stats/ent2.txt", 'w') as f1:
        f1.write(",".join(entInd_ori1)+'\n')
        f1.write(",".join(entInd_An1)+'\n')
        f1.write(",".join(entInd_Gibb1)+'\n')
    f1.close()
    
    with open("stats/ent3.txt", 'w') as f1:
        f1.write(",".join(entInd_ori2)+'\n')
        f1.write(",".join(entInd_An2)+'\n')
        f1.write(",".join(entInd_Gibb2)+'\n')
    f1.close()
    
    with open("stats/ent4.txt", 'w') as f1:
        f1.write(",".join(entInd_ori3)+'\n')
        f1.write(",".join(entInd_An3)+'\n')
        f1.write(",".join(entInd_Gibb3)+'\n')
    f1.close()

    ####write mean common rare
    
    with open("stats/mean2.txt", 'w') as f1:
        f1.write(",".join(mean1)+'\n')
        f1.write(",".join(com1)+'\n')
        f1.write(",".join(rare1)+'\n')
        
        f1.write(",".join(meanAn1)+'\n')
        f1.write(",".join(comAn1)+'\n')
        f1.write(",".join(rareAn1)+'\n')
        
        f1.write(",".join(meanGb1)+'\n')
        f1.write(",".join(comGb1)+'\n')
        f1.write(",".join(rareGb1)+'\n')
    f1.close()
    
    
    with open("stats/mean3.txt", 'w') as f1:
        f1.write(",".join(mean2)+'\n')
        f1.write(",".join(com2)+'\n')
        f1.write(",".join(rare2)+'\n')
        
        f1.write(",".join(meanAn2)+'\n')
        f1.write(",".join(comAn2)+'\n')
        f1.write(",".join(rareAn2)+'\n')
        
        f1.write(",".join(meanGb2)+'\n')
        f1.write(",".join(comGb2)+'\n')
        f1.write(",".join(rareGb2)+'\n')
    f1.close()
    
    with open("stats/mean4.txt", 'w') as f1:
        f1.write(",".join(mean3)+'\n')
        f1.write(",".join(com3)+'\n')
        f1.write(",".join(rare3)+'\n')
        
        f1.write(",".join(meanAn3)+'\n')
        f1.write(",".join(comAn3)+'\n')
        f1.write(",".join(rareAn3)+'\n')
        
        f1.write(",".join(meanGb3)+'\n')
        f1.write(",".join(comGb3)+'\n')
        f1.write(",".join(rareGb3)+'\n')
    f1.close()
    #######################################



    comm_2011 = [ 0.,1.,0.,1.,1.,0.,1.,0.,3.,0.,0.,0. ]
    rare_2011 = [0.,1.,0.,1.,1.,0.,3.,0.,0.,0.,4.,2.]
    comm_2012 = [ 0.,1.,0.,1.,1.,0.,1.,1.,3.,1.,1.,1.]
    rare_2012 = [5.,1.,0.,1.,1.,2.,1.,0.,3.,1.,1.,1.   ]
    comm_2013 = [ 0.,1., 0.,1.,1.,0.,3.,1.,3.,1.,1.,1.]
    rare_2013 = [ 0.,1.,0.,0,1.,3.,3.,4.,0.,0.,1.,1.]
    
    c11_c12_00 = [ 0.,1.,0.,1.,1.,0.,1.,0.,3.,0.,0.,0.,0.,1.,0.,1.,1.,0.,1.,1.,3.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0. ]
    c11_r12_00 = [ 0.,1.,0.,1.,1.,0.,1.,0.,3.,0.,0.,0.,5.,1.,0.,1.,1.,2.,1.,0.,3.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    c11_00_00 =[ 0.,1.,0.,1.,1.,0.,1.,0.,3.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    
    r11_c12_00 = [0.,1.,0.,1.,1.,0.,3.,0.,0.,0.,4.,2., 0.,1.,0.,1.,1.,0.,1.,1.,3.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0. ]
    r11_00_00 =[0.,1.,0.,1.,1.,0.,3.,0.,0.,0.,4.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    
    c11_c12_c13 = [ 0.,1.,0.,1.,1.,0.,1.,0.,3.,0.,0.,0.,0.,1.,0.,1.,1.,0.,1.,1.,3.,1.,1.,1.,0.,1., 0.,1.,1.,0.,3.,1.,3.,1.,1.,1.]
    c11_c12_r13 =[0.,1.,0.,1.,1.,0.,1.,0.,3.,0.,0.,0.,0.,1.,0.,1.,1.,0.,1.,1.,3.,1.,1.,1.,0.,1.,0.,0,1.,3.,3.,4.,0.,0.,1.,1.]
    
    print("common common ")
    print(math.exp(CalLocalLogP(c11_c12_00,finalObjList,0,23)-CalLocalLogP(c11_00_00,finalObjList,0,11)))
    print("common rare ")
    print(math.exp(CalLocalLogP(c11_r12_00,finalObjList,0,23)-CalLocalLogP(c11_00_00,finalObjList,0,11)))
    print("rare common ")
    print(math.exp(CalLocalLogP(r11_c12_00,finalObjList,0,23)-CalLocalLogP(r11_00_00,finalObjList,0,11)))
    print("common common common")
    print(math.exp(CalLocalLogP(c11_c12_c13,finalObjList,0,35)-CalLocalLogP(c11_c12_00,finalObjList,0,23)))
    print("common common rare")
    print(math.exp(CalLocalLogP(c11_c12_r13,finalObjList,0,35)-CalLocalLogP(c11_c12_00,finalObjList,0,23)))
   
    

    
if __name__ == "__main__":
    main()

