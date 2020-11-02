import csv
import random
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

def partition(filename):
    train=[['nuc', 'A', 'R', 'N', 'D', 'Q']]
    test=[['nuc', 'A', 'R', 'N', 'D', 'Q']]
    with open(filename+'.csv','r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line!=['nuc', 'A', 'R', 'N', 'D', 'Q']:
                t=random.random()
                if t <0.7:
                    train.append(line)
                else:
                    test.append(line)
    with open(filename+'_train.csv','w',encoding="utf-8") as csvfile:
        writer=csv.writer(csvfile)
        for line in train:
            writer.writerow(line)
    with open(filename+'_test.csv','w',encoding="utf-8") as csvfile:
        writer=csv.writer(csvfile)
        for line in test:
            writer.writerow(line)

def main():
    bn = gum.BayesNet('nuc_inf')
    #add variables to the network
    va=gum.LabelizedVariable('nuc','a labelized variable',2)
    va.addLabel('-1')
    nuc = bn.add(va)
    A = bn.add('A',6)
    R,N = [bn.add(name,7) for name in ['R','N']]
    D,Q = [bn.add(name,2) for name in ['D','Q']]
    partition("protein")
    learner = gum.BNLearner("protein_train.csv", bn)
    #These arcs can be added or deleted
    #learner.addMandatoryArc('A','nuc')
    #learner.addMandatoryArc('R','nuc')
    #learner.addMandatoryArc('Q','nuc')
    #learner.addMandatoryArc('N','nuc')
    #learner.addMandatoryArc('D','nuc')
    learner.useLocalSearchWithTabuList()
    bn0 = learner.learnBN()
    gnb.showBN(bn0)
    learner.useGreedyHillClimbing()
    bn1 = learner.learnBN()
    gnb.showBN(bn1)
    learner.useK2([5,4,3,2,1,0])
    bn2 = learner.learnBN()
    gnb.showBN(bn2)
    #We have 2 different BN structures according to the previous parts. Now, we do parameter learning
    learner = gum.BNLearner("protein_train.csv", bn)
    learner.setInitialDAG(bn0.dag())
    learner.useAprioriSmoothing(1)
    bn01 = learner.learnParameters()#first
    gnb.showBN(bn01)
    learner = gum.BNLearner("protein_train.csv", bn)
    learner.setInitialDAG(bn2.dag())
    learner.useAprioriSmoothing(1)
    bn11 = learner.learnParameters()#second
    gnb.showBN(bn11)
    #first
    ie1 = gum.LazyPropagation(bn01)
    ie1.makeInference()
    gnb.showInference(bn01,evs={})
    #second
    ie2 = gum.LazyPropagation(bn11)
    ie2.makeInference()
    gnb.showInference(bn11,evs={})
    with open('protein_test.csv','r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        count1=1
        count2=1
        acc1=0
        acc2=0
        for line in list(reader)[1:]:
            vnuc,vA,vR,vN,vD,vQ=[int(line[0]),int(line[1]),int(line[2]),int(line[3]),int(line[4]),int(line[5])]
            #print(vnuc,vA,vR,vN,vD,vQ)
            ie2.eraseAllEvidence()
            ie1.eraseAllEvidence()
            ie1.setEvidence({'A':vA, 'R':vR,'N':vN, 'D': vD,'Q':vQ})
            ie2.setEvidence({'A':vA, 'R': vR,'N':vN, 'D': vD,'Q':vQ})
            ie1.makeInference()
            ie2.makeInference()
            ie2.addTarget(nuc)
            ie1.addTarget(nuc)
            if len(ie2.posterior(nuc).argmax())==1:#if we have one determined value of prob
                #print(ie2.posterior(nuc))
                #print(ie2.posterior(nuc).argmax()[0]['nuc'])
                if ie2.posterior(nuc).argmax()[0]['nuc']==2:#nuc=-1
                    if vnuc==-1:
                        acc2=acc2+1
                if ie2.posterior(nuc).argmax()[0]['nuc']==vnuc:
                    acc2=acc2+1
                count2=count2+1
            if len(ie1.posterior(nuc).argmax())==1:
                #print(ie1.posterior(nuc))
                #print(ie1.posterior(nuc).argmax()[0]['nuc'])
                if ie1.posterior(nuc).argmax()[0]['nuc']==2:
                    if vnuc==-1:
                        acc1=acc1+1
                if ie1.posterior(nuc).argmax()[0]['nuc']==vnuc:
                    acc1=acc1+1
                count1=count1+1
        acc2=acc2/count2
        acc1=acc1/count1
    print(acc2,acc1)

if __name__=='__main__':
    main()
