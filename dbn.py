import pyAgrum as gum
import numpy as np
import pandas as pd
import math
import pyAgrum.lib.notebook as gnb
import csv
import matplotlib.pyplot as plt

def genre(s):
    r=[]
    for i in range(1,len(s)):
        r.append(s[i]/s[i-1])
    return r

def genbin(l):#discretize the price
    l_b=list(np.arange(0,math.ceil(max(l))+1,1))
    return list(pd.cut(l,bins=l_b,labels=False)),len(l_b)

def genbinv(v):#discretize the volume
    v_b=list(np.linspace(0,math.ceil(max(v)),2))
    return list(pd.cut(v,bins=v_b,labels=False)),len(v_b)

def gentt(filename):#generate train data and test data return the number of variables 
    df = pd.read_csv(filename+'.csv')
    df.dropna(axis=0, how='any', inplace=True)#drop the line with NAN in case there is missing data in the file
    Date=df['Date']
    index=list(Date).index('2015-11-13')#find the index of 2015-11-13, we need to slice the list later
        #generate returns and then discretized variables. 
    Open,ob=genbin(genre(df['Open']))
    High,hb=genbin(genre(df['High']))
    Low,lb=genbin(genre(df['Low']))
    Close,cb=genbin(genre(df['Close']))
    Volume,vb=genbinv(genre(df['Volume']))
    train_Open=Open[:index]
    test_Open=Open[index-1:]
    train_High=High[:index]
    test_High=High[index-1:]
    train_Low=Low[:index]
    test_Low=Low[index-1:]
    train_Close=Close[:index]
    test_Close=Close[index-1:]
    train_Volume=Volume[:index]
    test_Volume=Volume[index-1:]
    train=pd.DataFrame()#The train data
    test=pd.DataFrame()#The test data
    train['Close0']=train_Close[0:-1]#at t-1
    train['Close1']=train_Close[1:]#at t
    train['Open0']=train_Open[0:-1]#at t-1
    train['Open1']=train_Open[1:]#at t
    train['High0']=train_High[0:-1]#at t-1
    train['High1']=train_High[1:]#at t
    train['Low0']=train_Low[0:-1]#at t-1
    train['Low1']=train_Low[1:]#at t
    train['Volume0']=train_Volume[0:-1]#at t-1
    train['Volume1']=train_Volume[1:]#at t
    test['Close0']=test_Close[0:-1]#at t-1
    test['Open0']=test_Open[0:-1]#at t-1
    test['High0']=test_High[0:-1]#at t-1
    test['Low0']=test_Low[0:-1]#at t-1
    test['Volume0']=test_Volume[0:-1]#at t-1
    #####Generate the boolean var for accuracy calculation
    true=[]
    h=list(df['High'])[index:]
    c=list(df['Close'])[index:]
    for i in range(len(h)-1):
        if c[i]<h[i+1]:
            true.append(1)
        else:
            true.append(0)
    test['true']=true
    train.set_index('Close0', inplace=True)
    train.to_csv(filename+'_train.csv')
    test.set_index('Close0', inplace=True)
    test.to_csv(filename+'_test.csv')
    return ob,hb,lb,cb,vb

def trainmodel(filename):
    ob,hb,lb,cb,vb=gentt(filename)
    #print(ob,hb,lb,cb)
    #build the model
    bn = gum.BayesNet(filename)
    Open0=bn.add('Open0',ob)
    High0=bn.add('High0',hb)
    Low0=bn.add('Low0',lb)
    Close0=bn.add('Close0',cb)
    Volume0=bn.add('Volume0',vb)
    Open1=bn.add('Open1',ob)
    High1=bn.add('High1',hb)
    Low1=bn.add('Low1',lb)
    Close1=bn.add('Close1',cb)
    Volume1=bn.add('Volume1',vb)
    learner = gum.BNLearner(filename+"_train.csv", bn)
    learner.addForbiddenArc('Open1','Open0')
    learner.addForbiddenArc('Open1','Close0')
    learner.addForbiddenArc('Open1','High0')
    learner.addForbiddenArc('Open1','Low0')
    learner.addForbiddenArc('Open1','Volume0')
    learner.addForbiddenArc('High1','Open0')
    learner.addForbiddenArc('High1','Close0')
    learner.addForbiddenArc('High1','High0')
    learner.addForbiddenArc('High1','Low0')
    learner.addForbiddenArc('High1','Volume0')
    learner.addForbiddenArc('Low1','Open0')
    learner.addForbiddenArc('Low1','Close0')
    learner.addForbiddenArc('Low1','High0')
    learner.addForbiddenArc('Low1','Low0')
    learner.addForbiddenArc('Low1','Volume0')
    learner.addForbiddenArc('Close1','Open0')
    learner.addForbiddenArc('Close1','Close0')
    learner.addForbiddenArc('Close1','High0')
    learner.addForbiddenArc('Close1','Low0')
    learner.addForbiddenArc('Close1','Volume0')
    learner.addForbiddenArc('Volume1','Open0')
    learner.addForbiddenArc('Volume1','Close0')
    learner.addForbiddenArc('Volume1','High0')
    learner.addForbiddenArc('Volume1','Low0')
    learner.addForbiddenArc('Volume1','Volume0')
    #learner.addMandatoryArc('Close0','Close1')
    learner.useLocalSearchWithTabuList()
    bn = learner.learnBN()
    gnb.showBN(bn)
    learner = gum.BNLearner(filename+"_train.csv", bn)
    learner.setInitialDAG(bn.dag())
    learner.useAprioriSmoothing(1)
    bn = learner.learnParameters()
    #gnb.showInference(bn,evs={})
    #do inference and calculate the accuracy
    ie = gum.LazyPropagation(bn)
    ie.makeInference()
    N=0.0
    acc=0
    with open(filename+'_test.csv','r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for line in list(reader)[1:]:
            c,o,h,l,v,t=[line[0],line[1],line[2],line[3],line[4],line[5]]
            ie.eraseAllEvidence()
            ie.setEvidence({'Close0':c, 'Open0':o,'High0':h, 'Low0': l,'Volume0': v})
            ie.makeInference()
            prob=ie.posterior(Close1).tolist()
            if prob[0] < 0.6:
                N=N+1
                if t == '1':
                    acc=acc+1
    return acc,N

def genttk(filename,k):#generate the test and train set for k-order markov
    df = pd.read_csv(filename+'.csv')
    df.dropna(axis=0, how='any', inplace=True)#drop the line with NAN in case there is missing data in the file
    Date=df['Date']
    index=list(Date).index('2015-11-13')#find the index of 2015-11-13, we need to slice the list later
        #generate returns and then discretized variables. 
    Open,ob=genbin(genre(df['Open']))
    High,hb=genbin(genre(df['High']))
    Low,lb=genbin(genre(df['Low']))
    Close,cb=genbin(genre(df['Close']))
    Volume,vb=genbinv(genre(df['Volume']))
    train_Open=Open[:index]
    test_Open=Open[index-1:]
    train_High=High[:index]
    test_High=High[index-1:]
    train_Low=Low[:index]
    test_Low=Low[index-1:]
    train_Close=Close[:index]
    test_Close=Close[index-1:]
    train_Volume=Volume[:index]
    test_Volume=Volume[index-1:]
    train=pd.DataFrame()#The train data
    test=pd.DataFrame()#The test data
    for i in range(k+1):#from 0 to k
        if i!=k:
            train['Close'+str(i)]=train_Close[i:-k+i]#at i
            train['Open'+str(i)]=train_Open[i:-k+i]#at i
            train['High'+str(i)]=train_High[i:-k+i]#at i
            train['Low'+str(i)]=train_Low[i:-k+i]#at i
            train['Volume'+str(i)]=train_Volume[i:-k+i]#at i
        if i==k:
            train['Close'+str(i)]=train_Close[i:]#at k
            train['Open'+str(i)]=train_Open[i:]#at k
            train['High'+str(i)]=train_High[i:]#at k
            train['Low'+str(i)]=train_Low[i:]#at k
            train['Volume'+str(i)]=train_Volume[i:]#at k
        if i!=k:
            test['Close'+str(i)]=test_Close[i:-k+i]#at i
            test['Open'+str(i)]=test_Open[i:-k+i]#at i
            test['High'+str(i)]=test_High[i:-k+i]#at i
            test['Low'+str(i)]=test_Low[i:-k+i]#at i
            test['Volume'+str(i)]=test_Volume[i:-k+i]#at i
        #####Generate the boolean var for accuracy calculation
    true=[]
    h=list(df['High'])[index+k-1:]
    c=list(df['Close'])[index+k-1:]
    for i in range(len(h)-1):
        if c[i]<h[i+1]:
            true.append(1)
        else:
            true.append(0)
    test['true']=true
    train.set_index('Close0', inplace=True)
    train.to_csv(filename+'_train'+str(k)+'.csv')
    test.set_index('Close0', inplace=True)
    test.to_csv(filename+'_test'+str(k)+'.csv')
    return ob,hb,lb,cb,vb

def kmodel(filename,k):#generate a k-order markov chain and calculate its accuracy
    ob,hb,lb,cb,vb=genttk(filename,k)
    bn = gum.BayesNet(filename)
    Open=[bn.add('Open'+str(i),ob) for i in range(k+1)]
    High=[bn.add('High'+str(i),hb) for i in range(k+1)]
    Low=[bn.add('Low'+str(i),lb) for i in range(k+1)]
    Close=[bn.add('Close'+str(i),cb) for i in range(k+1)]
    Volume=[bn.add('Volume'+str(i),vb) for i in range(k+1)]
    learner = gum.BNLearner(filename+'_train'+str(k)+'.csv', bn)
    for i in range(1,k+1):#i=future
        for j in range(i):#j=past
            learner.addForbiddenArc('Open'+str(i),'Open'+str(j))
            learner.addForbiddenArc('Open'+str(i),'Close'+str(j))
            learner.addForbiddenArc('Open'+str(i),'High'+str(j))
            learner.addForbiddenArc('Open'+str(i),'Low'+str(j))
            learner.addForbiddenArc('Open'+str(i),'Volume'+str(j))
            learner.addForbiddenArc('High'+str(i),'Open'+str(j))
            learner.addForbiddenArc('High'+str(i),'Close'+str(j))
            learner.addForbiddenArc('High'+str(i),'High'+str(j))
            learner.addForbiddenArc('High'+str(i),'Low'+str(j))
            learner.addForbiddenArc('High'+str(i),'Volume'+str(j))
            learner.addForbiddenArc('Low'+str(i),'Open'+str(j))
            learner.addForbiddenArc('Low'+str(i),'Close'+str(j))
            learner.addForbiddenArc('Low'+str(i),'High'+str(j))
            learner.addForbiddenArc('Low'+str(i),'Low'+str(j))
            learner.addForbiddenArc('Low'+str(i),'Volume'+str(j))
            learner.addForbiddenArc('Close'+str(i),'Open'+str(j))
            learner.addForbiddenArc('Close'+str(i),'Close'+str(j))
            learner.addForbiddenArc('Close'+str(i),'High'+str(j))
            learner.addForbiddenArc('Close'+str(i),'Low'+str(j))
            learner.addForbiddenArc('Close'+str(i),'Volume'+str(j))
            learner.addForbiddenArc('Volume'+str(i),'Open'+str(j))
            learner.addForbiddenArc('Volume'+str(i),'Close'+str(j))
            learner.addForbiddenArc('Volume'+str(i),'High'+str(j))
            learner.addForbiddenArc('Volume'+str(i),'Low'+str(j))
            learner.addForbiddenArc('Volume'+str(i),'Volume'+str(j))
    learner.useLocalSearchWithTabuList()
    bn = learner.learnBN()
    #gnb.showBN(bn)
    learner = gum.BNLearner(filename+'_train'+str(k)+'.csv', bn)
    learner.setInitialDAG(bn.dag())
    learner.useAprioriSmoothing(1)
    bn = learner.learnParameters()
    ie = gum.LazyPropagation(bn)
    ie.makeInference()
    N=0.0
    acc=0
    with open(filename+'_test'+str(k)+'.csv','r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for line in list(reader)[1:]:
            t=line[-1]
            ie.eraseAllEvidence()
            for i in range(k):
                ie.setEvidence({'Close'+str(i):line[5*i], 'Open'+str(i):line[5*i+1],'High'+str(i):line[5*i+2], 'Low'+str(i): line[5*i+3],'Volume'+str(i): line[5*i+4]})
            ie.makeInference()
            prob=ie.posterior(Close[-1]).tolist()
            if prob[0] < 0.498:
                N=N+1
                if t == '1':
                    acc=acc+1
    #print(acc,N)
    return acc,N

def evaluate_k(k):
    filelist=['IDU','IHF','IYC','IYE','IYF','IYG','IYH','IYJ','IYK','IYM','IYR','IYT','IYW','IYZ']
    acc_c=[]
    N_c=[]
    for file in filelist:
        #rint(file+":")
        a,n=kmodel(file,k)
        acc_c.append(a)
        N_c.append(n)
    #print(accuracy)
    ev=sum(acc_c)/sum(N_c)
    '''
    result=pd.DataFrame()
    result['Sector']=filelist
    result['accuracy']=acc_c
    result['N']=N_c
    result.to_csv('result.csv')
    #rint(result,ev)
    '''
    return ev

def main():
    ev=[]
    for k in range(1,20):
        ev.append(evaluate_k(k))
    plt.scatter(list(range(1,20)),ev)

if __name__=='__main__':
    main()
