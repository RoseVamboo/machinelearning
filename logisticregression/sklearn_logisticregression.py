# -*-coding:utf-8 -*-

from __future__ import division
import numpy as np
import random
from sklearn.linear_model import LogisticRegression as LR
#加载训练数据
def loadTrainData_random():
    trainData=[]
    labelData=[]

    #标签为0的训练数据
    train0=np.load("train0.npy")
    #标签为1的训练数据
    train1=np.load("train1.npy")
    #构建标签数据
    label0=[0 for i in range(len(train0))]
    label1=[1 for i in range(len(train1))]

    #将训练数据train0和train1随机合并成一个数组
    lendata=len(train0)+len(train1)

    dataList=random.sample(range(0,lendata),len(train0))
    dataList.sort()
    index=0
    index0=0
    index1=0
    for i in range(0,len(train0)+len(train1)):
        if index<len(train0) and i==dataList[index]:
            index=index+1
            trainData.append(train0[index0])
            labelData.append(0)
            index0=index0+1
        else:
            trainData.append(train1[index1])
            labelData.append(1)
            index1=index1+1
    print(len(trainData))
    print(len(labelData))

    return trainData,labelData

def loadTrainData():
    trainData=[]
    labelData=[]

    #加载标签为1和0的数据
    train0=np.load("train0.npy")
    train1=np.load("train1.npy")

    #将train0和train1合并为一个数组
    for i in range(len(train0)):
        trainData.append(train0[i])
        labelData.append(0)
    for i in range(len(train1)):
        trainData.append(train1[i])
        labelData.append(1)

    print(len(trainData))
    print(len(labelData))
    return trainData,labelData

def loadTestData():
    testData=[]
    testlabel=[]

    #加载测试数据
    test0a=np.load("test0a.npy")
    test1a=np.load("test1a.npy")

    for i in range(len(test0a)):
        testData.append(test0a[i])
        testlabel.append(0)
    for i in range(len(test1a)):
        testData.append(test1a[i])
        testlabel.append(1)
    print(len(testData))
    print(len(testlabel))
    return testData,testlabel

def loadTestData1():

    # 加载测试数据
    test0a = np.load("test0a.npy")
    test1a = np.load("test1a.npy")

    label0a=[0 for i in range(len(test0a))]
    label1a=[1 for i in range(len(test1a))]

    return test0a,label0a,test1a,label1a

def train():
    x2,y2=loadTrainData_random()
    x3,y3=loadTestData()

    # 创建逻辑回归对象(3种情况：1.自设参数；2.balanced； 3.默认参数
    ##########################################################

    # 1 .自己设置模型参数
    # penalty = {0: 0.2, 1: 0.8}
    # lr = LR(class_weight = penalty)#设置模型分类的权重为penalty

    # 2. 选择样本平衡-balanced
    # lr = LR(class_weight='balanced')#样本平衡

    # 3. 默认参数，class_weight=none
    # lr = LR()
    lr = LR()
    ##############################################################

    #调用fit函数训练模型参数
    lr.fit(x2,y2)
    print("训练结束")

    #测试
    y3_pred=lr.predict(x3)
    print(u'模型准确率(测试集)为:%s'%lr.score(x3,y3))
    print(u'模型准确率(测试集,y=0)为:%s'%(sum(y3_pred[i]==0 for i,v in enumerate(y3) if v==0)/sum(1 for i,v in enumerate(y3) if v==0)))
    print(u'模型准确率(测试集,y=1)为:%s'%(sum(y3_pred[i]==1 for i,v in enumerate(y3) if v==1)/sum(1 for i,v in enumerate(y3) if v==1)))

    print("y3_pred=",y3_pred)
    print("y3=",y3)

if __name__=='__main__':
    train()