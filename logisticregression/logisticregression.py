import numpy as np
import random
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

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(trainData,labelData,alpha,maxIndex):
    #将数组转换为矩阵 size:m*n
    trainDataMat=np.mat(trainData)

    #size: m*1
    labelDataMat=np.mat(labelData)
    labelDataMat=labelDataMat.transpose()
    m,n=np.shape(trainDataMat)

    #初始化权重
    '''
    1.可以初始化成大小为n*1,元素值为0的矩阵
    weigh=np.zeros((n,1))
    2.也可以初始化成大小为n*1,元素值为1的矩阵
    weigh=np.ones((n,1))
    3.也可以初始化成大小为n*1,元素值为符合标准正态分布的随机数的矩阵
    weigh=np.random.randn(n)
    '''
    weigh=np.random.randn(n)
    weigh=np.mat(weigh)
    weigh=weigh.transpose()
    for i in range(maxIndex):
        h=sigmoid(trainDataMat*weigh)
        dh=labelDataMat-h  #size:m*1
        weigh=weigh+alpha*trainDataMat.transpose()*dh

    return weigh

def classify(testData,testLabelData,weigh):
    #将测试数据转换成矩阵的形式
    testDataMat=np.mat(testData)
    testLabelDataMat=np.mat(testLabelData)
    testLabelDataMat=testLabelDataMat.transpose()
    h=sigmoid(testDataMat*weigh)
    error=0
    pred=[]
    m=len(h)
    for i in range(m):
        if h[i]>0.5:
            pred.append(1)
            if int(testLabelDataMat[i])!=1:
                error+=1
                print("error")
        else:
            pred.append(0)
            if int(testLabelDataMat[i])!=0:
                error+=1
                print("error")
    print(u"错误率为：%.4f"%(error/m))
    return pred

def operation(alpha=0.001,maxIndex=2000):
    data,label=loadTrainData_random()
    weigh=gradAscent(data,label,alpha,maxIndex)
    testData,testLabel=loadTestData()
    pred=classify(testData,testLabel,weigh)
    print("testLabel=",testLabel)
    print("pred_ypred=",pred)

if __name__=="__main__":
    operation()