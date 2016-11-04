# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 10:39:23 2016

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt 

class Adaboost(object):
    
    def loadSimpData(self):
        datMat = np.matrix([[ 1. ,  2.1],
            [ 2. ,  1.1],
            [ 1.3,  1. ],
            [ 1. ,  1. ],
            [ 2. ,  1. ]])
        classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
        return datMat,classLabels
    
    def loadDataSet(self,fileName):      #general function to parse tab -delimited floats
        numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
        dataMat = []; labelMat = []
        fr = open(fileName)
        for line in fr.readlines():
            lineArr =[]
            curLine = line.strip().split('\t')
            for i in range(numFeat-1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        return dataMat,labelMat
    
    
    def stumpClassify(self,dataMatrix,dimen,threshVal,threshIneq):#just classify the data
        retArray = np.ones((np.shape(dataMatrix)[0],1))
        if threshIneq == 'lt':
            retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
        else:
            retArray[dataMatrix[:,dimen] > threshVal] = -1.0
        return retArray
  
    def buildStump(self, dataArr,classLabels,D):
        '''
        single desicion tree
        
        Args:
        dataArr:           data matrix
        classLabels:       data labels
        D:                 data weights
        
        Return:
        bestStump,minError,bestClasEst
        
        set different thresholds to build a tree(n*numsteps*2)
        '''
        dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
        m,n = np.shape(dataMatrix)
        numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
        minError = np.inf #init error sum, to +infinity
        for i in range(n):#loop over all dimensions(all features)
            rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
            stepSize = (rangeMax-rangeMin)/numSteps # set (numSteps) stepSizes
            for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
                for inequal in ['lt', 'gt']: #go over less than and greater than
                    threshVal = (rangeMin + float(j) * stepSize)
                    predictedVals = self.stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                    errArr = np.mat(np.ones((m,1)))
                    errArr[predictedVals == labelMat] = 0
                    # key step
                    weightedError = D.T*errArr  #calc total error multiplied by D
#                    print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                    if weightedError < minError:
                        minError = weightedError
                        bestClasEst = predictedVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        return bestStump,minError,bestClasEst

    def adaBoostTrainDS(self,dataArr,classLabels,numIt=40):
        """
        Train adaBoost algorithem
        
        Args:
        dataArr:           data matrix
        classLabels:       data class
        numIt:      the numbers of classifiers
        """
        weakClassArr = []
        m = np.shape(dataArr)[0]
        D = np.mat(np.ones((m,1))/m)   #init D to all equal
        aggClassEst = np.mat(np.zeros((m,1)))
        for i in range(numIt):
            bestStump,error,classEst = self.buildStump(dataArr,classLabels,D)#build Stump
            if error > 0.5: break
            #print "D:",D.T
            alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
            bestStump['alpha'] = alpha  
            weakClassArr.append(bestStump)                  #store Stump Params in Array
            #print "classEst: ",classEst.T
            expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst) #exponent for D calc, getting messy
            D = np.multiply(D,np.exp(expon))                              #Calc New D for next iteration
            D = D/D.sum()
            #calc training error of all classifiers, if this is 0 quit for loop early (use break)
            aggClassEst += alpha*classEst
            #print "aggClassEst: ",aggClassEst.T
            aggErrors = np.multiply(np.sign(aggClassEst) !=np. mat(classLabels).T,np.ones((m,1)))
            errorRate = aggErrors.sum()/m
#            print "total error: ",errorRate
            if errorRate == 0.0: break
        return weakClassArr,aggClassEst
    
    def adaClassify(self,datToClass,classifierArr):
        dataMatrix = np.mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
        m = np.shape(dataMatrix)[0]
        aggClassEst = np.mat(np.zeros((m,1)))
        for i in range(len(classifierArr)):
            classEst = self.stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                     classifierArr[i]['thresh'],\
                                     classifierArr[i]['ineq'])#call stump classify
            aggClassEst += classifierArr[i]['alpha']*classEst  # class probability
        print aggClassEst
        return np.sign(aggClassEst)
    
    def plotROC(self, predStrengths, classLabels):
        cur = (1.0,1.0) #cursor
        ySum = 0.0 #variable to calculate AUC
        numPosClas = sum(np.array(classLabels)==1.0)
        yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
        sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
        fig = plt.figure()
        fig.clf()
        ax = plt.subplot(111)
        #loop through all the values, drawing a line segment at each point
        for index in sortedIndicies.tolist()[0]:
            if classLabels[index] == 1.0:
                delX = 0; delY = yStep;
            else:
                delX = xStep; delY = 0;
                ySum += cur[1]
            #draw line from cur to (cur[0]-delX,cur[1]-delY)
            ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
            cur = (cur[0]-delX,cur[1]-delY)
        ax.plot([0,1],[0,1],'b--')
        plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
        plt.title('ROC curve for AdaBoost horse colic detection system')
        ax.axis([0,1,0,1])
        plt.show()
        print "the Area Under the Curve is: ",ySum*xStep

    
    def paint(self, datMat):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(datMat[:,0],datMat[:,1],'ro')
        ax.set_xticks(datMat[:,0])
        plt.show()

if __name__ == "__main__":
    ada = Adaboost()
#    datMat,classLabels = ada.loadSimpData()
#################################################test 1#####################################
#    D = np.mat(np.ones((5,1))/5)
#    ada.buildStump(datMat, classLabels, D)
#    print D
#    ada.paint(datMat)
#################################################test 2#####################################
#    weakClassArr,aggClassEst = ada.adaBoostTrainDS(datMat, classLabels, 40)
#    result = ada.adaClassify([[5,5],[0,0]],weakClassArr)
#    print result 
#################################################test 3#####################################
    datArr, labelArr = ada.loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = ada.adaBoostTrainDS(datArr, labelArr, 40)
    testArr, testLabelArr = ada.loadDataSet('horseColicTest2.txt')
    prediction = ada.adaClassify(testArr, classifierArray)
    errArr = np.mat(np.ones((67,1)))
    errsum = errArr[prediction != np.mat(testLabelArr).T].sum() / 67
    print errsum
    
    ada.plotROC(aggClassEst.T, labelArr)
    
    
    