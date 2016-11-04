# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 22:59:49 2016

@author: Administrator
"""
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from Adaboost import Adaboost
import numpy as np
import random
from Paint_File import Paint_File 
class BYClassifying(object):
    
    def __init__(self, num):
        self.n_estimators = num
    
    def loadDataSet(self,fileName):
       dataMat = [] 
       with open(fileName) as input_data:
           for line in input_data:
               curLine = [float(i) for i in line[1:len(line)-2].split(',')]
               dataMat.append(curLine)
       return dataMat
    
    def randomSample(self, samplenum, *dataSets):
       """
         get the sample data from dataSets
       """
       newDataSet = []
       for dataset in dataSets:
           newDataSet.extend(random.sample(dataset,samplenum))
       newDataSet = np.array(newDataSet)
       newD = np.squeeze(newDataSet)
       return np.array(newD)
     
    def splitData(self, dataSet, labels):
       data_train, data_test, label_train, label_test = train_test_split(dataSet, labels, \
                                                            test_size=0.33, random_state=42)
       return data_train, data_test, label_train, label_test
    
    def mergeData(self, data1, data2):
        newdata = np.vstack((data1, data2))
        return newdata
    
    def mergeLabel(self, label1, label2):
        newdata = np.hstack((label1, label2))
        return newdata
        
    def randomForestsClassifying(self, datMat, classlabels):
        rfc = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=None,
                                     min_samples_split=1, random_state=0)
        rfc.fit(datMat, classlabels)
        return rfc
    
    def extraForestsClassifying(self, datMat, classlabels):
        efc = ExtraTreesClassifier(n_estimators=self.n_estimators, max_depth=None, 
                                   min_samples_split=1, random_state=0)
        efc.fit(datMat, classlabels)
        return efc 
    
    def adaboostClassifying(self, datMat, classlabels):
        adaboost = Adaboost()
        ada, aggClassEst = adaboost.adaBoostTrainDS(datMat,classlabels, self.n_estimators)
        return ada
    
    def paint(self, datMat):
        sumData = np.sum(datMat, axis = 0)
        xais = [i for i in range(len(sumData))]
        pt = Paint_File()
        pt.paint(xais,sumData)
        pass

if __name__=='__main__':
    byclassifying = BYClassifying(100)
    
    filename = "SubSpectrumData/"+"SimulateData_intensity_iongroup"
    filename2 = "SubSpectrumData/"+"SimulateData_dual_intensity_iongroup"
    
    dataset = byclassifying.loadDataSet(filename)
    dataset2 = byclassifying.loadDataSet(filename2)
    
    alldata = byclassifying.mergeData(dataset,dataset2)
    
    labels = [] 
    for i in range(1):
        labels += [1] * 12000                          
    for i in range(1):
        labels += [-1] * 12000
    labels = np.array(labels)
    
    data_train, data_test, label_train, label_test = byclassifying.splitData(alldata,labels)
    
#    efclassifier = byclassifying.extraForestsClassifying(data_train,label_train)
#    efscore = efclassifier.score(data_test, label_test)

#################################################test 1 paint data################################
    byclassifying.paint(dataset)
    byclassifying.paint(dataset2)