# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 22:56:45 2016

@author: Administrator
"""
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from Adaboost import Adaboost
import numpy as np
import random
class IonClassifying(object):
    
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
        
    
    
    
if __name__=='__main__':
    ionclassifying = IonClassifying(100)
    
    filename = "SubSpectrumData/"+"SimulateData_intensity_iongroup"
    filename2 = "SubSpectrumData/"+"SimulateData_dual_intensity_iongroup"
    filename3 = "SubSpectrumData/"+"SimulateData_Noise_intensity_iongroup"
    filename4 = "SubSpectrumData/"+"SimulateData_bH2O_intensity_iongroup"
    filename5 = "SubSpectrumData/"+"SimulateData_bNH3_intensity_iongroup"
    filename6 = "SubSpectrumData/"+"SimulateData_atype_intensity_iongroup"
    filename7 = "SubSpectrumData/"+"SimulateData_y10+_intensity_iongroup"
    filename8 = "SubSpectrumData/"+"SimulateData_y45-_intensity_iongroup"
    filename9 = "SubSpectrumData/"+"SimulateData_y46-_intensity_iongroup"
    filename10 = "SubSpectrumData/"+"SimulateData_yH2O_intensity_iongroup"
    filename11 = "SubSpectrumData/"+"SimulateData_yNH3_intensity_iongroup"
     
    dataset = ionclassifying.loadDataSet(filename)
    dataset2 = ionclassifying.loadDataSet(filename2)
    dataset3 = ionclassifying.loadDataSet(filename3)
    dataset4 = ionclassifying.loadDataSet(filename4)
    dataset5 = ionclassifying.loadDataSet(filename5)
    dataset6 = ionclassifying.loadDataSet(filename6)
    dataset7 = ionclassifying.loadDataSet(filename7)
    dataset8 = ionclassifying.loadDataSet(filename8)
    dataset9 = ionclassifying.loadDataSet(filename9)
    dataset10 = ionclassifying.loadDataSet(filename10)
    dataset11 = ionclassifying.loadDataSet(filename11)
    
    byData = ionclassifying.randomSample(10000, dataset, dataset2)
    otherData = ionclassifying.randomSample(10000, dataset3, dataset4,\
                              dataset5, dataset6, dataset7, dataset8,\
                              dataset9, dataset10, dataset11)
    labels = [] 
    otherlabels = []
    for i in range(2):
        labels += [1] * 10000                          
    for i in range(9):
        otherlabels += [-1] * 10000
    labels = np.array(labels)
    otherlabels = np.array(otherlabels)
    
    data_train, data_test, label_train, label_test = ionclassifying.splitData(otherData,otherlabels)
    
    # create new dataset
    data_train = ionclassifying.mergeData(data_train, byData)    
    data_test = ionclassifying.mergeData(data_test, byData)
    label_train = ionclassifying.mergeLabel(label_train, labels)
    label_test = ionclassifying.mergeLabel(label_test, labels)
    
 #################################################test 1#####################################   
    efclassifier = ionclassifying.extraForestsClassifying(data_train,label_train)
    efscore = efclassifier.score(data_test, label_test)

#################################################test 1#####################################
#    adaclassifier = ionclassifying.adaboostClassifying(data_train,label_train)
#    adaboost = Adaboost()
#    prediction = adaboost.adaClassify(data_test, adaclassifier)
#    m, n = np.shape(data_test)
#    errArr = np.mat(np.ones((m,1)))
#    errsum = errArr[prediction != np.mat(label_test).T].sum() / m
    