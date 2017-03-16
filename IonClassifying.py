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
from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt 
from sklearn.cross_validation import StratifiedKFold  
from scipy import interp  
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
    
#    filename = "SubSpectrumData/"+"new#CHPP#LM3#RP3#2_intensity_iongroup"
#    filename2 = "SubSpectrumData/"+"new#CHPP#LM3#RP3#2_dual_intensity_iongroup"
#    filename3 = "SubSpectrumData/"+"new#CHPP#LM3#RP3#2_atype_intensity_iongroup"
#    filename4 = "SubSpectrumData/"+"new#CHPP#LM3#RP3#2_bH2O_intensity_iongroup"
#    filename5 = "SubSpectrumData/"+"new#CHPP#LM3#RP3#2_bNH3_intensity_iongroup"
#    filename6 = "SubSpectrumData/"+"new#CHPP#LM3#RP3#2_Noise_intensity_iongroup"
#    filename7 = "SubSpectrumData/"+"new#CHPP#LM3#RP3#2_y10+_intensity_iongroup"
#    filename8 = "SubSpectrumData/"+"new#CHPP#LM3#RP3#2_y45-_intensity_iongroup"
#    filename9 = "SubSpectrumData/"+"new#CHPP#LM3#RP3#2_y46-_intensity_iongroup"
#    filename10 = "SubSpectrumData/"+"new#CHPP#LM3#RP3#2_yH2O_intensity_iongroup"
#    filename11 = "SubSpectrumData/"+"new#CHPP#LM3#RP3#2_yNH3_intensity_iongroup"
     
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
    
#    data_train, data_test, label_train, label_test = ionclassifying.splitData(otherData,otherlabels)
#    
#    # create new dataset
#    data_train = ionclassifying.mergeData(data_train, byData)    
#    data_test = ionclassifying.mergeData(data_test, byData)
#    label_train = ionclassifying.mergeLabel(label_train, labels)
#    label_test = ionclassifying.mergeLabel(label_test, labels)
    
 #################################################test 1#####################################   
#    efclassifier = ionclassifying.extraForestsClassifying(data_train,label_train)
#    probas_ = efclassifier.predict_proba(data_test)
#    fpr, tpr, thresholds = roc_curve(label_test, probas_[:, 1]) 
#    roc_auc = auc(fpr, tpr)
#    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
#    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')    
#    plt.xlim([-0.05, 1.05])  
#    plt.ylim([-0.05, 1.05])  
#    plt.xlabel('False Positive Rate')  
#    plt.ylabel('True Positive Rate')  
#    plt.title('Receiver operating characteristic example')  
#    plt.legend(loc="lower right")  
#    plt.show() 
#    efscore = efclassifier.score(data_test, label_test)

#################################################test 1#####################################
#    adaclassifier = ionclassifying.adaboostClassifying(data_train,label_train)
#    adaboost = Adaboost()
#    prediction = adaboost.adaClassify(data_test, adaclassifier)
#    m, n = np.shape(data_test)
#    errArr = np.mat(np.ones((m,1)))
#    errsum = errArr[prediction != np.mat(label_test).T].sum() / m

#################################################test 1#####################################
#    cv = StratifiedKFold(otherlabels, n_folds=5)
#    efclassifier = RandomForestClassifier(n_estimators=50, max_depth=None,
#                                     min_samples_split=1, random_state=0)
#    mean_tpr = 0.0  
#    mean_fpr = np.linspace(0, 1, 100)  
#    all_tpr = []     
#    for i, (train, test) in enumerate(cv):  
#        # 合成数据集
#        data_train = ionclassifying.mergeData(otherData[train], byData)    
#        data_test = ionclassifying.mergeData(otherData[test], byData)
#        label_train = ionclassifying.mergeLabel(otherlabels[train], labels)
#        label_test = ionclassifying.mergeLabel(otherlabels[test], labels)
#        
#        # 训练
#        efmodel = efclassifier.fit(data_train, label_train)
#        print efmodel.score(data_train, label_train)
#        # 预测
#        probas_ = efmodel.predict_proba(data_test)
#        #通过roc_curve()函数，求出fpr和tpr，以及阈值  
#        fpr, tpr, thresholds = roc_curve(label_test, probas_[:, 1])  
#        probas_
#        mean_tpr += interp(mean_fpr, fpr, tpr)          #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数  
#        mean_tpr[0] = 0.0                               #初始处为0  
#        roc_auc = auc(fpr, tpr)  
        #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来  
#        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))  
    #画对角线  
#    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
      
#    mean_tpr /= len(cv)                     #在mean_fpr100个点，每个点处插值插值多次取平均  
#    mean_tpr[-1] = 1.0                      #坐标最后一个点为（1,1）  
#    mean_auc = auc(mean_fpr, mean_tpr)      #计算平均AUC值  
    #画平均ROC曲线  
    #print mean_fpr,len(mean_fpr)  
    #print mean_tpr  
#    plt.plot(mean_fpr, mean_tpr, 'k--',  
#             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)  
#      
#    plt.xlim([-0.05, 1.05])  
#    plt.ylim([-0.05, 1.05])  
#    plt.xlabel('False Positive Rate')  
#    plt.ylabel('True Positive Rate')  
#    plt.title('Adaboost ROC')   
#    plt.legend(loc="lower right")  
#    plt.show() 

#################################################test 1#####################################
    est = [1,50]
    for i in est:
        cv = StratifiedKFold(otherlabels, n_folds=10)
        efclassifier = RandomForestClassifier(n_estimators=i, max_depth=None,
                                         min_samples_split=1, random_state=0)
        mean_tpr = 0.0  
        mean_fpr = np.linspace(0, 1, 100)  
        all_tpr = []     
        trainScore = []
        testScore = []
        for i, (train, test) in enumerate(cv):  
            # 合成数据集
            data_train = ionclassifying.mergeData(otherData[train], byData)    
            data_test = ionclassifying.mergeData(otherData[test], byData)
            label_train = ionclassifying.mergeLabel(otherlabels[train], labels)
            label_test = ionclassifying.mergeLabel(otherlabels[test], labels)
            
            # 训练
            efmodel = efclassifier.fit(data_train, label_train)
            trainS = efmodel.score(data_train, label_train)
            testS = efmodel.score(data_test, label_test)
            trainScore.append(trainS)
            testScore.append(testS)
        
        meanTrain = np.mean(trainScore)
        meanTest = np.mean(testScore)
        print 1-meanTrain
        print 1-meanTest