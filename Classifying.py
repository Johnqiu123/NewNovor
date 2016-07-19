# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:32:07 2016

@author: Johnqiu
"""
import time
import numpy as np
import matplotlib.pyplot as plt 
import random
import sklearn.neighbors as skneighbour
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import cPickle as cpickle

class Classifying(object):

   def loadDataSet(self,fileName):      #general function to parse tab -delimited floats
       dataMat = []                #assume last column is target value
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
   
   def shuffleData(self, dataset, labels):
       """
       np.random.shuffle(x)
       np.random.permutation(x)
       """
       dim = np.shape(dataset)[1]
       labels = np.array([labels]).T
       newDataSet = np.hstack((dataset, labels))
       newDS = np.random.permutation(newDataSet)
       
       newLabel = newDS[:,-1]
       dataSet = newDS[:,:dim]
       return dataSet,newLabel
    
   def KnnClassifying(self, k, weights, algorithm):
       """
       Args:
       k:        Number of neighbors to use 
       weights:   weight function used in prediction.  value conclude:{'uniform','distance'}
       algorithm: Algorithm used to compute the nearest neighbors.
                  values conclude:{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
       
       Return：
       clf
       """
       clf = skneighbour.KNeighborsClassifier(n_neighbors=k, weights=weights,algorithm=algorithm)
       return clf
     
   def classifyingTest(self, estimator, dataset, labels):
        print "classifyingTest"
        hoRatio = 0.20      #hold out 10%
        m = dataset.shape[0]
        print m
        redudata = PCA(n_components=4).fit_transform(dataset)
        numTestVecs = int(m*hoRatio)
        errorCount = 0.0
        estimator.fit(redudata[numTestVecs:m,:],labels[numTestVecs:m])
        classifierResult = estimator.predict(redudata[0:numTestVecs,:])
        lab = labels[0:numTestVecs]
        errorCount = len(np.nonzero(lab!=classifierResult)[0])    
        correct = classifierResult[np.nonzero(lab==classifierResult)[0]]
        count0 = len(np.nonzero(correct==0)[0])
    
        print "the total error rate is: %f" % (errorCount/float(numTestVecs))
        print "the total correct rate is: %f" % (1-errorCount/float(numTestVecs))
        print errorCount    
        print count0
        print float(count0) / len(correct)
        return estimator
    
   def paintDataset(self, dataset, num, labelnum):
       # 数据集合并降维图      
       redudata = PCA(n_components=2).fit_transform(np.mat(dataset))
#       print np.shape(redudata)
       
       plt.figure()
       plt.clf()  # clear figure       
       
       colorset = np.linspace(0, 1, labelnum)
       colorsets = plt.cm.Spectral(colorset)
       
       begin = 0
       for i in range(1,labelnum):     
           end = begin + num
#           print end
           redud = redudata[begin:end]
#           print np.shape(redud)
           plt.plot(redud[:, 0], redud[:, 1], 'k.', color = colorsets[i], markersize=2)
           begin = end
          
       plt.show()
   
   def paintResult(self, dataset, labels, labelnum):
       # 数据结果图     
       redudata = PCA(n_components=2).fit_transform(np.mat(dataset))
#       print np.shape(redudata)
       
       plt.figure()
       plt.clf()  # clear figure       
       
       colorset = np.linspace(0, 1, labelnum)
       colorsets = plt.cm.Spectral(colorset)

       for i in range(labelnum):
           redud = redudata[np.nonzero(labels==i)[0]]
           print np.shape(redud)
           plt.plot(redud[:, 0], redud[:, 1], 'k.', color = colorsets[i], markersize=2)
       plt.show() 
    
   def paint3DDataset(self, dataset, num, labelnum):
        # 数据集合并降维图      
       redudata = PCA(n_components=3).fit_transform(np.mat(dataset))
#       print np.shape(redudata)
       
       fig = plt.figure()
       plt.clf()  # clear figure
       ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)       
       
       colorset = np.linspace(0, 1, labelnum)
       colorsets = plt.cm.Spectral(colorset)
       
       begin = 0
       for i in range(1,labelnum):     
           end = begin + num
           print end
           redud = redudata[begin:end]
#           print np.shape(redud)
           ax.scatter(redud[:, 0], redud[:, 1], redud[:, 2], c=colorsets[i])
           begin = end
          
       ax.w_xaxis.set_ticklabels([])
       ax.w_yaxis.set_ticklabels([])
       ax.w_zaxis.set_ticklabels([])
       ax.set_xlabel('x')
       ax.set_ylabel('y')
       ax.set_zlabel('z')    
       plt.show()

   def cpStoreClassifier(self,filename, classifier):
       """
          store a classifier by cpickle
          args:
             filename
             datas
       """
       with open(filename,'w') as fw:
           cpickle.dump(classifier, fw)
    
   def cpLoadClassifier(self, filename):
       """
         generate a Classifier by cpickle
         args:
            filename
              
         return:
            datas
       """
       with open(filename,'r') as fr:
           classifier = cpickle.load(fr)
           return classifier  

if __name__=='__main__':
    classifying = Classifying()
    
    filename = "SubSpectrumData/"+"IonGroups_Int"
    filename2 = "SubSpectrumData/"+"IonGroups_DualInt"
    filename3 = "SubSpectrumData/"+"IonGroups_NoiInt"
    filename4 = "SubSpectrumData/"+"IonGroups_AtypeInt"
    filename5 = "SubSpectrumData/"+"IonGroups_yNH3Int"
    filename6 = "SubSpectrumData/"+"IonGroups_yH2OInt"
    filename7 = "SubSpectrumData/"+"IonGroups_bH2OInt"
    filename8 = "SubSpectrumData/"+"IonGroups_bNH3Int"
    filename9 = "SubSpectrumData/"+"IonGroups_y46-Int"
    filename10 = "SubSpectrumData/"+"IonGroups_y45-Int"
    filename11 = "SubSpectrumData/"+"IonGroups_y10+Int"
    
#################################################test 1#####################################
    dataset = classifying.loadDataSet(filename)
    dataset2 = classifying.loadDataSet(filename2)
    dataset3 = classifying.loadDataSet(filename3)
    dataset4 = classifying.loadDataSet(filename4)
    dataset5 = classifying.loadDataSet(filename5)
    dataset6 = classifying.loadDataSet(filename6)
    dataset7 = classifying.loadDataSet(filename7)
    dataset8 = classifying.loadDataSet(filename8)
    dataset9 = classifying.loadDataSet(filename9)
    dataset10 = classifying.loadDataSet(filename10)
    dataset11 = classifying.loadDataSet(filename11)
    
    newData = classifying.randomSample(20000, dataset, dataset2, dataset3, dataset4,\
                              dataset5, dataset6, dataset7, dataset8,\
                              dataset9, dataset10, dataset11)
#    classifying.paintDataset(newData,20000,11)  
#    classifying.paint3DDataset(newData,10000,11)       
    labels = [] 
    for i in range(2):
        labels += [0] * 20000                          
    for i in range(9):
        labels += [1] * 20000 
    
#    for i in range(11):
#        labels += [i] * 20000

    newD,newLabel = classifying.shuffleData(newData,labels)
#    classifying.paintDataset(newD,20000,11) 
    estimator = classifying.KnnClassifying(30,'distance','ball_tree')
    newstimator = classifying.classifyingTest(estimator, newD,newLabel)
    classifying.cpStoreClassifier("KnnClassifier",newstimator)
#    classifying.paintResult(newD,labelreuslt,2) 
    
#    hoRatio = 0.20      #hold out 10%
#    m = newData.shape[0]
#    numTestVecs = int(m*hoRatio)
#    errorCount = 0.0
#    estimator.fit(newData[numTestVecs:m,:],labels[numTestVecs:m])
#    classifierResult = estimator.predict(newData[0:numTestVecs,:])
#    lab = labels[0:numTestVecs]
#    count = np.nonzero(lab!=classifierResult)[0]
#    for i in range(numTestVecs):
#        estimator.fit(dataset[numTestVecs:m,:],labels[numTestVecs:m])
#        classifierResult = estimator.predict(dataset[i,:])
##        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
#        if (classifierResult != labels[i]): errorCount += 1.0
#    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
#    print errorCount        
    
#################################################test 2 twolabel#####################################
    
    
    