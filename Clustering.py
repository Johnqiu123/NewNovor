# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 20:43:32 2016

@author: Johnqiu
"""
"""
自带的距离计算：
    ============     ====================================
    metric           Function
    ============     ====================================
    'cityblock'      metrics.pairwise.manhattan_distances
    'cosine'         metrics.pairwise.cosine_distances
    'euclidean'      metrics.pairwise.euclidean_distances
    'l1'             metrics.pairwise.manhattan_distances
    'l2'             metrics.pairwise.euclidean_distances
    'manhattan'      metrics.pairwise.manhattan_distances
    ============     ====================================

'k-means++' : selects initial cluster centers for k-mean
clustering in a smart way to speed up convergence. See section
Notes in k_init for more details.

'random': generate k centroids from a Gaussian with mean and
variance estimated from the data.

"""
import sklearn.cluster as skcluster 
from sklearn.decomposition import PCA
import time
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt 
import random
from mpl_toolkits.mplot3d import Axes3D

class Clustering(object):
    
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
       return np.mat(newD)


   def KMeans(self, init, clusternum, inititer):
       """
          KMeans
       """
       estimator = skcluster.KMeans(init=init, n_clusters=clusternum, n_init=inititer)
       return estimator
   
   def MinibatchKMeans(self, init, clusternum):
       """
          MinibatchKMeans
       """
       estimator = skcluster.MiniBatchKMeans(init=init, n_clusters=clusternum, n_init=10)
       return estimator
    
   def AgglomerativeClustering(self, linkage):
       # If linkage is "ward", only "euclidean" is accepted
       estimator = skcluster.AgglomerativeClustering(linkage = linkage)
       return estimator
      
   def DBscan(self, eps, min_samples):
       estimator = skcluster.DBSCAN(eps=eps, min_samples=min_samples)
       return estimator
   
   def SpectralClustering(self, clusternum):
       estimator = skcluster.SpectralClustering(n_clusters=clusternum,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
       return estimator
   def Birch(self, clusternum):
       """
          Birch
       """
       estimator = skcluster.Birch(n_clusters=clusternum)
       return estimator
       
   def sl_clusting(self, estimator, name,  dataset):
       #  
       t0 = time.clock()
       estimator.fit(dataset)
       score = metrics.silhouette_score(dataset, estimator.labels_,metric='euclidean',
                                      sample_size=300)
       print('% 9s %.2fs %i %.3f'
              % (name, (time.clock() - t0),estimator.inertia_, score))
       return estimator,score
      
   def paint(self, data, noidata=None):
       reduced_data = PCA(n_components=2).fit_transform(data)
       kmeans = self.KMeans("k-means++", 3, 10)
       kmeans.fit(reduced_data)
       
       # Step size of the mesh. Decrease to increase the quality of the VQ.
       h = .02 # point in the mesh [x_min, m_max]x[y_min, y_max].
       
       # Plot the decision boundary. For that, we will assign a color to each
       x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
       y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
       xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) # meshgrid
       
       print xx.shape,yy.shape

       # Obtain labels for each point in mesh. Use last trained model.
       Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
       print Z.shape[0]

       # Put the result into a color plot
       Z = Z.reshape(xx.shape)
       plt.figure(1)
       plt.clf()  # clear figure
       plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')
       plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.',markersize=2)

       # Plot the centroids as a white X
       centroids = kmeans.cluster_centers_
       plt.scatter(centroids[:, 0], centroids[:, 1],
                   marker='x', s=169, linewidths=3,
                   color='r', zorder=10)
                   
       plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                'Centroids are marked with white cross')
       plt.xlim(x_min, x_max)
       plt.ylim(y_min, y_max)
       plt.xticks(())
       plt.yticks(())
       plt.show()
       
       return Z

   def paint2(self, *dataset):
       colors = ['black','b', 'r', 'c', 'm']
       reduced_datas = []
       
       for data in dataset:
           redudata = PCA(n_components=2).fit_transform(data)
           reduced_datas.append(redudata)
       
#       
       kmeans = self.KMeans("k-means++", 2, 10)
       kmeans.fit(reduced_datas[0])
       
#        Step size of the mesh. Decrease to increase the quality of the VQ.
       h = .02 # point in the mesh [x_min, m_max]x[y_min, y_max].
       
#        Plot the decision boundary. For that, we will assign a color to each
       x_min, x_max = reduced_datas[0][:, 0].min() + 1, reduced_datas[0][:, 0].max() - 1
       y_min, y_max = reduced_datas[0][:, 1].min() + 1, reduced_datas[0][:, 1].max() - 1
       
       xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) # meshgrid
       
#       orginlabel = kmeans.labels_
#       noilabel = kmeans.predict(reduced_data2)
#       duallabel = kmeans.predict(reduced_data3)
       
#        Obtain labels for each point in mesh. Use last trained model.
       Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#       print Z.shape[0]

       # Put the result into a color plot
       Z = Z.reshape(xx.shape)
       plt.figure()
       plt.clf()  # clear figure
       plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower') 
       i = 0   
       for redudata in reduced_datas:
           plt.plot(redudata[:, 0], redudata[:, 1], 'k.', color = colors[i], markersize=2)
           i += 1
       
#       for i in range(reduced_data2.shape[0]):
#           k = noilabel[i]
#           plt.plot(reduced_data2[:, 0], reduced_data2[:, 1], 'o',color=plt.rcParams['axes.color_cycle'][k+1],markersize=2)

#        Plot the centroids as a white X
       centroids = kmeans.cluster_centers_
       plt.scatter(centroids[:, 0], centroids[:, 1],
                   marker='x', s=169, linewidths=3,
                   color='w', zorder=10)
#                   
       plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                'Centroids are marked with white cross')
       plt.xlim(x_min, x_max)
       plt.ylim(y_min, y_max)
       plt.grid(True)
       plt.xticks(())
       plt.yticks(())
       plt.show()

   def paint3(self, *dataset):
       colors = ['black','b', 'r', 'c', 'm','black','b', 'r', 'c', 'm','black','b', 'r', 'c', 'm']
       reduced_datas = []
       redudata = []

       i = 0
       for data in dataset:
           plt.figure()  
           redudata = [data[:,1],data[:,0]]
           reduced_datas.append(redudata)
           plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                'Centroids are marked with white cross')
           plt.plot(redudata[0], redudata[1], 'k.', color = colors[i], markersize=2)
           i += 1
           plt.show()

   def paint4(self, dataset):
        X = PCA(n_components=2).fit_transform(dataset)
        db = skcluster.DBSCAN(eps=0.3, min_samples=30).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        
        print('Estimated number of clusters: %d' % n_clusters_)
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, labels))
              
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        plt.figure(111)
        plt.clf()  # clear figure
#        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        colors = ['red','red', 'red','red', 'red']
        print unique_labels
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'black'
            print col
            class_member_mask = (labels == k)
            print class_member_mask
        
            xy = X[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], '.', color=col, markersize=2)
        
#            xy = X[class_member_mask & ~core_samples_mask]
#            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#                     markeredgecolor='k')
        
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
        return labels

   def paint5(self, *dataset):
       colorset = np.linspace(0, 1, len(dataset))
       print colorset
       colors = plt.cm.Spectral(colorset)
       reduced_datas = []
       
       for data in dataset:
           redudata = PCA(n_components=2).fit_transform(data)
           reduced_datas.append(redudata)
           
       plt.figure()
       plt.clf()  # clear figure

       i = 0   
       for redudata in reduced_datas:
           plt.plot(redudata[:, 0], redudata[:, 1], 'k.', color = colors[i], markersize=2)
           i += 1               
       plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                'Centroids are marked with white cross')
       plt.grid(True)
       plt.xticks(())
       plt.yticks(())
       plt.show()


   def paint6(self, *dataset):
       colors = ['black','b', 'r', 'c', 'm']
       reduced_datas = []
       
       for data in dataset:
           redudata = PCA(n_components=2).fit_transform(data)
           reduced_datas.append(redudata)
       
#       
       db = skcluster.DBSCAN(eps=0.3, min_samples=10).fit(reduced_datas[0])
       
#        Step size of the mesh. Decrease to increase the quality of the VQ.
       h = .02 # point in the mesh [x_min, m_max]x[y_min, y_max].
       
#        Plot the decision boundary. For that, we will assign a color to each
       x_min, x_max = reduced_datas[0][:, 0].min() + 1, reduced_datas[0][:, 0].max() - 1
       y_min, y_max = reduced_datas[0][:, 1].min() + 1, reduced_datas[0][:, 1].max() - 1
       
       xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) # meshgrid
       
#       orginlabel = kmeans.labels_
#       noilabel = kmeans.predict(reduced_data2)
#       duallabel = kmeans.predict(reduced_data3)
       
#        Obtain labels for each point in mesh. Use last trained model.
       Z = db.fit_predict(np.c_[xx.ravel(), yy.ravel()])
#       print Z.shape[0]

       # Put the result into a color plot
       Z = Z.reshape(xx.shape)
       plt.figure()
       plt.clf()  # clear figure
       plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower') 
       i = 0   
       for redudata in reduced_datas:
           plt.plot(redudata[:, 0], redudata[:, 1], 'k.', color = colors[i], markersize=2)
           i += 1
#                   
       plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                'Centroids are marked with white cross')
       plt.xlim(x_min, x_max)
       plt.ylim(y_min, y_max)
       plt.grid(True)
       plt.xticks(())
       plt.yticks(())
       plt.show()

   def paint7(self, *dataset):
       colors = ['black','r', 'b', 'c', 'm']
       reduced_datas = []
       for data in dataset:
          redudata = PCA(n_components=3).fit_transform(data)
          reduced_datas.append(redudata)
          
       fig = plt.figure()
       plt.clf()
       ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
       plt.cla()
       
       i = 0   
       for redudata in reduced_datas:
           ax.scatter(redudata[:, 0], redudata[:, 1], redudata[:, 2], c=colors[i])
           i += 1
#       for redudata in reduced_datas:
#           ax.scatter(redudata[:, 0], redudata[:, 1], redudata[:, 2])
    
       ax.w_xaxis.set_ticklabels([])
       ax.w_yaxis.set_ticklabels([])
       ax.w_zaxis.set_ticklabels([])
       ax.set_xlabel('x')
       ax.set_ylabel('y')
       ax.set_zlabel('z')    
       plt.show()

   def paint8(self, *dataset):
       colors = ['black','b', 'r', 'c', 'm']
       reduced_datas = []
       
       for data in dataset:
           redudata = PCA(n_components=2).fit_transform(data)
           reduced_datas.append(redudata)
       
#       
       db = skcluster.DBSCAN(eps=0.3, min_samples=100).fit(reduced_datas[0])
       
       unique_labels = set(db.labels_)
       print unique_labels
       plt.figure()
       plt.clf()  # clear figure

       i = 0   
       for redudata in reduced_datas:
           plt.plot(redudata[:, 0], redudata[:, 1], 'k.', color = colors[1], markersize=2)
           i += 1
#                   
       plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                'Centroids are marked with white cross')
       plt.grid(True)
       plt.show()
       return unique_labels

   def paint9(self, *dataset):
       colors = ['black','r', 'y','m','c','b','']
       
       # 数据集分开降维图
       newdataset = []
       plt.figure()
       plt.clf()  # clear figure     
       j = 0 
       for ndata in dataset:
           newdataset.extend(ndata)
           redundata = PCA(n_components=2).fit_transform(np.mat(ndata))
           plt.plot(redundata[:, 0], redundata[:, 1], 'k.', color = colors[j], markersize=2)
           j += 1
       plt.show()   
       
       # 数据集降维合并图
       plt.figure()
       plt.clf()  # clear figure     
       newDD = PCA(n_components=2).fit_transform(np.mat(dataset[0]))
       m = 0
       for ndata in dataset:
           m += 1
           if m==1: continue
           rdata = PCA(n_components=2).fit_transform(np.mat(ndata))
           newDD = np.vstack((newDD,rdata))
       plt.plot(newDD[:, 0], newDD[:, 1], 'k.', color = colors[0], markersize=2)
       plt.show() 
       
       # 数据集合并降维图      
       redudata = PCA(n_components=2).fit_transform(np.mat(newdataset))
       print np.shape(redudata)
       
       plt.figure()
       plt.clf()  # clear figure       
#       plt.plot(redudata[:, 0], redudata[:, 1], 'k.', color = colors[0], markersize=2)
       redud1 = redudata[:len(dataset[0])]
       redud2 = redudata[len(dataset[0]):]
       print np.shape(redud1)
       print np.shape(redud2)
       plt.plot(redud1[:, 0], redud1[:, 1], 'k.', color = colors[0], markersize=2)
       plt.plot(redud2[:, 0], redud2[:, 1], 'k.', color = colors[1], markersize=2)
       plt.show()
            
       # KMeans图
       kmeans = self.KMeans("k-means++", 5, 10)
       kmeans.fit(newDD)
       
       plt.figure()
       plt.clf()  # clear figure
       unique_labels = set(kmeans.labels_)
       colorset = np.linspace(0, 1, len(unique_labels))
       colorsets = plt.cm.Spectral(colorset)
       print unique_labels
       labels = kmeans.labels_
       i = 0
       for label in unique_labels:
           print colorset[i]
           print label
           if label == 0:
               newD =  newDD[np.nonzero(labels==0)[0],:]
               plt.plot(newD[:, 0], newD[:, 1], 'k.', color = colorsets[i], markersize=2)
           else:
               newD =  newDD[np.nonzero(labels==label)[0],:]
               print labels == label
               plt.plot(newD[:, 0], newD[:, 1], 'k.', color = colorsets[i], markersize=2)
           i += 1                 
       plt.show()
       
       # DBSCAN图
#       db = skcluster.DBSCAN(eps=0.8, min_samples=500).fit(newDD)    
#       plt.figure()
#       plt.clf()  # clear figure
#       unique_labels = set(db.labels_)
#       print unique_labels
#       colorset = np.linspace(0, 1, len(unique_labels))
#       colorsets = plt.cm.Spectral(colorset)
#       labels = db.labels_
#       i = 0
#       for label in unique_labels:
##           print colorsets[i]
##           print label
#           if label == 0:
#               newD =  newDD[np.nonzero(labels==0)[0],:]
#               plt.plot(newD[:, 0], newD[:, 1], 'k.', color = colorsets[i], markersize=2)
#           else:
#               newD =  newDD[np.nonzero(labels==label)[0],:]
#               print labels == label
#               plt.plot(newD[:, 0], newD[:, 1], 'k.', color = colorsets[i], markersize=2)
#           i += 1                 
#       plt.show()
       
       
       # Birch图
#       birch = skcluster.Birch(n_clusters=2).fit(newDD)
#       plt.figure()
#       plt.clf()  # clear figure
#       unique_labels = set(birch.labels_)
#       print unique_labels
#       colorset = np.linspace(0, 1, len(unique_labels))
#       colorsets = plt.cm.Spectral(colorset)
#       labels = birch.labels_
#       i = 0
#       for label in unique_labels:
##           print colorsets[i]
##           print label
#           if label == 0:
#               newD =  newDD[np.nonzero(labels==0)[0],:]
#               plt.plot(newD[:, 0], newD[:, 1], 'k.', color = colorsets[i], markersize=2)
#           else:
#               newD =  newDD[np.nonzero(labels==label)[0],:]
#               print labels == label
#               plt.plot(newD[:, 0], newD[:, 1], 'k.', color = colorsets[i], markersize=2)
#           i += 1                 
#       plt.show()  

       # spcectral图
#       spcectral = skcluster.SpectralClustering(n_clusters=2,
#                                          eigen_solver='arpack',
#                                          affinity="nearest_neighbors").fit(newDD)
#       plt.figure()
#       plt.clf()  # clear figure
#       unique_labels = set(spcectral.labels_)
#       print unique_labels
#       colorset = np.linspace(0, 1, len(unique_labels))
#       colorsets = plt.cm.Spectral(colorset)
#       labels = spcectral.labels_
#       i = 0
#       for label in unique_labels:
##           print colorsets[i]
##           print label
#           if label == 0:
#               newD =  newDD[np.nonzero(labels==0)[0],:]
#               plt.plot(newD[:, 0], newD[:, 1], 'k.', color = colorsets[i], markersize=2)
#           else:
#               newD =  newDD[np.nonzero(labels==label)[0],:]
#               print labels == label
#               plt.plot(newD[:, 0], newD[:, 1], 'k.', color = colorsets[i], markersize=2)
#           i += 1                 
#       plt.show() 
       return labels

      
if __name__=='__main__':
    
    clustering = Clustering()
    
    filename = "SubSpectrumData/"+"IonGroups_Int"
    filename2 = "SubSpectrumData/"+"IonGroups_NoiInt"
    filename3 = "SubSpectrumData/"+"IonGroups_DualInt"
    filename4 = "SubSpectrumData/"+"IonGroups_AtypeInt"
    filename5 = "SubSpectrumData/"+"IonGroups_yNH3Int"
    filename6 = "SubSpectrumData/"+"IonGroups_yH2OInt"
    filename7 = "SubSpectrumData/"+"IonGroups_bH2OInt"
    filename8 = "SubSpectrumData/"+"IonGroups_bNH3Int"
    filename9 = "SubSpectrumData/"+"IonGroups_y46-Int"
    filename10 = "SubSpectrumData/"+"IonGroups_y45-Int"
    filename11 = "SubSpectrumData/"+"IonGroups_y10+Int"
 

   
#    dataset = np.mat(clustering.loadDataSet(filename)) 
#    dataset2 = np.mat(clustering.loadDataSet(filename2)) 
#    dataset3 = np.mat(clustering.loadDataSet(filename3))
#    dataset4 = np.mat(clustering.loadDataSet(filename4))
#    dataset5 = np.mat(clustering.loadDataSet(filename5))
#    dataset6 = np.mat(clustering.loadDataSet(filename6))
#    dataset7 = np.mat(clustering.loadDataSet(filename7))
#    dataset8 = np.mat(clustering.loadDataSet(filename8))
#    dataset9 = np.mat(clustering.loadDataSet(filename9))
#    dataset10 = np.mat(clustering.loadDataSet(filename10))
#    dataset11 = np.mat(clustering.loadDataSet(filename11))
    
#    clustering.paint6(dataset,dataset3)
#    clustering.paint7(dataset, dataset2, dataset3, dataset4)
#    label = clustering.paint8(dataset, dataset3)
#    print label
#    dataset12 = np.mat(dataset,dataset2)
    
#    newdataset = np.mat(clustering.loadDataSet(filename).extend(clustering.loadDataSet(filename3))) 
#    clustering.paint2(newdataset)
#    clustering.paint2(dataset, dataset2, dataset3)
#    clustering.paint2(dataset, dataset2, dataset3, dataset4)
#    clustering.paint3(dataset, dataset2, dataset3, dataset4)
    
#    data=np.clip(np.random.randn(5,5),-1,1) #生成随机数据,5行5列,最大值1,最小值-1 
#    
#    KMeansPP = clustering.sl_clusting(clustering.KMeans("k-means++", 4, 10),"kmeans++", dataset)
#    label1 = KMeansPP.labels_
#    
#    KMeansR = clustering.sl_clusting(clustering.KMeans("random", 4, 10),"random", dataset)
#    label2 = KMeansR.labels_
#    
#    pca = PCA(n_components=4).fit(dataset)
#    comp =  pca.components_
#    dataset2 = PCA(n_components=10).fit_transform(dataset)
#    label3 = clustering.sl_clusting(clustering.KMeans(pca.components_, 4, 1),"PCA-based", dataset)
#    label4 = clustering.sl_clusting(clustering.KMeans("k-means++", 4, 1),"PCA-based2", dataset2)    
    
#    for linkage in ('ward', 'average', 'complete'):
#        aggclustering =clustering.sl_clusting(clustering.AgglomerativeClustering(linkage),linkage+"_agg",dataset2[:10000])

#################################################test 2#####################################
#    reduce_data = PCA(n_components=3).fit_transform(dataset)
#    maxIndex = 0
#    maxScore = 0
#    for i in range(2,10):
#        for j in range(2):
#            print "i="+str(i)
#            KMeansPP,pscore = clustering.sl_clusting(clustering.KMeans("k-means++", i, 10),"kmeans++", reduce_data)
#            label1 = KMeansPP.labels_    
#            KMeansR,Rscore = clustering.sl_clusting(clustering.KMeans("random", i, 10),"random", reduce_data)
#            label2 = KMeansR.labels_    
#            if pscore > maxScore:
#                maxScore = pscore
#                maxIndex = i
#            if Rscore > maxScore:
#                maxScore = Rscore
#                maxIndex = i
#            print "*"*20
#    
#    print maxIndex,maxScore

#################################################test 3#####################################
#    reduce_data = PCA(n_components=3).fit_transform(dataset2)
#    maxIndex = 0
#    maxScore = 0
#    for i in range(2,10):
#        for j in range(2):
#            print "i="+str(i)
#            KMeansPP,pscore = clustering.sl_clusting(clustering.KMeans("k-means++", i, 10),"kmeans++", reduce_data)
#            label1 = KMeansPP.labels_    
#            KMeansR,Rscore = clustering.sl_clusting(clustering.KMeans("random", i, 10),"random", reduce_data)
#            label2 = KMeansR.labels_    
#            if pscore > maxScore:
#                maxScore = pscore
#                maxIndex = i
#            if Rscore > maxScore:
#                maxScore = Rscore
#                maxIndex = i
#            print "*"*20
#    
#    print maxIndex,maxScore

#################################################test 4#####################################  
#    reduce_data = PCA(n_components=3).fit_transform(dataset)
##    reduce_data2 = PCA(n_components=3).fit_transform(dataset2)
##    reduce_data3 = PCA(n_components=3).fit_transform(dataset3)
#    reduce_data4 = PCA(n_components=3).fit_transform(dataset4)
#    estimator = clustering.KMeans("k-means++", 4, 10)
#    estimator.fit(reduce_data)
##    labels1 = estimator.labels_
##    labels2 = estimator.predict(reduce_data2)
##    labels3 = estimator.predict(reduce_data3)
##    labels4 = estimator.predict(reduce_data4)

#################################################test 4#####################################  
#    print dataset[:,0]
#    print dataset[:,1]
#    k = np.mat(dataset[:,0], dataset[:,1])

#################################################test 5#####################################  
#    print dataset[:,0]
##    sampleDatas = clustering.randomSample(4000, dataset, dataset2, dataset3, dataset4)
##    labels = clustering.paint4(sampleDatas)
#    clustering.paint3(sampleDatas)
#################################################test 5#####################################  
#    print dataset[:,0]
#    sampleDatas = clustering.randomSample(2000, dataset, dataset2, dataset3, dataset4,\
#                                          dataset5, dataset6, dataset7, dataset8,\
#                                          dataset9, dataset10, dataset11)                                               
#    labels = clustering.paint4(sampleDatas)
#    clustering.paint3(sampleDatas)
#    clustering.paint5(dataset, dataset2, dataset3, dataset4,\
#                      dataset5, dataset6, dataset7, dataset8,\
#                      dataset9, dataset10, dataset11)
#    clustering.paint5(dataset, dataset2, dataset3, dataset4)

#################################################test 6#####################################  
#    sampleDatas = clustering.randomSample(3000, dataset, dataset2, dataset3, dataset4,\
#                                          dataset5, dataset6, dataset7, dataset8,\
#                                          dataset9, dataset10, dataset11)                                               
#    clustering.paint2(sampleDatas)
#    clustering.paint3(sampleDatas)
#    clustering.paint4(sampleDatas)
#    clustering.paint5(dataset, dataset2, dataset3, dataset4,\
#                      dataset5, dataset6, dataset7, dataset8,\
#                      dataset9, dataset10, dataset11)
#    clustering.paint5(dataset, dataset2, dataset3, dataset4)

#################################################test 6#####################################  
    newD1 = clustering.loadDataSet(filename)
    newD3 = clustering.loadDataSet(filename3)
#    newD1.extend(newD3)
#    newData = np.mat(newD1) # merge new dataset
    lab = clustering.paint9(newD1,newD3)

    