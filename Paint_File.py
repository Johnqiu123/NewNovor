# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:59:57 2015

@author: Johnqiu
"""
import matplotlib.pyplot as plt 
import numpy as np
class Paint_File(object):
    
    def paint(self,xdata,ydata):
        # paint vline
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.vlines(xdata,[0],ydata) 
        ax.set_xticks(xdata) # link with xdata
#        plt.xlabel('m/z')
#        plt.ylabel('Relative Counts')
        plt.show()
    
    def paintploylines(self, xdata, ydata):
        # paint ploylines
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdata,ydata)
#        plt.xlabel("position")
#        plt.ylabel("ChiSquared")
        plt.show()
    
    def paintBar(self, xdata, ydata):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(xdata, ydata, 0.5, align='center')
#        plt.xlabel("pepbond")
        ax.set_xticks(xdata) # link with xdata
        ax.set_xticklabels(xdata)
        plt.show()   
        
    def paintmultiBars(self, xdata, *ydatas):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        width = 0.25
        i = -1
#        xdata = np.array(xdata)
        newxdata = np.arange(1,len(xdata)+1)
        for ydata in ydatas:
            ax.bar(newxdata + width * i, ydata, width, color=plt.rcParams['axes.color_cycle'][i+1])
            i += 1
        ax.set_xticks(newxdata) # link with xdata
        ax.set_xticklabels(xdata)
        plt.show()
    
    def paintmultiploylines(self, xdata, ydata, ydata2):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.plot(xdata,ydata)
        ax2.plot(xdata,ydata2)
        ax1.set_xticks(xdata) 
        ax2.set_xticks(xdata) 
        plt.show()       
        


if __name__ == '__main__':
    
    paint = Paint_File()
#    x = [0,1,2,3]
#    y = [6,4,3,2]
#    paint.paintploylines(x,y)
#    x = [12,34,14,21]
#    y = [k for k in range(1,len(x)+1)]
#    y1 = ['a','b','c','d']
#    print x,y
#    k = zip(x,y)
#    k.sort(key=lambda x:x[0],reverse=True) # 按第0列排序
##    print [m for m in k[]]
#    ch = []
#    for i in range(len(k)):
#        ch.append(k[i][0])
#    x.sort(reverse=True)
#    print ch,x
#    print k[0:4]
##    paint.paintploylines(y,x)
#    
#    paint.paintBar(y,x)
    
    x=[1789, 229, 49, 1949, 329, 369, 1229, 1519, 1809, 339, 1909, 1239, 1689, 59, 509, 1509] 
    y1=[900, 13730, 7822, 356, 13022, 7204, 569, 5013, 532, 9340, 288, 586, 14139, 7584, 21454, 4338] 
    y2=[2222, 5000, 3041, 1728, 7165, 2144, 1694, 8484, 1765, 5535, 1743, 1991, 27195, 3697, 11352, 8044]
    y3=[7814, 1294, 572, 8272, 1232, 679, 10094, 14312, 7673, 1160, 7940, 10017, 25790, 630, 3166, 13865]
    
#    x = np.arange(len(y1))
#    result = np.array(x) + 0.5 # 添加一项
#    print result
#    x = np.arange(5)
#    y1, y2, y3 = np.random.randint(1, 25, size=(3, 5))
    print x,y1,y2,y3
    paint.paintmultiBars(x,y1,y2,y3)
    