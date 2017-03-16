# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:59:57 2015

技巧:一条语句画不了，就用两条语句

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
#        ax.set_xticks(xdata) # link with xdata
        plt.xlabel('M/Z')
#        plt.xlabel('Bins')
#        plt.xlabel('PeptideLen')
        
#        plt.ylabel('0/1 Counts')
#        plt.ylabel('Intensity Counts')
#        plt.ylabel('Chi-square Value')
#        plt.ylabel('Peptide Number')
        plt.ylabel('Intensity')
        plt.title('Spectrum')
        plt.show()
    
    def paintploylines(self, xdata, ydata):
        # paint ploylines
#        ymean = np.mean(ydata)
#        xmean = [i for i in range(9)]
#        ymeans = [ymean for i in range(9)]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdata,ydata, 'ro')
#        ax.plot(xmean,ymeans,'r--')
        ax.plot(xdata,ydata)
        ax.axis([0, 8, 0, 200])
        plt.xlabel("Ion Type")
        plt.ylabel("F value")
        plt.title("F value Distribution")
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
        # color=plt.rcParams['axes.color_cycle'][i+1]
        labels = ["left","center","right"]
        patterns = ('/','-', '\\')
        newxdata = np.arange(1,len(xdata)+1)
        for ydata in ydatas:
            ax.bar(newxdata + width * i,ydata, width, label= labels[i+1], color='white', edgecolor='black',hatch=patterns[i+1])
            i += 1
        ax.set_xticks(newxdata) # link with xdata
        ax.set_xticklabels(xdata)
        plt.xlabel("ion position")
        plt.ylabel("The number of occurrences")
        plt.legend()
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
        
    def paintDoubleploylines(self, xdata, ydata, ydata2):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        a1, = ax.plot(xdata,ydata, 'go')
        a2, = ax.plot(xdata,ydata2,'r^')
        a11, = ax.plot(xdata,ydata)
        a22, = ax.plot(xdata,ydata2)
        plt.xlim(8,15)
#        plt.ylim(0,3500,100)
        plt.ylim(0,1000,100)
        plt.xlabel("Peptide Length")
        plt.ylabel("Vartex")
#        plt.ylabel("Edge")
        plt.title("Spectrum Graph")
#        plt.legend()
        plt.legend([(a1,a11),(a2,a22)],['Newnovo','Pepnovo'], loc =2)
        
#        z = np.random.randn(10)
#        p1a, = plt.plot(z, "ro", ms=10, mfc="r", mew=2, mec="r") # red filled circle
#        p1b, = plt.plot(z[:5], "w+", ms=10, mec="w", mew=2) # white cross
#        plt.legend([p1a, (p1a, p1b)], ["Attr A", "Attr A+B"])
#        plt.show()
        
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
#    print x,y1,y2,y3
    paint.paintmultiBars(x,y1,y2,y3)
################################ Test 1#####################################
#    xaix = [i for i in range(1,8)]
##    yaix = [38.34,35.24,30.44,28.33,33.55,34.11,36.76]
#    yaix = [38.34,35.24,30.44,28.33,157.33,34.11,36.76]
#    xvar = np.mean(yaix)
#    print xvar
#    paint.paintploylines(xaix,yaix)
################################ Test 1#####################################
#    xaix = [i for i in range(8,16)]
#    yaix=[88.2,99.45,110.34,124.22,149.33,158.33,173.44,200.56]
#    yaix2 = [301.22, 320.2, 400.22, 510.32, 540.34, 645.21, 750.22, 950.44]
##    print xvar
#    paint.paintDoubleploylines(xaix,yaix,yaix2)

################################ Test 2#####################################
#    xaix = [i for i in range(8,16)]
#    yaix=[500.1,750.22,1200.11,1250.34,1400.22,1350.22,1450.33,1500.56]
#    yaix2 = [1300.22, 1500.32, 2050.22, 2300.32, 2220.34, 3100.21, 3000.22, 3400.44]
##    print xvar
#    paint.paintDoubleploylines(xaix,yaix,yaix2)