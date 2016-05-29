# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:13:13 2015

@author: Johnqiu
"""

class PaintBins(object):
    
    # read SubSpectrumData
    def loadSubSpectrum(self,filename):
        with open(filename) as fr:
            print("OK")
            for data in fr:
                if "allNtermbins" in data:
                    print("OK")
                if "allCtermbins" in data:
                    print data
        fr.close()
    
    # draw picture
    def paint(self,xdata,ydata):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.vlines(xdata,[0],ydata) # 画竖线
        plt.xlabel('m/z')
        plt.ylabel('Relative Counts')
        plt.show()

if __name__ =='__main__':
   file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
   paintBins = PaintBins()
   paintBins.loadSubSpectrum(file_name)  