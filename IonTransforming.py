# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 21:10:25 2016

@author: Administrator
"""
import math
import numpy as np 
class IonTransforming(object):
    
    def __init__(self,mass, num, binlen, arealen):
        self.pepmass = mass
        self.binnum = num
        self.binlen = binlen
        self.arealen = arealen
    
    def generateIonMass(self, peaks, ionpois):
        """
        generate peak's assume mass according ion positons
        """
        offset = self.getOffset(ionpois)
        peakAllMass = {}
        for peak in peaks:
            mz = peak.getMz()
            ionmass = []
            for val in offset:
                if val > mz: # c-term
                    nmass = val - mz
                else:
                    nmass = mz -val
                ionmass.append((val, nmass))
            peakAllMass[peak] = ionmass
        return peakAllMass

    def getOffset(self, ionpois):
        """
        compute ion offset according ion position
        """
        offset = []
        for poi in ionpois:
            if poi <= self.binnum:
                temp = math.ceil(poi * self.binlen - self.arealen)
                offset.append(temp)
            else:
                temp = math.ceil((poi - self.binnum) * self.binlen - self.arealen)
                temp = self.pepmass + temp
                offset.append(temp)
        return offset
    
    def varAnalysis(self, iongroups, theshold):
        """
        """
        ionsum = np.sum(iongroups, axis = 1)
        ionvar = np.var(ionsum)
        if ionvar < theshold:
            return -1;
        else:
            index = np.argmax(ionsum) # get the index of max number
        return index

if __name__=="__main__":
    
    print math.ceil(2.3)
#    a=np.array([1,2,3,4,5,6,7])
    a = np.array([8,9,10,11])
    print np.mean(a)
    print np.var(a)
                
            