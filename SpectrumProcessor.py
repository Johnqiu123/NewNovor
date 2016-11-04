# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 16:57:37 2016

@author: Administrator
"""
from SpectrumParser import SpectrumParser
from write_file import write_file
import cPickle as cpickle
class SpectrumProcessor(object):
    
    def preprocessing(self, spectra, windownum, peaknum):
         """
         preprocessing spectrum without sequance
         """
         sampleDict = {}
         for spectrum in spectra:
             assert spectrum.getAnnotation() is None
             parentMass = spectrum.getParentMass()
             winSize = parentMass / windownum
             winList = [0]
             mass = 0
             for k in range(windownum+1):
                 mass = mass + winSize
                 winList.append(mass)
             
             i = 1           
             selecpeaks = []
             peaks = []
             for peak in spectrum.getPeaks():
                mz = peak.getMz()
                while i < len(winList)-1:             
                    if mz > winList[i]: 
                        i += 1
                        if peaks != []:
                           temppeaks = self.sortedPeak(peaks,peaknum)
                           selecpeaks.extend(temppeaks)
                        peaks = []
                    else:  
                        peaks.append(peak)
                        break
                if i >= len(winList) : break   
             title =  spectrum.getTitle()   
             sampleDict[title] = selecpeaks
#             print spectrum.getParentMass()
#             print spectrum.getTitle()
#             break
         return sampleDict
     
    def sortedPeak(self, peaks, peaknum):
         """
         Sort peaks
         """
         peaklen = len(peaks)
         if peaklen <= peaknum:
             return peaks
         else:
             intdict = {}
             for peak in peaks:
                 intensity = peak.getIntensity()
                 intdict[intensity] = peak
             
             pintdict = sorted(intdict.iteritems(), key = lambda x:x[0], reverse = True)
             poilist = []
             for i in range(peaknum):
                 temppeak = pintdict[i][1]
#                 print temppeak
                 mz = temppeak.getMz()
                 peakturple = (mz, temppeak)
                 poilist.append(peakturple)
            
             ppoidict = sorted(poilist, key = lambda x:x[0])
             newpeaks = []
             for i in range(peaknum):
                 tpeak = ppoidict[i][1]
                 newpeaks.append(tpeak)
    #             print poidict
    #             print ppoidict
             return newpeaks
    
    def writetoFile(self, spects, filename):
        wf = write_file()
        wf.writeSampleFile(spects,filename)
    
    def writetoFile_cp(self, sampleDict, filename):
        # write calculated bins into file
        with open(filename,'w') as fw:
            cpickle.dump(sampleDict, fw)
    
    def generateSampleFile_cp(self, filename):
        # write calculated bins into file
        with open(filename,'r') as fr:
            data = cpickle.load(fr)
            return data
        
        

if __name__ == '__main__':
    
    
    ################################ Test 1#####################################
#    file_name = 'data/1_3M_JD_sample1_A.mgf'
#    parser = SpectrumParser()
#    specs = list(parser.readSpectrum(file_name))
##    print specs[0].getPeaks()
# 
#    xaix = []
#    yaix = []   
#    for peak in specs[0].getPeaks():
#        xaix.append(peak.getMz())
#        yaix.append(peak.getIntensity())
#
#    pt = Paint_File()
#    pt.paint(xaix,yaix)
#
#
#    end = time.clock()
##    print len(specs)
#    print 'time consuming %s seconds.' % (end-start)

################################ Test 1#####################################
    file_name = 'data/SimulateData_test'
    parser = SpectrumParser()
    spectprocer = SpectrumProcessor()
    specs = parser.readSpectrum(file_name)   
    sampleDict = spectprocer.preprocessing(specs,10, 2)
    file_name2 = 'data/SimulateData_test_sample_cp'
    spectprocer.writetoFile_cp(sampleDict,file_name2)
    



#    k  = [x for x in range(0,5,2)]
#    for x in range(5):
#        print x
#    print k
#    p = []
#    if p == []:
#        print 1