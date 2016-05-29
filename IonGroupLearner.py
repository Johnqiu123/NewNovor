# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 17:16:51 2016

@author: Johnqiu
"""
import time
import random
from Paint_File import Paint_File 
import numpy as np
from SubSpectrumGenerator import SubSpectrumGenerator
from ChiSquared import ChiSquared
from scripts import ProteinWeightDict
import itertools
import pandas as pd
from SubSpectrumProcessor import SubSpectrumProcessor
from write_file import write_file
from SpectrumParser import SpectrumParser
import cPickle as cpickle

class IonGroupLearner(object):
    
    def generateIonGroup(self, subspects, pois):
        """
          generate groups of ionlists
          
          Args:
           -subspects:      a set of subspectrum
           -pois         a set of poistion
          
          Return:
           -ionLists    
        """  
        choicepois = [1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1] # by people
        newpois = [choicepois[i] and pois[i] for i in range(len(pois))]
        newpois = filter((lambda x: x>0), newpois)
        
        print newpois
        
        ionLists = []
        for subspect in subspects:
            num = subspect.getNumberofNBins()  # get the number of bins
            ionList = []
            for newpoi in newpois:
                if newpoi <= num: # N-term
                    flag = subspect.getNtermBins()[newpoi]
                    ionList.append(1 if flag else 0)
                else:  # C-term
                    flag = subspect.getCtermBins()[newpoi-num]
                    ionList.append(1 if flag else 0)
            ionLists.append(ionList)      
        return ionLists

    def generateIonGroup_Int(self, subspects, pois, spectMaxInt):
        """
          generate groups of ionlists
          
          Args:
           -subspects:      a set of subspectrum with intensity
           -pois         a set of poistion
          
          Return:
           -ionLists    
        """  
        choicepois = [1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1] # by people
        newpois = [choicepois[i] and pois[i] for i in range(len(pois))]
        newpois = filter((lambda x: x>0), newpois)
        
        print newpois
        
        ionLists = []
        for subspect in subspects:
            title = subspect.getTitle()
            maxInt = spectMaxInt[title]
            num = subspect.getNumberofNBins()  # get the number of bins
            ionList = []
            for newpoi in newpois:
                if newpoi <= num: # N-term
                    flag = subspect.getNtermBins()[newpoi]
                    ionList.append(round(flag/maxInt*100,3) if flag else 0) # generational
                else:  # C-term
                    flag = subspect.getCtermBins()[newpoi-num]
                    ionList.append(round(flag/maxInt*100,3) if flag else 0)
            ionLists.append(ionList)      
        return ionLists
    
    def generateMaxIntentity(self, spectra):
        spectMaxInt = {}
        for spectrum in spectra:
            title = spectrum.getTitle()
            maxInt = 0.0        
            for peak in spectrum.getPeaks():
                if peak.getIntensity() > maxInt:
                    maxInt = peak.getIntensity()
            spectMaxInt[title] = maxInt
        return spectMaxInt

    def generateMaxIntentityFile(self, filename):
        with open(filename,'r') as fr:
            spectMaxInt = cpickle.load(fr)
            return spectMaxInt
                                
    def generateIonGroupFile(self, filename):
        ionLists = []
        with open(filename) as input_data:
            for line in input_data:
                line = [float(i) for i in line[1:len(line)-2].split(',')]
                ionLists.append(line)
        return ionLists
    
    def generateIonPoitionFile(self, filename):
        ionpois = []
        with open(filename) as input_data:
            for line in input_data:
                ionpois.append(int(line))
        return ionpois
        
    def calcuIonGroup(self, iongroups):
        ionCountDict = {}
        for iongroup in iongroups:
            iongroup = np.array(iongroup)
            num = iongroup.sum()
            if ionCountDict.has_key(num):
                ionCountDict[num] += 1
            else:
                ionCountDict[num] = 1
        return ionCountDict

    def calcuIonGroup_axis(self, iongroups):
        """
          calculate groups of ionlists by axis
          
          Args:
           -iongroups:      groups of ionlists
          
          Return:
           -ionLists    
        """          
        ionLists = np.zeros(int(len(iongroups[0])))
        for iongroup in iongroups:
            ionLists += np.array(iongroup)      
        return ionLists
    
    def paintIonCount(self, ionCounts):
        xdata = []
        ydata = []
        pt = Paint_File()
        for key in ionCounts:
            xdata.append(key)
            ydata.append(ionCounts[key])
        pt.paintBar(xdata, ydata)
    
    def paintMaxInt(self, spectMaxInt):
        xdata = [x for x in range(1,len(spectMaxInt)+1)]
        ydata = []
        pt = Paint_File()
        for key in spectMaxInt:
            ydata.append(spectMaxInt[key])
        
        pt.paintploylines(xdata, ydata)
        
    
    
    

if __name__=='__main__':
    
    start = time.clock()
################################ Test 1#####################################
#    filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_20151208"
#    filename2 ="SubSpectrumData/"+"new_CHPP_LM3_RP3_2_Noise_20151208"
#    subparser = SubSpectrumGenerator()
#    subspects = list(subparser.generateSubSpecfile(filename))
#    noisubspects = list(subparser.generateNoiSubfile(filename2)) # noise
#
#    subprocessor = SubSpectrumProcessor()
#    allNtermbins,allCtermbins,allSubbins,subNum = subprocessor.calculateBins(subspects)
#    allNOiNtermbins,allNoiCtermbins,allNoibins,noiNum = subprocessor.calculateBins(noisubspects)
#    
###     n-term
#    NchiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allNtermbins,allNOiNtermbins)
###     c-term
#    CchiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allCtermbins,allNoiCtermbins)
###     all
#    chiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allSubbins,allNoibins)
###    
###    
#    poiChiValues,poichiV = subprocessor.sortChiValues(chiValues)
#    orginalpois = [poiChiValues[i][1] for i in range(len(poiChiValues))][0:21] # get top 10 chivalues
##    randsubspects = subprocessor.generateRandSample(subspects,10)
#
##     file_name4 = "SubSpectrumData/"+"IonGroups"
#    iglearner = IonGroupLearner()
#    ionLists = iglearner.generateIonGroup(subspects, orginalpois)
###    
#    wfile = write_file()
#    wfile.writeIonPoi(orginalpois)
#    wfile.writeIonGroups(ionLists,file_name4)
    
#    filename = "SubSpectrumData/"+"minIonGroups"
#    ionLists = iglearner.generateIonGroupFile(filename)
#    
#    ionCounts = iglearner.calcuIonGroup(ionLists)
#    iglearner.paintIonCount(ionCounts)
#    
#    print ionCounts

################################ Test 2#####################################
#    filename1 = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_20151208"
#    subparser = SubSpectrumGenerator()
#    subspects = list(subparser.generateSubSpecfile(filename1))
#  
#    filename2 = "SubSpectrumData/"+"IonPostion"   
#    iglearner = IonGroupLearner()
#    orginalpois = iglearner.generateIonPoitionFile(filename2) # directly get original pois from file
#    
#    ionLists = iglearner.generateIonGroup(subspects, orginalpois)
#    
#       
#    file_name4 = "SubSpectrumData/"+"IonGroups"
#    wfile = write_file()
##    wfile.writeSpectMaxInt(spectMaxInt)
#    
##    iglearner.paintMaxInt(spectMaxInt)
##    wfile.writeIonPoi(orginalpois)
#    wfile.writeIonGroups(ionLists, file_name4)
#    
##    ionLists = iglearner.generateIonGroup(subspects, orginalpois)
##    
################################ Test 2#####################################
#    filename1 = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_intensity_20160120"
#    subparser = SubSpectrumGenerator()
#    subspects = list(subparser.generateSubSpecfile(filename1,'intensity'))
#  
#    filename2 = "SubSpectrumData/"+"IonPostion"   
#    iglearner = IonGroupLearner()
#    orginalpois = iglearner.generateIonPoitionFile(filename2) # directly get original pois from file
#    
##    file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
##    parser = SpectrumParser()
##    specs = list(parser.readSpectrum(file_name)) # orignal datas file
##    
##    spectMaxInt = iglearner.generateMaxIntentity(specs)
#    
#    file_name3 = "SubSpectrumData/"+"SpectMaxInt"
#    spectMaxInt = iglearner.generateMaxIntentityFile(file_name3)
#    
#    ionLists = iglearner.generateIonGroup_Int(subspects, orginalpois, spectMaxInt)
#    
#    
#    file_name4 = "SubSpectrumData/"+"IonGroups_Int"
#    wfile = write_file()
##    wfile.writeSpectMaxInt(spectMaxInt)
#    
##    iglearner.paintMaxInt(spectMaxInt)
##    wfile.writeIonPoi(orginalpois)
#    wfile.writeIonGroups(ionLists, file_name4)
#    
##    ionLists = iglearner.generateIonGroup(subspects, orginalpois)
##    

################################ Test 3#####################################
#    filename1 = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_Noise_intensity_20160120"
#    subparser = SubSpectrumGenerator()
#    noisubspects = list(subparser.generateNoiSubfile(filename1,'intensity'))
#  
#    filename2 = "SubSpectrumData/"+"IonPostion"   
#    iglearner = IonGroupLearner()
#    orginalpois = iglearner.generateIonPoitionFile(filename2) # directly get original pois from file
#    
##    file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
##    parser = SpectrumParser()
##    specs = list(parser.readSpectrum(file_name)) # orignal datas file
##    
##    spectMaxInt = iglearner.generateMaxIntentity(specs)
#    
#    file_name3 = "SubSpectrumData/"+"SpectMaxInt"
#    spectMaxInt = iglearner.generateMaxIntentityFile(file_name3)
#    
#    ionLists = iglearner.generateIonGroup_Int(noisubspects, orginalpois, spectMaxInt)
#    
#    
#    file_name4 = "SubSpectrumData/"+"IonGroups_NoiInt"
#    wfile = write_file()
##    wfile.writeSpectMaxInt(spectMaxInt)
#    
##    iglearner.paintMaxInt(spectMaxInt)
##    wfile.writeIonPoi(orginalpois)
#    wfile.writeIonGroups(ionLists, file_name4)
#    
##    ionLists = iglearner.generateIonGroup(subspects, orginalpois)
##
    
################################ Test 4#####################################
#    filename1 = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_dual_intensity_20160120"
#    subparser = SubSpectrumGenerator()
#    dualsubspects = list(subparser.generateSubSpecfile(filename1,'intensity'))
#  
#    filename2 = "SubSpectrumData/"+"IonPostion"   
#    iglearner = IonGroupLearner()
#    orginalpois = iglearner.generateIonPoitionFile(filename2) # directly get original pois from file
#    
##    file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
##    parser = SpectrumParser()
##    specs = list(parser.readSpectrum(file_name)) # orignal datas file
##    
##    spectMaxInt = iglearner.generateMaxIntentity(specs)
#    
#    file_name3 = "SubSpectrumData/"+"SpectMaxInt"
#    spectMaxInt = iglearner.generateMaxIntentityFile(file_name3)
#    
#    ionLists = iglearner.generateIonGroup_Int(dualsubspects, orginalpois, spectMaxInt)
#    
#    
#    file_name4 = "SubSpectrumData/"+"IonGroups_DualInt"
#    wfile = write_file()
##    wfile.writeSpectMaxInt(spectMaxInt)
#    
##    iglearner.paintMaxInt(spectMaxInt)
##    wfile.writeIonPoi(orginalpois)
#    wfile.writeIonGroups(ionLists, file_name4)
#    
##    ionLists = iglearner.generateIonGroup(subspects, orginalpois)
###    
################################ Test 5#####################################
#    filename1 = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_atype_intensity_20160120"
#    subparser = SubSpectrumGenerator()
#    atypesubspects = list(subparser.generateSubSpecfile(filename1,'intensity'))
#  
#    filename2 = "SubSpectrumData/"+"IonPostion"   
#    iglearner = IonGroupLearner()
#    orginalpois = iglearner.generateIonPoitionFile(filename2) # directly get original pois from file
#    
#    print orginalpois
##    file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
##    parser = SpectrumParser()
##    specs = list(parser.readSpectrum(file_name)) # orignal datas file
##    
##    spectMaxInt = iglearner.generateMaxIntentity(specs)
#    
#    file_name3 = "SubSpectrumData/"+"SpectMaxInt"
#    spectMaxInt = iglearner.generateMaxIntentityFile(file_name3)
#    
#    ionLists = iglearner.generateIonGroup_Int(atypesubspects, orginalpois, spectMaxInt)
#    
#    
#    file_name4 = "SubSpectrumData/"+"IonGroups_AtypeInt"
#    wfile = write_file()
##    wfile.writeSpectMaxInt(spectMaxInt)
#    
##    iglearner.paintMaxInt(spectMaxInt)
##    wfile.writeIonPoi(orginalpois)
#    wfile.writeIonGroups(ionLists, file_name4)
##    
###    ionLists = iglearner.generateIonGroup(subspects, orginalpois)
##
################################ Test 5#####################################
    names=['y10+ ']
    for name in names:
        filename1 = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_"+name+"_intensity_20160120"
        subparser = SubSpectrumGenerator()
        atypesubspects = list(subparser.generateSubSpecfile(filename1,'intensity'))
      
        filename2 = "SubSpectrumData/"+"IonPostion"   
        iglearner = IonGroupLearner()
        orginalpois = iglearner.generateIonPoitionFile(filename2) # directly get original pois from file
        
    #    print orginalpois
    #    file_name = 'data/new_CHPP_LM3_RP3_2.mgf' 
    #    parser = SpectrumParser()
    #    specs = list(parser.readSpectrum(file_name)) # orignal datas file
    #    
    #    spectMaxInt = iglearner.generateMaxIntentity(specs)
        
        file_name3 = "SubSpectrumData/"+"SpectMaxInt"
        spectMaxInt = iglearner.generateMaxIntentityFile(file_name3)
        
        ionLists = iglearner.generateIonGroup_Int(atypesubspects, orginalpois, spectMaxInt)
        
        
        file_name4 = "SubSpectrumData/"+"IonGroups_"+name+"Int"
        wfile = write_file()
    #    wfile.writeSpectMaxInt(spectMaxInt)
        
    #    iglearner.paintMaxInt(spectMaxInt)
    #    wfile.writeIonPoi(orginalpois)
        wfile.writeIonGroups(ionLists, file_name4)
    #    
    ##    ionLists = iglearner.generateIonGroup(subspects, orginalpois)
################################ Test 5#####################################   
#    file_name1 = "SubSpectrumData/"+"IonGroups"
#    file_name2 = "SubSpectrumData/"+"IonGroups_Int"
#    file_name3 = "SubSpectrumData/"+"IonGroups_NoiInt"
#    file_name4 = "SubSpectrumData/"+"IonGroups_DualInt"
#    file_name5 = "SubSpectrumData/"+"IonGroups_AtypeInt"
#    file_names = []
#    file_names.append(file_name1)
#    file_names.append(file_name2)
#    file_names.append(file_name3)
#    file_names.append(file_name4)
#    file_names.append(file_name5)
#    
#    for file_name in file_names:       
#        iglearner = IonGroupLearner()
#        ionLists = iglearner.generateIonGroupFile(file_name)
#        ionLists2 = iglearner.calcuIonGroup_axis(ionLists)
#        print ionLists2
#    end = time.clock()
#    print 'time consuming %s seconds.' % (end-start)


        