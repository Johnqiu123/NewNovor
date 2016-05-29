# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:00:33 2015

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
from write_file import write_file

aa_table = ProteinWeightDict()
class SubSpectrumProcessor(object):
    
    def generateRandSample(self, subspects, samplenum):
        """
        generate random samples from subspects
        
        Args:
          -subspects: a set of subspectrum
          -samplenum: the number of samples
        
        Return:
          -randsubspects: a set of random subspects
        """
        rannum = random.sample(xrange(len(subspects)),samplenum)
        randsubspects = []
        for i in rannum:
            randsubspects.append(subspects[i])
        return randsubspects
    
    def calculateBins(self, subspects):
        """
        calculate bins from subspects
        
        Args:
         -subspects:  a set of subspectrum
        
        Return:
         -allNtermbins: ndarray
         -allCtermbins: ndarray
         -allbins: ndarray, merge allNtermbins and allCtermbins
         -num： the num of subspects
        """
        binlen = float(subspects[0].getBinLength())
        binrange = int(subspects[0].getBinArea())
        
        allNtermbins = np.zeros(int(binrange*2/binlen))
        allCtermbins = np.zeros(int(binrange*2/binlen))
        
        
        
        num = len(subspects)
        
        for subspec in subspects:
#            print np.array(subspec.getNtermBins())
            allNtermbins += np.array(subspec.getNtermBins())
            allCtermbins += np.array(subspec.getCtermBins())
        
        # merge two bins
        allbins = np.hstack((allNtermbins,allCtermbins))
        
        return allNtermbins,allCtermbins,allbins,num
    
    def generateSubTable(self, subNum, noiNum,subPeak, noiPeak):
        """
          generate a SubTable
          
          Args:
           -subNum:         the number of subspectrum
           -noiNum          the number of noise spectrum
           -subPeak:   the number of peaks which fall in the subbin
           -noiPeak:   the number of peaks which fall in the noibin
          
          Return:
           -table    
        """
        noiNoPeak = noiNum - noiPeak # noise no peak
        subNoPeak = subNum - subPeak # subspectrum no peak
        table = np.array([[noiNoPeak,subNoPeak],[noiPeak,subPeak]])
        return table
     
    def generateIonAminoPairsTable(self, subspects, pois):
        """
          generate a IonAminoPairsTable
          
          Args:
           -subspects:      a set of subspectrum
           -pois         a set of poistion
          
          Return:
           -poitables    
        """   
        ionaptables = {}
        aminosDict = {}
        for poi in pois:
            allAminosets= itertools.product(aa_table.keys(),aa_table.keys())
            aminosD = aminosDict.fromkeys(allAminosets,(0,0))
#            print len(aminosD)
            for subspect in  subspects:
                LR = (subspect.getLA(), subspect.getRA())
                num = subspect.getNumberofNBins()  # get the number of bins
                if aminosD.has_key(LR):
                    if poi <= num:  # N-term
                       flag =  subspect.getNtermBins()[poi]
                       temp = aminosD[LR]
#                       print("Nflag="+str(flag))
                       if flag == 1:
                           aminosD[LR] = (temp[0],temp[1]+1)
                       else:
                           aminosD[LR] = (temp[0]+1,temp[1])
                    else:  # c-term
                        temppoi = poi - num - 1
                        flag =  subspect.getCtermBins()[temppoi]
                        temp = aminosD[LR]
#                        print("Cflag="+str(flag))
                        if flag == 1:
                           aminosD[LR] = (temp[0],temp[1]+1)
                        else:
                           aminosD[LR] = (temp[0]+1,temp[1])  
#            print len(aminosD)
            poidf = pd.DataFrame(aminosD)
            ionaptables[poi] = poidf
        return ionaptables
    
    def generateIonTable(self, subspects, pois):
        """
          generate a IonTable
          
          Args:
           -subspects:      a set of subspectrum
           -pois         a set of poistion
          
          Return:
           -iontables    
        """   
        combpois = list(itertools.combinations(pois,2))
        iontables = {}
        for poix,poiy in combpois:
            iontable = {0:(0,0),1:(0,0)}
            for subspect in subspects:
                num = subspect.getNumberofNBins()  # get the number of bins
                if poix <= num: # N-term
                    if poiy <= num: # N-term
                        flagx = subspect.getNtermBins()[poix]
                        flagy = subspect.getNtermBins()[poiy]
                        iontable = self.updateIonTable(iontable, flagx, flagy)
                    else:
                        flagx = subspect.getNtermBins()[poix]
                        flagy = subspect.getCtermBins()[poiy-num-1]
                        iontable = self.updateIonTable(iontable, flagx, flagy)
                else:  # C-term
                    if poiy <= num:  # C-term
                        flagx = subspect.getCtermBins()[poix-num-1]
                        flagy = subspect.getNtermBins()[poiy]
                        iontable = self.updateIonTable(iontable, flagx, flagy)
                    else:   # C-term
                        flagx = subspect.getCtermBins()[poix-num-1]
                        flagy = subspect.getCtermBins()[poiy-num-1]
                        iontable = self.updateIonTable(iontable, flagx, flagy)
            poiDF = pd.DataFrame(iontable)       
            iontables[(poix,poiy)] = poiDF
        return iontables
        
    def updateIonTable(self, iontable, flagx, flagy):
        if flagy: # (0,1),(1,1)
            if flagx: # (1,1)
                temp = iontable[1]
                iontable[1] = (temp[0],temp[1]+1)
            else: # (0,1)
                temp = iontable[1]
                iontable[1] = (temp[0]+1,temp[1])
        else:  # (0,0),(1,0)
            if flagx: # (1,0)
                temp = iontable[0]
                iontable[0] = (temp[0],temp[1]+1)
            else:  # (0,0)
                temp = iontable[0]
                iontable[0] = (temp[0]+1,temp[1])
            
        return iontable
    
    def generateIonPepbondpoiTable(self, subspects, pois, computeflag):
        """
          generate a IonPepbondpoiTable
          
          Args:
           -subspects:      a set of subspectrum
           -pois         a set of poistion
          
          Return:
           -ionpbptables    
        """           
        ionpbptables = {}
        for poi in pois:
            ionpbptable = {'L':(0,0),'M':(0,0),'R':(0,0)}
            for subspect in subspects:
                subpoi = subspect.getPosition()
                sublen = subspect.getLength()
                flag = 1.0/3.0
                if computeflag == 'length':
                    temp = float(subpoi)/float(sublen)
                elif computeflag == 'mass':
                    pep = subspect.getAnnotation()
                    m = float(pep.getMass())
                    mpi = float(pep.getMassList()[subpoi])
                    temp = mpi / m
                num = subspect.getNumberofNBins()  # get the number of bins
                if temp <= flag: # no more than 1/3
                    ionpbptable = self.updateionpbptable(ionpbptable, subspect, 'L', poi, num)
                elif temp <= flag * 2: # no more than 2/3
                    ionpbptable = self.updateionpbptable(ionpbptable, subspect, 'M', poi, num)
                else:
                    ionpbptable = self.updateionpbptable(ionpbptable, subspect, 'R', poi, num)
            ionpbpDF = pd.DataFrame(ionpbptable)       
            ionpbptables[poi] = ionpbpDF
        return ionpbptables

    def updateionpbptable(self, ionpbptable, subspect, poiflag, poi, num):
        if poi <= num: 
            value = subspect.getNtermBins()[poi]
            tabletemp = ionpbptable[poiflag]
            if value:
                ionpbptable[poiflag] = (tabletemp[0],tabletemp[1]+1)
            else:
                ionpbptable[poiflag] = (tabletemp[0]+1,tabletemp[1])
        else:
            value = subspect.getCtermBins()[poi-num]
            tabletemp = ionpbptable[poiflag]
            if value:
                ionpbptable[poiflag] = (tabletemp[0],tabletemp[1]+1)
            else:
                ionpbptable[poiflag] = (tabletemp[0]+1,tabletemp[1])   
        return ionpbptable
        

    def paintSubSpects(self, allbins, binlen=0.1,binrange=50):
        """
           paint SubSpectrum
           
           Args:
             -binlen:  the length of bin
             -binrange:  the range of bin
             -allbins: ndarray
        """
        # xaix
        xaix = np.arange(-binrange,binrange,binlen)
#        xaix = np.arange(0,400,binlen)
        # paint
        pt = Paint_File()
        pt.paint(xaix,allbins)
    
    def paintChiValues(self, chivalues):
        pt = Paint_File()
        xais = [i for i in range(len(chivalues))]
#        yais = [value[1] for value in chivalues]
        yais = chivalues
        pt.paint(xais,yais)
    
    def paintionpbpTable(self, ionpbptables):
        pt = Paint_File()
        xais = []
        yaisL = []
        yaisM = []
        yaisR = []
#        ionpbptables = dict(sorted(ionpbptables.iteritems(),key=lambda x:x[0])) # 按第0列排序
        for key in ionpbptables:
            xais.append(key)
            yaisL.append(ionpbptables[key].iloc[1,0])
            yaisM.append(ionpbptables[key].iloc[1,1])
            yaisR.append(ionpbptables[key].iloc[1,2])    
        print xais,yaisL,yaisM,yaisR
        pt.paintmultiBars(xais,yaisL,yaisM,yaisR)
    
    def sortChiValues(self, chiValues):
        """
           sort ChiValues
           
           Args:
             -chiValues:  the chiValues of all positions
          
           Return:
             -poiChiValues the sorted chiVlaues, turple with postion
             -chiValues: the sorted chiVlaues, no turple
        """
        pois = [k for k in range(0, len(chiValues)+1)]
        poiChiValues = zip(chiValues,pois)
        poiChiValues.sort(key=lambda x:x[0],reverse=True) # 按第0列排序
        chiValues.sort(reverse=True)
        return poiChiValues,chiValues
        
        
    def ChiSquared_TypeandBreakPoint(self, subNum, noiNum, allSubbins, allNoibins):
        """
          compute the chi-squared statistic for Type and BreakPoint 
          Args:
            -subNum:    the number of subspectrum
            -allSubbins: a ndarray of subbins
            -noiNum:    the number of noise spectrum
            -allNoisebins: a ndarray of noisebins
          
          Return:
            -chiValues :  a list of chisquared value
        """
        chiValues = []    
#        print type(allSubbins),type(allNoibins)
        chi = ChiSquared()
        for subbin,noibin in zip(allSubbins,allNoibins):
            table = self.generateSubTable(subNum,noiNum,subbin,noibin)
#            chiValue = chi.ComplexChiSquared(table)
            chiValue = chi.OrgainChiSquard(table)
            chiValues.append(chiValue)
        return chiValues

    def ChiSquared_TypeandAminoPairs(self, subspects, orginalpois):
        """
          compute the chi-squared statistic for Type and AminoPairs
          Args:
            -subspects:    a set of subspectrum
            -orginalpois: a ndarray of subbins
          
          Return:
            -chiValues :  a list of chisquared value
        """
        choicepois = [1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1] # by people
        orgpois = [choicepois[i] and orginalpois[i] for i in range(len(orginalpois))]
        orgpois = filter((lambda x: x>0), orgpois)
        ionaptables = self.generateIonAminoPairsTable(subspects,orgpois)
#        print ionaptables
        
        chiValues = []  
        chi = ChiSquared()
        for key in ionaptables:
#            print key         
            table = np.array([ionaptables[key].iloc[0],ionaptables[key].iloc[1]])
            chiValue = chi.OrgainChiSquard(table)
#            print table
            chiValues.append(chiValue)
        return chiValues

    def ChiSquared_TypeandType(self, subspects, orginalpois):
        """
          compute the chi-squared statistic for Type and Type
          Args:
            -subspects:    a set of subspectrum
            -orginalpois: a ndarray of subbins
          
          Return:
            -chiValues :  a list of chisquared value
        """
        choicepois = [1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1] # by people
        orgpois = [choicepois[i] and orginalpois[i] for i in range(len(orginalpois))]
        orgpois = filter((lambda x: x>0), orgpois)
        iontables = self.generateIonTable(subspects,orgpois)
        
        chiValues = []  
        chi = ChiSquared()
        for key in iontables:
            table = np.array(iontables[key].T)
            chiValue = chi.OrgainChiSquard(table)
            chiValues.append(chiValue)
        return chiValues

    def ChiSquared_TypeandPepbondPoi(self, subspects, orginalpois, computeflag='length'):
        """
          compute the chi-squared statistic for Type and Pepbond‘s poisition
          Args:
            -subspects:    a set of subspectrum
            -orginalpois: a ndarray of subbins
            -computeflag: choose compute mode, include length and mass
          
          Return:
            -chiValues :  a list of chisquared value
        """
        choicepois = [1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1] # by people
        orgpois = [choicepois[i] and orginalpois[i] for i in range(len(orginalpois))]
        orgpois = filter((lambda x: x>0), orgpois)
        ionpbptables = self.generateIonPepbondpoiTable(subspects,orgpois, computeflag)

        
        chiValues = []  
        chi = ChiSquared()
        for key in ionpbptables:
#            print key
            table = np.array(ionpbptables[key].T)
            chiValue = chi.OrgainChiSquard(table)
            chiValues.append(chiValue)
        return chiValues

if __name__=='__main__':

    start = time.clock()
################################ Test 1#####################################
    filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_20151208"
    filename2 = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_Noise_20151208"
    subparser = SubSpectrumGenerator()
    subspects = list(subparser.generateSubSpecfile(filename))
    noisubspects = list(subparser.generateNoiSubfile(filename2)) # noise
#    
    subprocessor = SubSpectrumProcessor()
    allNtermbins,allCtermbins,allSubbins,subNum = subprocessor.calculateBins(subspects)
#    allNOiNtermbins,allNoiCtermbins,allNoibins,noiNum = subprocessor.calculateBins(noisubspects)
##     #n-term
#    NchiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allNtermbins,allNOiNtermbins)
##     #c-term
#    CchiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allCtermbins,allNoiCtermbins)
##     #all
#    chiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allSubbins,allNoibins)
#    
##    print chiValues
##    
#    subprocessor.paintSubSpects(allNtermbins)
#    subprocessor.paintSubSpects(allCtermbins)   
#    subprocessor.paintSubSpects(allSubbins)
#    subprocessor.paintSubSpects(NchiValues)
#    subprocessor.paintSubSpects(CchiValues)
#    subprocessor.paintSubSpects(chiValues)
    
#
#    print allNtermbins,allCtermbins,allbins
#    print noinum
#
#    table = subprocessor.generateSubTable(112,4,78)
#    chi = ChiSquared()
#    result = chi.EasyChiSquared(table)
#    result2 = chi.ComplexChiSquared(table)
#    print result
#    print result2

################################ Test 2#####################################
#    filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2"
#    filename2 = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_Noise"
#    subparser = SubSpectrumGenerator()
#    subspects = subparser.generateSubSpecfile(filename)
#    noisubspects = subparser.generateNoiSubfile(filename2) # noise
#    
#    subprocessor = SubSpectrumProcessor()
#    for i in [100,200,300,500]:
#        randsubspects = subprocessor.generateRandSample(subspects,i)
#        randnoisubspects = subprocessor.generateRandSample(noisubspects,i)
#        allNtermbins,allCtermbins,allSubbins,subNum = subprocessor.calculateBins(randsubspects)
#        allNoiNtermbins,allNoiCtermbins,allNoibins,noiNum = subprocessor.calculateBins(randnoisubspects)
#        chiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allNtermbins,allNoiNtermbins)
##        schiValues2 = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allCtermbins,allNoiCtermbins)
#        subprocessor.paintSubSpects(chiValues)
##    subprocessor.paintSubSpects(chiValues2)

################################ Test 3#####################################
#    start = time.clock()
#    filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_20151208"
#    filename2 ="SubSpectrumData/"+"new_CHPP_LM3_RP3_2_Noise_20151208"
#    subparser = SubSpectrumGenerator()
#    subspects = subparser.generateSubSpecfile(filename)
#    noisubspects = subparser.generateNoiSubfile(filename2) # noise
##
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
#    subprocessor.paintSubSpects(poichiV)
    
   
#    orginalpois = [poiChiValues[i][1] for i in range(len(poiChiValues))][0:21] # get top 10 chivalues
#    randsubspects = subprocessor.generateRandSample(subspects,10)
#    subsps = subspects# get 100 subspects
#    ionapChiValues = subprocessor.generateIonAminoPairsTable(subsps,orginalpois)
#    ionapChiValues = subprocessor.ChiSquared_TypeandAminoPairs(subsps,orginalpois)
#    subprocessor.paintChiValues(ionapChiValues)
    
#    choicepois = [1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1] # by people
#    orgpois = [choicepois[i] and orginalpois[i] for i in range(len(orginalpois))]
#    orgpois = filter((lambda x: x>0), orgpois)
#    iontables = subprocessor.generateIonTable(subsps,orgpois)
#    ionpbptables_len = subprocessor.generateIonPepbondpoiTable(subsps,orgpois, 'length')
#    ionpbptables_mass = subprocessor.generateIonPepbondpoiTable(subsps,orgpois, 'mass')
#    
#    ionpbpchiValues_len = subprocessor.ChiSquared_TypeandPepbondPoi(subsps,orginalpois, 'length')
#    ionpbpchiValues_mass = subprocessor.ChiSquared_TypeandPepbondPoi(subsps,orginalpois, 'mass')
    
#    subprocessor.paintionpbpTable(ionpbptables_len)
#    subprocessor.paintionpbpTable(ionpbptables_mass)
    
#    subprocessor.paintChiValues(ionpbpchiValues_len)
#    subprocessor.paintChiValues(ionpbpchiValues_mass)
#    ionchiValues = subprocessor.ChiSquared_TypeandType(subsps,orginalpois)
#    subprocessor.paintChiValues(ionchiValues)
#    
#    ionkeyValues = {}
#    for key,value in zip(iontables.keys(),ionchiValues):
#        ionkeyValues[key] = value
#    
#    ionkeyValues = sorted(ionkeyValues.iteritems(),key=lambda x:x[1],reverse=True) # 按第0列排序
#    
#    subprocessor.paintChiValues(ionkeyValues)
#    chiValues = []  
#    chi = ChiSquared()
#    for iontable in iontables:
#        print type(iontable)
#        table = np.array(iontables[iontable].T)
#        chiValue = chi.OrgainChiSquard(table)
#        chiValues.append(chiValue)
    
#    for sub in subsps:
#        print sub.getPosition()
#    print poitables[orgpois[0]]
    
#    end = time.clock()
#    print 'time consuming %s seconds.' % (end-start)
    
    
    # paint 
#    paint = Paint_File()
#    paint.paintploylines(pois,poichiV)

#    c = itertools.product(aa_table.keys(),aa_table.keys())
#    dict = {}
#    dict = dict.fromkeys(c,(0,0))
#    a = ('S', 'W')
#    print dict
#    if dict.has_key(a):
#        ss = dict[a]
#        dict[a] = (ss[0],ss[1]+1)
##        print dict[a][0]
#    poidf = pd.DataFrame(dict)
#    print poidf
#    print dict
    
#    dict = {}
#    for i in range(5):
#        c = itertools.product(aa_table.keys(),aa_table.keys())
#        dict = dict.fromkeys(c,(0,0))
#        print dict
#        if dict.has_key(a):
#            ss = dict[a]
#            dict[a] = (ss[0],ss[1]+1)
#        poidf = pd.DataFrame(dict)
#        print poidf

################################ Test 4#####################################
#    start = time.clock()
#    filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_20151208"
#    subparser = SubSpectrumGenerator()
#    subspects = subparser.generateSubSpecfile(filename)
#
#    subprocessor = SubSpectrumProcessor()
#    allNtermbins,allCtermbins,allSubbins,subNum = subprocessor.calculateBins(subspects)
#    
##    print allSubbins
#    
#    subprocessor.paintSubSpects(allSubbins)
#
#    end = time.clock()
#    print 'time consuming %s seconds.' % (end-start)

################################ Test 5#####################################
#    fw = write_file()
#    subparser = SubSpectrumGenerator()
#    filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_intensity_20160120"
#    filename2 = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_Noise_intensity_20160120"
#
#    subspect = subparser.generateSubSpecfile(filename,'intensity')
#    noisubspect = subparser.generateNoiSubfile(filename2,'intensity') # noise
#    
#    subspects = []
#    noisubspects = []
#    for i in range(10000):
#        subspects.append(subspect.next())
#        noisubspects.append(noisubspect.next())
#    
#    subprocessor = SubSpectrumProcessor()
#    allNtermbins,allCtermbins,allSubbins,subNum = subprocessor.calculateBins(subspects)
#    allNOiNtermbins,allNoiCtermbins,allNoibins,noiNum = subprocessor.calculateBins(noisubspects)
#    
#    print type(allSubbins)
#    
#    filename_bin = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_intensity_subBins"
#    filename_noibin = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_intensity_noisubBins"
#    
#    fw.writeCalculateBins(filename_bin,allNbins=allNtermbins,allCbins=allCtermbins,\
#                          allbins = allSubbins, allNum = subNum)
#    fw.writeCalculateBins(filename_noibin,allNbins=allNOiNtermbins,allCbins=allNoiCtermbins,\
#                          allbins = allNoibins, allNum = noiNum)
#    allNtermbins,allCtermbins,allSubbins,subNum = subparser.generateCalBinsfile(filename_bin)
#    allNOiNtermbins,allNoiCtermbins,allNoibins,noiNum = subparser.generateCalBinsfile(filename_noibin)
##     #n-term
#    NchiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allNtermbins,allNOiNtermbins)
##     #c-term
#    CchiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allCtermbins,allNoiCtermbins)
##     #all
#    chiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allSubbins,allNoibins)
    
#    print chiValues
##    
#    subprocessor.paintSubSpects(allNtermbins)
#    subprocessor.paintSubSpects(allCtermbins)   
#    subprocessor.paintSubSpects(allSubbins)
#    subprocessor.paintSubSpects(NchiValues)
#    subprocessor.paintSubSpects(CchiValues)
#    subprocessor.paintSubSpects(chiValues)

################################ Test 6#####################################
    fw = write_file()
    subparser = SubSpectrumGenerator()
    filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_intensity_20160120"
    filename2 = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_Noise_intensity_20160120"

    subspect = subparser.generateSubSpecfile(filename,'intensity')
    noisubspect = subparser.generateNoiSubfile(filename2,'intensity') # noise
    
#    subspects = list(subparser.generateSubSpecfile(filename,'intensity'))
#    noisubspects = list(subparser.generateNoiSubfile(filename2,'intensity')) # noise
#    subspects = []
#    noisubspects = []
#    for i in range(10000):
#        subspects.append(subspect.next())
#        noisubspects.append(noisubspect.next())
##    
#    subprocessor = SubSpectrumProcessor()
#    allNtermbins,allCtermbins,allSubbins,subNum = subprocessor.calculateBins(subspects)
#    allNOiNtermbins,allNoiCtermbins,allNoibins,noiNum = subprocessor.calculateBins(noisubspects)
#    
#    print type(allSubbins)
#    
#    filename_bin = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_intensity_subBins_cp"
#    filename_noibin = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_intensity_noisubBins_cp"
    
#    fw.writeCalculateBins_cp(filename_bin, allNtermbins,allCtermbins,allSubbins,subNum)
#    fw.writeCalculateBins_cp(filename_noibin,allNOiNtermbins,allNoiCtermbins,allNoibins,noiNum)
#    allNtermbins,allCtermbins,allSubbins,subNum = subparser.generateCalBinsfile_cp(filename_bin)
#    allNOiNtermbins,allNoiCtermbins,allNoibins,noiNum = subparser.generateCalBinsfile_cp(filename_noibin)
##     #n-term
#    NchiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allNtermbins,allNOiNtermbins)
##     #c-term
#    CchiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allCtermbins,allNoiCtermbins)
##     #all
#    chiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allSubbins,allNoibins)
    
#    print chiValues
##    
#    subprocessor.paintSubSpects(allNtermbins)
#    subprocessor.paintSubSpects(allCtermbins)   
#    subprocessor.paintSubSpects(allSubbins)
#    subprocessor.paintSubSpects(NchiValues)
#    subprocessor.paintSubSpects(CchiValues)
#    subprocessor.paintSubSpects(chiValues)

    
    end = time.clock()
    print 'time consuming %s seconds.' % (end-start)