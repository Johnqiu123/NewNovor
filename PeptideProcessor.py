# -*- coding: utf-8 -*-
"""
Created on Tue Dec 08 21:34:15 2015

@author: Johnqiu
"""
from scripts import ProteinWeightDict
import itertools
from SpectrumParser import SpectrumParser
import pandas as pd
from Paint_File import Paint_File 
import math

aa_table = ProteinWeightDict()
class PeptideProcessor(object):
    
    def generatePepbondTable(self, spects):
        """
          generate a PepbondTable(without the repeat of amino acid)
          
          Args:
           -spects:      a set of spectrum
          
          Return:
           -pepbonddf
        """   
        aminosDict = {}
        allAminosets = itertools.combinations(aa_table,2)
        aminosD = aminosDict.fromkeys(allAminosets,0)
#        print aminosD
        for spect in spects:
            pep =  [acid.getResidue() for acid in spect.getAnnotation().getAcids()]
            pepbonds = [(pep[i],pep[i+1]) for i in range(len(pep)-1)]
            for pepbond in pepbonds:
#                print pepbond
                temp = pepbond
                reversepepb = (temp[1],temp[0])
                if aminosD.has_key(pepbond):
                    aminosD[pepbond] += 1
                elif aminosD.has_key(reversepepb):
                    aminosD[reversepepb] += 1
        aminosD = sorted(aminosD.iteritems(),key=lambda x:x[1],reverse=True) # 按第1列排序,返回list 
        pepbonddf = pd.DataFrame(aminosD)
        return pepbonddf
    
    def generateAllpepbondTable(self, spects):
        """
          generate a all PepbondTable
          
          Args:
           -spects:      a set of spectrum
          
          Return:
           -allpepbonddf
        """   
        aminosDict = {}
        allAminosets= itertools.product(aa_table.keys(),aa_table.keys())
        aminosD = aminosDict.fromkeys(allAminosets,0)
#        print aminosD
        for spect in spects:
            pep =  [acid.getResidue() for acid in spect.getAnnotation().getAcids()]
            pepbonds = [(pep[i],pep[i+1]) for i in range(len(pep)-1)]
            for pepbond in pepbonds:
                if aminosD.has_key(pepbond):
                    aminosD[pepbond] += 1
        aminosD = dict(sorted(aminosD.iteritems(),key=lambda x:x[1],reverse=True))# 按第1列排序
        allpepbonddf = pd.DataFrame([aminosD])
        return allpepbonddf
    
    def generatePepLenTable(self, spects):
        """
          generate a PepLenTable
          
          Args:
           -spects:      a set of spectrum
          
          Return:
           -pepLendf
        """   
        lenDict = {}
        for spect in spects:
            length = len(spect.getAnnotation().getAcids())
            if lenDict.has_key(length):
                lenDict[length] += 1
            else:
                lenDict[length] = 1
        pepLendf = pd.DataFrame([lenDict])
        return pepLendf

    def generateIonPoitionFile(self, filename):
        ionpois = []
        with open(filename) as input_data:
            for line in input_data:
                ionpois.append(int(line))
        return ionpois
    
    def paintPeplen(self, pepLendf):
        # paint
        pt = Paint_File()
        xais = pepLendf.columns
        yais = pepLendf.ix[0]
        print xais,yais
        print pepLendf.iloc[:,0]
        pt.paint(xais,yais)
    
    def paintPepbond(self, pepbonddf):
        # paint
        pt = Paint_File()
        xais = [i for i in range(len(pepbonddf[0]))]
        yais = [math.log(j) for j in pepbonddf[1] ]
        pt.paintploylines(xais,yais)
        
    def paintAllPepbond(self, allpepbonddf):
        allAminosets =list(itertools.combinations(aa_table,2))
        # paint       
        pt = Paint_File()
        xais = [i for i in range(len(allAminosets))]
        yais1 = []
        yais2 = []
        for amino in allAminosets:
            temp = amino
            revamino = (temp[1],temp[0])
            yais1.append(allpepbonddf[amino])
            yais2.append(-allpepbonddf[revamino])
        print yais1
        
        pt.paintmultiploylines(xais,yais1,yais2) 

        
#            print pepbond
if __name__=='__main__':
    file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
    parser = SpectrumParser()
    spects = list(parser.readSpectrum(file_name))
    peppro = PeptideProcessor()
#    pepbonddf = peppro.generatePepbondTable(spects)
##    print len(pepdf[0])
#    peppro.paintPepbond(pepbonddf)
#    
#    allpepbonddf = peppro.generateAllpepbondTable(spects)
    pepLendf = peppro.generatePepLenTable(spects)
#    print allpepbonddf[0][('E', 'E')]
#    peppro.paintAllPepbond(allpepbonddf)
    peppro.paintPeplen(pepLendf)
#    
#    allAminosets = itertools.combinations(aa_table,2)
#    yais1 = []
#    yais2 = []
#    for amino in allAminosets:
#        print amino
#        temp = amino
#        revamino = (temp[1],temp[0])
#        yais1.append(allpepbonddf[amino])
#        yais2.append(-allpepbonddf[revamino])
#    print yais1,yais2
    
#    allAminosets = itertools.combinations(aa_table,2)
#    for amino in allAminosets:
#        print amino,
#    print len(list(allAminosets))
    
###################################################Test1#################################
#    dataDict = {'nation':['Japan','S.Korea','China'],\
#           'capital':['Tokyo','Seoul','Beijing'],\
#           'GDP':[4900,1300,9100]}
#    DF2 = pd.DataFrame(dataDict)
#    print DF2['nation']

    
#    s = {('A','L'):1,('D','G'):0}
#    s1 = {'A':1,'B':2}
#    print(type(s))
#    df = pd.DataFrame(s)
#    k = df[('A','L')]
#    print k+1
#    a = ('A','L')
#    if s.has_key(a) or s.has_key(a[::-1]):
#        s[a] += 1
#    else:
#        print False
#        
#    print s
        
