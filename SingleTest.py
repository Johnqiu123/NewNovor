# -*- coding: utf-8 -*-
"""
Created on Fri Mar 04 09:26:16 2016

@author: Johnqiu
"""
from SpectrumParser import SpectrumParser
import SubSpectrum
import NoiseSubSpectrum
from scripts import ProteinWeightDict
import time
import random
from AminoAcid import AminoAcid
from Peptide import Peptide
from SubSpectrumProcessor import SubSpectrumProcessor
import numpy as np
"""
使用生成器的时间：181.301952486s.

直接生产图谱的时间：196.564444245s
"""
aa_table = ProteinWeightDict()
class SingleTest(object):
    def generateSubSpecfile(self,filename, flag='num'):
         '''generate a set of subSpectrums from filename'''
#         with open(pepfile) as pep_file:
#             title_peptide = {line.strip().split(' ')[0]:line.strip().split(' ')[1] \
#             for line in pep_file}     
         subspects = [] # a set of subspectrum
         with open(filename) as input_data:   
             for line in input_data:
                 line = line[1:len(line)-2].split('###')
                 subspectData = {data.split('=')[0]:data.split('=')[1] \
                     for data in line}
                 # without precursor_peak
                 title = subspectData['Title']
                 seq = subspectData['SEQ'] # peptide
                 pos = int(subspectData['POSITION'])
                 length = int(subspectData['LENGTH'])
                 leftaa = subspectData['LeftAA']
                 rightaa = subspectData['RightAA']
                 binlen = float(subspectData['BIN_LENGTH'])
                 binarea = int(subspectData['BIN_RANGE'])
                 
                 ntermbin = subspectData['NTermBIN']
                 ctermbin = subspectData['CTermBIN']
                 if flag == 'num':
                     ntermbin = [int(k) for k in ntermbin[1:len(ntermbin)-1].split(',')]
                     ctermbin = [int(k) for k in ctermbin[1:len(ctermbin)-1].split(',')]
                 elif flag == 'intensity':
                     ntermbin = [float(k) for k in ntermbin[1:len(ntermbin)-1].split(',')]
                     ctermbin = [float(k) for k in ctermbin[1:len(ctermbin)-1].split(',')]                    
                 
                 # generate peptide
                 acids = []
                 acids += [AminoAcid(s,'',aa_table[s]) for s in seq if s in aa_table]    
                 pep = Peptide(acids)
                 
                 subspect = SubSpectrum.SubSpectrum(None,title,pos,leftaa,rightaa)
                 subspect.setBinLength(binlen)
                 subspect.setBinArea(binarea)
                 subspect.setCtermBins(ctermbin)
                 subspect.setNtermBins(ntermbin)
                 subspect.setLength(length)  # new code 15/12/27
                 subspect.setAnnotation(pep)
                  
                 subspects.append(subspect)
         input_data.close()
         return subspects

    def generateSubSpecfile2(self,filename, flag='num'):
         '''generate a set of subSpectrums from filename'''
#         with open(pepfile) as pep_file:
#             title_peptide = {line.strip().split(' ')[0]:line.strip().split(' ')[1] \
#             for line in pep_file}     
         with open(filename) as input_data:   
             for line in input_data:
                 line = line[1:len(line)-2].split('###')
                 subspectData = {data.split('=')[0]:data.split('=')[1] \
                     for data in line}
                 # without precursor_peak
                 title = subspectData['Title']
                 seq = subspectData['SEQ'] # peptide
                 pos = int(subspectData['POSITION'])
                 length = int(subspectData['LENGTH'])
                 leftaa = subspectData['LeftAA']
                 rightaa = subspectData['RightAA']
                 binlen = float(subspectData['BIN_LENGTH'])
                 binarea = int(subspectData['BIN_RANGE'])
                 
                 ntermbin = subspectData['NTermBIN']
                 ctermbin = subspectData['CTermBIN']
                 if flag == 'num':
                     ntermbin = [int(k) for k in ntermbin[1:len(ntermbin)-1].split(',')]
                     ctermbin = [int(k) for k in ctermbin[1:len(ctermbin)-1].split(',')]
                 elif flag == 'intensity':
                     ntermbin = [float(k) for k in ntermbin[1:len(ntermbin)-1].split(',')]
                     ctermbin = [float(k) for k in ctermbin[1:len(ctermbin)-1].split(',')]                    
                 
                 # generate peptide
                 acids = []
                 acids += [AminoAcid(s,'',aa_table[s]) for s in seq if s in aa_table]    
                 pep = Peptide(acids)
                 
                 subspect = SubSpectrum.SubSpectrum(None,title,pos,leftaa,rightaa)
                 subspect.setBinLength(binlen)
                 subspect.setBinArea(binarea)
                 subspect.setCtermBins(ctermbin)
                 subspect.setNtermBins(ntermbin)
                 subspect.setLength(length)  # new code 15/12/27
                 subspect.setAnnotation(pep)
                 yield subspect
                 
if __name__=='__main__':
    
#    filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_20151208"
#    test = SingleTest()
#    
#    start = time.clock()
#    subspects = test.generateSubSpecfile(filename)
#
##    subspects = list(test.generateSubSpecfile2(filename))
#
#    subprocessor = SubSpectrumProcessor()
#    allNtermbins,allCtermbins,allSubbins,subNum = subprocessor.calculateBins(subspects)
#
#    end = time.clock()
#    print 'time consuming %s seconds.' % (end-start)
    
    string = '[ 1 2 3 ]'
    mystr = string[1:len(string)-1].strip().split(' ')
    print mystr
    k = np.array(map(float,mystr))
#    k = np.array(mystr, dtype=int)
    print k.shape
    s = np.array(map(int, "100110"))
    print s