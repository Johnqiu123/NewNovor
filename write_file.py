# -*- coding: utf-8 -*-
from SubSpectrumGenerator import SubSpectrumGenerator
from SpectrumParser import SpectrumParser
from SubSpectrum import SubSpectrum
import time
import numpy as np
import matplotlib.pyplot as plt 
import cPickle as cpickle
import os
'''
  Process raw data by adding their corresponding peptide annotations.
  created on Sep 22, 2015 by mht,Johnqiu'''
  
class write_file(object):
    
    def rewriteFile(self, IPFile, PLFile, OUTFile):
       ## all peptide in simple.peplist
        IPFile = "data/" + IPFile
        PLFile = "data/" + PLFile
        OUTFile = "data/" + OUTFile
        if not os.path.exists(IPFile) or not os.path.exists(PLFile):
            return False
        parser = SpectrumParser()
        specs = list(parser.readSpectrum(IPFile))
        
        with open(PLFile) as pep_file:
            title_peptide = {line.strip().split(' ')[0]:line.strip().split(' ')[1] \
            for line in pep_file}
        # rewrite data
        with open(OUTFile,'w') as output_data:
            for spec in specs:
                title = spec.getTitle()
                if title in title_peptide:
                    output_data.write('BEGIN IONS'+'\n')
                    output_data.write('TITLE='+title+'\n')
                    output_data.write('SEQ='+title_peptide[title]+'\n')
                    output_data.write('PEPMASS='+str(spec.getPrecursorPeak().getMass())+'\n')
                    output_data.write('CHARGE='+str(spec.getCharge())+'+'+'\n')
                    for peak in spec.getPeaks():
                        if peak.getIntensity() != 0:
                            output_data.write(str(peak.getMz())+' '+str(peak.getIntensity())+'\n')
                            output_data.write('END IONS'+'\n')
        return True
                          
    def writeSubSepc(self, File, binlen=0.1, arealen=50, flag = 'num'):
       ## write all subspectrum into file
        file_name = "data/" + File
        fname = File.split('.')[0]
        parser = SpectrumParser()
        specs = list(parser.readSpectrum(file_name))
        print len(specs)
        subparser = SubSpectrumGenerator()
        subspecs = list(subparser.generateSubSpectra(specs, binlen, arealen, flag))
        if flag == 'num':
            filename = "SubSpectrumData/"+ fname
        elif flag == 'intensity':
            filename = "SubSpectrumData/"+fname+"_intensity"
        allNtermbins = np.zeros(int(arealen*2/binlen))
        allCtermbins = np.zeros(int(arealen*2/binlen))
                
        with open(filename,'w')  as  data:
            for subspec in subspecs:
               seq = ''
               for acid in subspec.getAnnotation().getAcids():
                    seq = seq + acid.getResidue()
               data.write('[Title='+ subspec.getTitle() + '###'\
                          +'SEQ=' +seq+ '###'\
                          +'POSITION='+ str(subspec.getPosition())+ '###'\
                          +'LENGTH='+str(len(subspec.getAnnotation().getAcids()))+'###'\
                          +'LeftAA='+subspec.getLA()+'###'\
                          +'RightAA='+subspec.getRA()+'###'\
                          +'BIN_LENGTH='+str(subspec.getBinLength())+'###'\
                          +'BIN_RANGE='+str(subspec.getBinArea())+'###'\
                          +'NTermBIN='+str(subspec.getNtermBins())+'###'\
                          +'CTermBIN='+str(subspec.getCtermBins())+']'+'\n')
            data.close() # close file

    def writeSpecialSubSepc(self, binlen=0.1, arealen=50, flag = 'num', stype='dual'):
       ## write all subspectrum into file
        file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
        parser = SpectrumParser()
        specs = list(parser.readSpectrum(file_name))
        subparser = SubSpectrumGenerator()
        subspecs = list(subparser.generateSpecialSubSpectra(specs, binlen, arealen, flag, stype))
        if flag == 'num':
            filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_"+stype+"_20151208"
        elif flag == 'intensity':
            filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_"+stype+"_intensity_20160120"
        allNtermbins = np.zeros(int(arealen*2/binlen))
        allCtermbins = np.zeros(int(arealen*2/binlen))
                
        with open(filename,'w')  as  data:
            for subspec in subspecs:
               seq = ''
               for acid in subspec.getAnnotation().getAcids():
                    seq = seq + acid.getResidue()
               data.write('[Title='+ subspec.getTitle() + '###'\
                          +'SEQ=' +seq+ '###'\
                          +'POSITION='+ str(subspec.getPosition())+ '###'\
                          +'LENGTH='+str(len(subspec.getAnnotation().getAcids()))+'###'\
                          +'LeftAA='+subspec.getLA()+'###'\
                          +'RightAA='+subspec.getRA()+'###'\
                          +'BIN_LENGTH='+str(subspec.getBinLength())+'###'\
                          +'BIN_RANGE='+str(subspec.getBinArea())+'###'\
                          +'NTermBIN='+str(subspec.getNtermBins())+'###'\
                          +'CTermBIN='+str(subspec.getCtermBins())+']'+'\n')
            data.close() # close file
            
    def writeNoiseSubSepc(self, File, num=3, binlen=0.1, arealen=50, flag = 'num'):
        file_name = "data/"+ File
        fname = File.split('.')[0]
        parser = SpectrumParser()
        specs = list(parser.readSpectrum(file_name))
        noisesubparser = SubSpectrumGenerator()
        noisesubspecs = list(noisesubparser.generateNoiseSubSpectra(specs, num, binlen, arealen, flag))
        print(len(noisesubspecs))
        if flag == 'num':
            filename = "SubSpectrumData/"+ fname + "_Noise"
        elif flag == 'intensity':
            filename = "SubSpectrumData/"+ fname + "_Noise_intensity"
#        allbins = np.zeros(int(arealen*2/binlen))
        with open(filename,'w')  as  data:
          # add total value        
          for noisesubspec in noisesubspecs:
              data.write('[Title='+ noisesubspec.getTitle() + '###'\
                        +'PositionMZ='+ str(noisesubspec.getPositionmz())+ '###'\
                        +'ParentMass='+ str(noisesubspec.getAnnotation().getParentMass())+ '###'\
                        +'NTermBIN='+str(noisesubspec.getNtermBins())+'###'\
                        +'CTermBIN='+str(noisesubspec.getCtermBins())+']'+'\n')
#                    allbins += np.array(subspec.getBins())
#                data.write('[allbins='+str(allbins)+']'+'\n')
          data.close()
    
    def writeIonPoi(self, ionpois, filename):
        filename = "SubSpectrumData/"+filename+"_IonPostion"
        with open(filename, 'w') as data:
            for ionpoi in ionpois:
                data.write(str(ionpoi) + '\n')
            data.close()
    
    def writeIonGroups(self, iongroups, filename):
        with open(filename, 'w') as data:
            for iongroup in iongroups:
                data.write(str(iongroup) + '\n')
            data.close()
    
    def writeCalculateBins(self, filename, **args):
        # write calculated bins into file
        dictname = ['allNbins','allCbins','allbins','allNum']
#        print args['allbins']
        with open(filename,'w') as data:
            data.write('[')
            for name in dictname:
                data.write(name+'='+str(args[name]) + '###')
            data.write(']')

    def writeCalculateBins_cp(self, filename, *args):
        # write calculated bins into file
        with open(filename,'w') as fw:
            cpickle.dump(args, fw)
    
    def writeSpectMaxInt(self, spectMaxInt):
        filename = "SubSpectrumData/"+"SpectMaxInt"
        with open(filename, 'w') as fw:
            cpickle.dump(spectMaxInt, fw)
        

if __name__ == '__main__':
    fw = write_file()
    start = time.clock()
#    fw.writeSubSepc()
#    fw.writeSubSepc(0.1,50,'intensity')
#    fw.writeNoiseSubSepc(8,0.1,50,'intensity')
#    fw.writeNoiseSubSepc(8)
#    fw.writeDualSubSepc(0.1, 50, 'intensity')
    fw.writeSpecialSubSepc(0.1, 50, 'intensity', 'y10+')
    end = time.clock()
    print 'time consuming %s seconds.' % (end-start)

