from SpectrumParser import SpectrumParser
import SubSpectrum
import NoiseSubSpectrum
from scripts import ProteinWeightDict
import time
import random
from AminoAcid import AminoAcid
from Peptide import Peptide
import numpy as np
import cPickle as cpickle
''' parse spectrum file with mgf format.
   created on Sep 22, 2015 by Johnqiu.
'''
aa_table = ProteinWeightDict()
class SubSpectrumGenerator(object):
    
    def generateSubSpectra(self,spectra, bin_len=0.1, bin_area=50, flag = 'num'):
        '''generate a set of subSpectrums 
           created on Sep 22, 2015 by Johnqiu.'''
        for spectrum in spectra:
            pep = spectrum.getAnnotation() 
            title = spectrum.getTitle()
            masses = pep.getMassList()
            nterm_bins,cterm_bins = [],[]
            for i in range(len(masses)-1):
                nmass,cmass = 0,0
                nmass = masses[i]
                cmass =pep.getMass()-nmass
                LA = pep.getAcids()[i-1].getResidue()
                RA = pep.getAcids()[i].getResidue()
                nterm_bins = self.generateSubSpectrum(spectrum,nmass,bin_len,bin_area, flag) 
                cterm_bins = self.generateSubSpectrum(spectrum,cmass,bin_len,bin_area, flag)
                subspec = SubSpectrum.SubSpectrum(spectrum.getPrecursorPeak(),title,i+1,LA,RA)           
                subspec.setAnnotation(pep)
                subspec.setNtermBins(nterm_bins)
                subspec.setCtermBins(cterm_bins)
                yield subspec
               
    def generateNoiseSubSpectra(self,spectra,num, bin_len=0.1, bin_area=50, flag = 'num'):
        '''get NoisesubSpectrum from spectrum.
           created on Sep 22, 2015 by Johnqiu.'''
        for spectrum in spectra:
            pep = spectrum.getAnnotation() 
            title = spectrum.getTitle()
            Allmass = pep.getMass()
            
            masses = []
            for i in range(num):
                masses.append(random.uniform(0+bin_area,Allmass-bin_area))  #ycy0+50,Allmass-50   
        
            nterm_bins,cterm_bins = [],[]
            for i in range(len(masses)):
                nmass,cmass = 0,0
                nmass = masses[i]
                cmass = pep.getMass()-nmass
                nterm_bins = self.generateSubSpectrum(spectrum,nmass,bin_len,bin_area, flag) 
                cterm_bins = self.generateSubSpectrum(spectrum,cmass,bin_len,bin_area, flag)
                noisesubspec = NoiseSubSpectrum.NoiseSubSpectrum(spectrum.getPrecursorPeak(),title,masses[i])          
                noisesubspec.setAnnotation(pep)
                noisesubspec.setNtermBins(nterm_bins)
                noisesubspec.setCtermBins(cterm_bins)
                yield noisesubspec
                
    def generateSpecialSubSpectra(self,spectra,bin_len=0.1, bin_area=50, flag = 'num', stype='dual'):
        """
           get Spectal subSpectrum from spectrum.
           created on Sep 22, 2015 by Johnqiu.
           
           Args:
           stype:   dual|atype|yNH3|yH2O|bH2O|bNH3|y46-|y45-|y10+
        """
        for spectrum in spectra:
            pep = spectrum.getAnnotation() 
            title = spectrum.getTitle()
            masses = pep.getMassList()
            nterm_bins,cterm_bins = [],[]
            
            for i in range(len(masses)-1):
                nmass,cmass = 0,0
                if stype == 'dual':
                    nmass = pep.getMass()- masses[i] + 18  # cmass + H2O
                elif stype == 'atype':
                    nmass = masses[i]-28  
                elif stype == 'yNH3':
                    nmass = pep.getMass()- masses[i] + 1 
                elif stype == 'yH2O':
                    nmass = pep.getMass()- masses[i]  
                elif stype == 'bH2O':
                    nmass = masses[i]-10
                elif stype == 'bNH3':
                    nmass = masses[i]-17
                elif stype == 'y46-':
                    nmass = pep.getMass()- masses[i] + 18-46
                elif stype == 'y45-':
                    nmass = pep.getMass()- masses[i] + 18-45
                elif stype == 'y10+':
                    nmass = pep.getMass()- masses[i] + 18 + 10
                    
                cmass =pep.getMass()-nmass
                LA = pep.getAcids()[i-1].getResidue()
                RA = pep.getAcids()[i].getResidue()
                nterm_bins = self.generateSubSpectrum(spectrum,nmass,bin_len,bin_area, flag) 
                cterm_bins = self.generateSubSpectrum(spectrum,cmass,bin_len,bin_area, flag)
                dualsubspec = SubSpectrum.SubSpectrum(spectrum.getPrecursorPeak(),title,i+1,LA,RA)           
                dualsubspec.setAnnotation(pep)
                dualsubspec.setNtermBins(nterm_bins)
                dualsubspec.setCtermBins(cterm_bins)
                yield dualsubspec
                
    def generateSubSpectrum(self,spectrum,mass,bin_len,bin_area, flag='num'):
         '''generate SubSpectrum.created on Sep 22, 2015 by Johnqiu.'''
         bins = [0] * int(bin_area*2/bin_len)
         max_mass = mass + bin_area + bin_len/2  #ycy poisition in center
         min_mass = mass - bin_area + bin_len/2
     
         for peak in spectrum.getPeaks():
            mz = peak.getMz()
            if  mz < min_mass: continue
            if mz > max_mass: break
            elif mz >= min_mass and mz <= max_mass:
                pos = int((mz - min_mass) / bin_len)
                if flag == 'num':
                    bins[pos] += 1 # detect a peak
                elif flag == 'intensity':
                    bins[pos] = peak.getIntensity()  # set intensity
         return bins
    
    # change way to generate subspecturm, you will use list
    def generateSubSpecfile(self,filename, flag='num'):
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

    def generateNoiSubfile(self,filename, flag='num'):
         '''generate a set of Noise subSpectrums from filename'''  
         with open(filename) as input_data:   
             for line in input_data:
                 line = line[1:len(line)-2].split('###')
                 noisubData = {data.split('=')[0]:data.split('=')[1] \
                     for data in line}
                 # without precursor_peak
                 title = noisubData['Title']
                 pos = float(noisubData['PositionMZ'])
                 parentMass = float(noisubData['ParentMass'])
                 
                 ntermbin = noisubData['NTermBIN']
                 ctermbin = noisubData['CTermBIN']
                 
                 if flag == 'num':
                     ntermbin = [int(k) for k in ntermbin[1:len(ntermbin)-1].split(',')]           
                     ctermbin = [int(k) for k in ctermbin[1:len(ctermbin)-1].split(',')]
                 elif flag == 'intensity':
                     ntermbin = [float(k) for k in ntermbin[1:len(ntermbin)-1].split(',')]           
                     ctermbin = [float(k) for k in ctermbin[1:len(ctermbin)-1].split(',')]               

                 
                 noisubspect = NoiseSubSpectrum.NoiseSubSpectrum(None,title,pos)
                 noisubspect.setCtermBins(ctermbin)
                 noisubspect.setNtermBins(ntermbin)                 
                 yield noisubspect
    
    def generateCalBinsfile(self, filename):
        '''generate the result of calculating bins in spectrum from file'''
        with open(filename) as input_data:
            lines = ''
            for line in input_data:
                lines = lines + line
            lines = lines[1:len(lines)-5].split('###')
#            print lines
            binData = {data.split('=')[0]:data.split('=')[1] \
                    for data in lines}
                
#            print binData
           
            allNbins = binData['allNbins']
            allCbins = binData['allCbins']
            allbins = binData['allbins']
            
#            print allNbins[1:len(allNbins)-1].strip().split(' ')
#            
            allNbins = np.array(map(float,allNbins[1:len(allNbins)-1].strip().split('   ')))
            allCbins = np.array(map(float,allCbins[1:len(allCbins)-1].strip().split('   ')))
#            allbins = np.array(map(float,allbins[1:len(allbins)-1].strip().split('   ')))
            
#            allNbins = np.array(allNbins[1:len(allNbins)-1].strip().split('   '))
#            allCbins = np.array(allCbins[1:len(allCbins)-1].strip().split('   '))
#            allbins = np.array(allbins[1:len(allbins)-1].strip().split('   '))
            
            allNum = int(binData['allNum'])
            
                
            return allNbins,allCbins,allbins,allNum
        
    def generateCalBinsfile_cp(self, filename): 
        with open(filename,'r') as fr:
            data = cpickle.load(fr)
            return data
    

if __name__ == '__main__':

    start = time.clock()
    file_name = 'data/SRW.mgf'
#    file_name = 'data/new_CHPP_LM3_RP3_2.mgf'

    parser = SpectrumParser()
    parser = parser.readSpectrum(file_name)
 #   spectrum = parser.next()
#    spectra = [spectrum]
#    for i in range(1000):
 #       spectra +=[parser.next()]

    spectra = list(parser)

    subparser = SubSpectrumGenerator()
    subspecs = list(subparser.generateSubSpectra(spectra, 0.1, 50))
#    for subspec in subspecs:
#               print('[Title='+ subspec.getTitle() + '###'\
#                          +'POSITION='+ str(subspec.getPosition())+ '###'\
#                          +'LENGTH='+str(len(subspec.getAnnotation().getAcids()))+'###'\
#                          +'LeftAA='+subspec.getLA()+'###'\
#                          +'RightAA='+subspec.getRA()+'###'\
#                          +'BIN_LENGTH='+str(subspec.getBinLength())+'###'\
#                          +'BIN_RANGE='+str(subspec.getBinArea())+'###'\
#                          +'NTermBIN='+str(subspec.getNtermBins())+'###'\
#                          +'CTermBIN='+str(subspec.getCtermBins())+']'+'\n')
#   
    with open('data/SRW_sub','w') as data:
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
    
    Subspects = subparser.generateSubSpecfile('data/SRW_sub')
    for subspec in Subspects:
        print subspec.getAnnotation().getMassList()[1]
        print subspec.getLA()
    
#    print Subspects[0].getCtermBins()
#    subparser.paintSubSpects(Subspects)
       
##    noisesubparser = SubSpectrumGenerator()
##    noisesubspecs = list(noisesubparser.generateNoiseSubSpectra(spectra, 3, 0.5, 50))
##    for noisesubspec in noisesubspecs:
##        print ('[TITLE='+ noisesubspec.getTitle() + ','\
##               +'PositionMZ='+ str(noisesubspec.getPositionmz())+ ','\
##               +'ParentMass='+ str(noisesubspec.getAnnotation().getParentMass())
##               +'NTermBIN= '+str(noisesubspec.getNtermBins())+','\
##               +'CTermBIN='+str(noisesubspec.getCtermBins())+']'+'\n')
#   
#
#"
#    
    end = time.clock()
#    print len(subspec)
    print 'time consuming %s seconds.' % (end-start)
#
#        
#
#    
#
