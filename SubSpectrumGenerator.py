from SpectrumParser import SpectrumParser
import SubSpectrum
import NoiseSubSpectrum
from Spectrum import Spectrum
from scripts import ProteinWeightDict
import time
import random
''' parse spectrum file with mgf format.
   created on Sep 22, 2015 by Johnqiu.
'''
aa_table = ProteinWeightDict()
class SubSpectrumGenerator(object):
    
    
    
    def generateSubSpectra(self,spectra, bin_len=0.5, bin_area=50):
        '''get subSpectrum from spectrum.
           created on Sep 22, 2015 by Johnqiu.'''
        for spectrum in spectra:
            pep = spectrum.getAnnotation() 
            title = spectrum.getTitle()
            masses = pep.getMassList()
            nterm_bins,cterm_bins = [],[]
            for i in range(len(masses)-1):
                nmass,cmass = 0,0
                nmass,cmass = masses[i],pep.getMass()-nmass
                LA = pep.getAcids()[i-1].getResidue()
                RA = pep.getAcids()[i].getResidue()
                nterm_bins = self.generateSubSpectrum(spectrum,nmass,bin_len,bin_area) 
                cterm_bins = self.generateSubSpectrum(spectrum,cmass,bin_len,bin_area)
                subspec = SubSpectrum.SubSpectrum(spectrum.getPrecursorPeak(),title,i+1,LA,RA)           
                subspec.setAnnotation(pep)
                subspec.setNtermBins(nterm_bins)
                subspec.setCtermBins(cterm_bins)
                yield subspec
                
    def generateNoiseSubSpectra(self,spectra,num, bin_len=0.5, bin_area=50):
        '''get NoisesubSpectrum from spectrum.
           created on Sep 22, 2015 by Johnqiu.'''
        for spectrum in spectra:
            pep = spectrum.getAnnotation() 
            title = spectrum.getTitle()
            Allmass = pep.getMass()
            
            masses = []
            for i in range(num):
                masses.append(random.uniform(0,Allmass))   
        
            nterm_bins,cterm_bins = [],[]
            for i in range(len(masses)):
                nmass,cmass = 0,0
                nmass,cmass = masses[i],pep.getMass()-nmass
                nterm_bins = self.generateSubSpectrum(spectrum,nmass,bin_len,bin_area) 
                cterm_bins = self.generateSubSpectrum(spectrum,cmass,bin_len,bin_area)
                noisesubspec = NoiseSubSpectrum.NoiseSubSpectrum(spectrum.getPrecursorPeak(),title,masses[i])          
                noisesubspec.setAnnotation(pep)
                noisesubspec.setNtermBins(nterm_bins)
                noisesubspec.setCtermBins(cterm_bins)
                yield noisesubspec
                
    def generateSubSpectrum(self,spectrum,mass,bin_len,bin_area):
        bins = [0] * int(bin_area*2/bin_len)
        max_mass = mass + bin_area
        min_mass = mass - bin_area

        for peak in spectrum.getPeaks():
            mz = peak.getMz()
            if  mz < min_mass: continue
            if mz > max_mass: break
            elif mz >= min_mass and mz <= max_mass:
                pos = int((mz - min_mass) / bin_len)
                bins[pos] += 1 # detect a peak
        return bins


if __name__ == '__main__':
    start = time.clock()
    file_name = 'data/AGGPGLER.mgf'
#    file_name = 'data/new_CHPP_LM3_RP3_2.mgf'

    parser = SpectrumParser()
    parser = parser.readSpectrum(file_name)
 #   spectrum = parser.next()
#    spectra = [spectrum]
#    for i in range(1000):
 #       spectra +=[parser.next()]

    spectra = list(parser)

#    subparser = SubSpectrumGenerator()
#    subspecs = list(subparser.generateSubSpectra(spectra, 0.5, 50))
#    for subspec in subspecs:
#        print ('[TITLE='+ subspec.getTitle() + ','\
#               +'LENGTH='+str(len(subspec.getAnnotation().getAcids()))+','\
#               +'POSITION='+ str(subspec.getPosition())+ ','\
#               +'LEFTAA='+subspec.getLA()+','\
#               +'RIGHTAA='+subspec.getRA()+','\
#                             +'NBINS= '+str(subspec.getNtermBins())+','\
#               +'CBINS='+str(subspec.getCtermBins())+']'+'\n')
#   
    noisesubparser = SubSpectrumGenerator()
    noisesubspecs = list(noisesubparser.generateNoiseSubSpectra(spectra, 3, 0.5, 50))
    for noisesubspec in noisesubspecs:
        print ('[TITLE='+ noisesubspec.getTitle() + ','\
               +'PositionMZ='+ str(noisesubspec.getPositionmz())+ ','\
               +'ParentMass='+ str(noisesubspec.getAnnotation().getParentMass())
               +'NTermBIN= '+str(noisesubspec.getNtermBins())+','\
               +'CTermBIN='+str(noisesubspec.getCtermBins())+']'+'\n')
    
    end = time.clock()
#    print len(subspec)
    print 'time consuming %s seconds.' % (end-start)

        

    

