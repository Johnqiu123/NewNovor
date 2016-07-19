from scripts import ProteinWeightDict
from Peak import Peak
from Spectrum import Spectrum

aa_table = ProteinWeightDict()
class SubSpectrum(Spectrum):
    ''' represents a subspectrum given a spectrum.
        created on Sep 22, 2015, by Johnqiu,mht
    '''
    bin_length = 0.5
    bin_area = 50 
    def __init__(self, precursor_peak, title, pos,LA,RA):
        Spectrum.__init__(self,precursor_peak,title)
        self.position = pos  # bp number
        self.nterm_bins = []
        self.cterm_bins = []
        self.LA = LA
        self.RA = RA

    def __str__(self):
        return 'SubSepctrum pos: %d, #Nbins: %d, #Cbins: %d' % (self.getPosition(), \
        self.getNumberofNBins(),self.getNumberofCBins())

    def getPosition(self): return self.position
    def getNumberofNBins(self):  return len(self.nterm_bins)
    def getNumberofCBins(self):  return len(self.cterm_bins)   
    def getNtermBins(self): return self.nterm_bins
    def getCtermBins(self): return self.cterm_bins
    def setNtermBins(self,bins): self.nterm_bins = bins
    def setCtermBins(self,bins): self.cterm_bins = bins
    def setPosition(self,pos): self.position = pos
    def getLA(self): return self.LA
    def getRA(self): return self.RA

    def getBinLength(self):
        return SubSpectrum.bin_length
    def setBinLength(length):
        bin_length = length
    def getBinArea(self):
        return SubSpectrum.bin_area

if __name__ == '__main__':
    bins = [0] * int(50*2/0.5)
#    print peak.getNumberofbins()
    
    import numpy as np
    s = np.array([0,2])
    s1 = np.array([1,1])
    print s+s1
    


    
