from scripts import ProteinWeightDict
from Peak import Peak
from Spectrum import Spectrum


aa_table = ProteinWeightDict()
class NoiseSubSpectrum(Spectrum):
    ''' represents a subspectrum given a spectrum.
        created on Sep 22, 2015, by Johnqiu,mht
    '''
    bin_length = 0.1
    bin_area = 50 
    def __init__(self, precursor_peak, title, positionmz):
        Spectrum.__init__(self,precursor_peak,title)
        self.positionmz = positionmz  # Noise mass
        self.nterm_bins = []
        self.cterm_bins = []


    def __str__(self):
         return 'NoiseSubSepctrum pos: %d, #Nbins: %d, #Cbins: %d' % (self.getPosition(), \
        self.getNumberofNBins(),self.getNumberofCBins())


    def getPositionmz(self): return self.positionmz
    def getNumberofNBins(self):  return len(self.nterm_bins)
    def getNumberofCBins(self):  return len(self.cterm_bins)   
    def getNtermBins(self): return self.nterm_bins
    def getCtermBins(self): return self.cterm_bins
    def setNtermBins(self,bins): self.nterm_bins = bins
    def setCtermBins(self,bins): self.cterm_bins = bins
    def setPositionmz(self,mass): self.positionmz = mass

    def getBinLength(self):
        return NoiseSubSpectrum.bin_length
    def setBinLength(length):
        bin_length = length
    def getBinArea(self):
        return NoiseSubSpectrum.bin_area


if __name__ == '__main__':
    bins = [0] * int(50*2/0.5)*2
    print len(bins)
#    print peak.getNumberofbins()
    
    import numpy as np
    s = np.array([0,2])
    s1 = np.array([1,1])
    print s+s1
    
    masses = []

    
