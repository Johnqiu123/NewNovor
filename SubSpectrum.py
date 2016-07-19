from scripts import ProteinWeightDict
from Peak import Peak
from Spectrum import Spectrum

aa_table = ProteinWeightDict()
class SubSpectrum(Spectrum):
    ''' represents a subspectrum given a spectrum.
        created on Sep 22, 2015, by Johnqiu,mht
    '''
    bin_length = 0.1
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
    def setBinLength(self,length):
        bin_length = length
    def getBinArea(self):
        return SubSpectrum.bin_area
    def setBinArea(self,area):
        bin_area = area

    # new code 15/12/27
    def setLength(self, length): self.length = length
    def getLength(self): return self.length
if __name__ == '__main__':
    bins = [0] * int(50*2/0.5)
#    print peak.getNumberofbins()
    
    import numpy as np
    s=[]
    s0 = np.array(s)
    s1 = np.array([[2,5],[6,7]])
    s3 = np.arange(-50,50,0.5)
    
    s4 = [x**2 for x in s1]
#    print np.mat(s4)[:,0]
    

    x,y = 0,0
    x,y = 3, 5-x
    
    str = "[ssssss]"

    print str[1:len(str)-1]
    


    
