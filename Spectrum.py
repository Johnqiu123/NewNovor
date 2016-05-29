from scripts import Composition
from scripts import ProteinWeightDict
from Peak import Peak
from Peptide import Peptide
from AminoAcid import AminoAcid
import copy

aa_table = ProteinWeightDict()
class Spectrum(object):
    ''' represents a simple spectrum given a mass.
        created on Sep 10, 2015, by mht.
    '''
    def __init__(self, precursor_peak=Peak(),title=''):
        self.peaks = []
        self.annotation = None # peptide annotation
        self.precursor_peak = precursor_peak
        self.title = title
        self.peaks.append(precursor_peak)

    def __str__(self):
        return 'Sepctrum-mz: %g, #peaks: %d' % (self.getPrecursorPeak().getMz(), self.getNumberOfPeaks())

    def getPeaks(self):        return self.peaks
    def getTitle(self): return self.title
    def getPrecursorPeak(self): return self.precursor_peak
    def getNumberOfPeaks(self): return len(self.peaks)
    def getParentMass(self): return self.precursor_peak.getMass()
    def getCharge(self):        return self.precursor_peak.getCharge()
    def getAnnotation(self): return self.annotation
    def getPeptideMass(self):
        '''returns the peptide mass of this spectrum: parentMass-mass(H2O).'''
        return (self.getPrecursorPeak().getMass() - Composition.H2O)

    def setTitle(self,title): self.title = title
    def setPrecursor(self,peak): self.precursor_peak = peak
    def setAnnotation(self,peptide):
        if isinstance(peptide, Peptide):
            self.annotation = peptide
        else: # peptide is a string
            acids = []
            acids += [AminoAcid(s,'',aa_table[s]) for s in peptide if s in
    aa_table]
            self.annotation = Peptide(acids)            
    
    def addPeak(self,peak):  self.peaks.append(peak)
    def removePeak(self,peak):  self.peaks.remove(peak)
    def clearPeaks(self): self.peaks = []
    def sortPeaks(self): self.peaks = sorted(self.peaks)
    def getPeakByMass(self,mass, tolerance):
        '''returns the most intensity peak within tolerance of the target mass.
        args:
          mass-target mass
        returns:
          a peak if there is match or none otherwise.'''
        matches = self.getPeaksByMass(self, mass, tolerance)
        if (matches == None or len(matches) == 0):
            return None
        else:
            return max(matches)

    def getPeaksByMass(self, mass, tol=0.5):
        '''returns a list of peaks that match the target mass within the
        tolernace value. The absolute distance between mass and a returned peak
        is less or equal the tolerance value.'''
        return self.getPeaksByMassRange(mass-tol, mass+tol)

    def getPeaksByMassRange(self,min_mass, max_mass):
        '''returns a list of peaks that match the mass within the specified
        range. Assuming spectrum is sorted by mass.'''
        matches = []
        peaks = sorted(self.getPeaks())

        for peak in peaks:
            if peak.getMz() < min_mass: break
            if peak.getMz() > max_mass: break
            else:
                matches.append(peak)
        return matches

if __name__ == '__main__':
    p1 = Peak(1, 274.112,40.1)
    p2 = Peak(1,361.121,80.1)
    peak = Peak(2,448.225,80.1)
    spec = Spectrum(peak)
    spec.addPeak(p1)
    print spec.getAnnotation()
    for peak in spec.getPeaks():
        print peak.getCharge()
    



    
