from scripts import Composition
from Peptide import Peptide
from Peak import Peak
from Spectrum import Spectrum
from IonType import IonType,SuffixIon,PrefixIon,PrecursorIon
from AminoAcid import AminoAcid, aa_table


virtual_peak_intensity = float('inf') #

class PeakGenerator(object):
    '''generates theoretical peak of a peptide and an ion.
       created on Sep 9, 2015 by mht.
    '''
    def __init__(self,spec):
        self.peptide = spec.getAnnotation()        
    def getPeptide(self): return self.peptide

    def getTheoreticalPeaks(self,ion):
        if self.peptide == None: return None
        peaks = []
        if isinstance(ion,PrefixIon):
            peaks = self.getTheoreticalPrefixPeaks(True,ion.getCharge(),
        ion.getOffset()*ion.getCharge())
        elif isinstance(ion,SuffixIon):
            peaks = self.getTheoreticalPrefixPeaks(False,ion.getCharge(),ion.getOffset()*ion.getCharge())
        elif isinstance(ion, PrecursorIon):
            peaks.append(self.getTheoreticalPrecursorPeak(ion.getCharge(),ion.getOffset()*ion.getCharge()))
        return peaks

    def getTheoreticalPrefixPeaks(self, is_prefix,charge,offset=0):
        ''' returns theoretical prefix peaks.
          args: is_prefix: Ture if prefix otherwise suffix.'''
        peaks = []
        if self.getPeptide() == None:
            return None
        masses = self.getPeptide().getPrefixMasses(is_prefix)
        for m,pos in masses.items():
            peaks.append(Peak(charge, (m+offset)/charge, virtual_peak_intensity,pos))
        return peaks

    def getTheoreticalPrecursorPeak(self,charge,offset=0):
        if self.peptide == None: return None
        return Peak(charge,(self.peptide.getParentMass()+offset)/charge+Composition.PROTON,virtual_peak_intensity,len(self.peptide.getAcids()))

def getComplementaryPeak(peak, charge, pm_or_spec):
    '''gets the complementary peak from a parent mass or a spectrum.'''
    if isinstance(pm_or_spec, float):
        return Peak(charge, (pm_or_spec/charge - peak.getMz() + 2*Composition.PROTON), virtual_peak_intensity)
    elif isinstance(pm_or_spec,Spectrum):
        return Peak(charge, (float(pm_or_spec.getParentMass())/charge -
        peak.getMz() + 2*Composition.PROTON),virtual_peak_intensity)

def getPrefixMass(peak, ion, spec):
    '''returns the prefix mass.'''
    mass = ion.getMass(peak.getMz())
    if isinstance(ion,SuffixIon):
        mass = spec.getPeptideMass() - mass
    return mass

def testGetPrefixPeaks():
        ion1 = Peak(1,274.112,40.1) #
        ion2 = Peak(1,361.121,80.1) 
        spec = Spectrum()
        spec.addPeak(ion1)
        spec.addPeak(ion2)
        
        seq = 'SWR'
        acids = []
        acids += [AminoAcid(s,'',aa_table[s]) for s in seq]
        peptide = Peptide(acids)
        spec.setAnnotation(peptide)
        
        pg = PeakGenerator(spec)
        ppeaks = pg.getTheoreticalPrefixPeaks(True,1)
        print 'theoretical prefix peaks are: ---------'
        for peak in ppeaks: print peak.getMz(),':', peak.getPosition()
        print 'theoretical suffix peaks are:----------'
        speaks = pg.getTheoreticalPrefixPeaks(False,1)
        for peak in speaks: print peak.getMz(),':',peak.getPosition()
        print 'theoretical precursor peak is : -------'
        rpeak = pg.getTheoreticalPrecursorPeak(1)
        print rpeak.getMz(),':',rpeak.getPosition()

        compared_peaks = []
        compared_peaks += [peak for peak in spec.getPeaks() if
        peak.getIntensity() > 0]
        print 'compared peaks are: ---------'
        for peak in compared_peaks : print peak
        
if __name__ == '__main__':
    testGetPrefixPeaks()

