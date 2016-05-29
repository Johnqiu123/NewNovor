import FeatureFrequencyFunction
from Spectrum import Spectrum
from Peak import Peak
from PeakGenerator import PeakGenerator
from AminoAcid import AminoAcid
from Peptide import Peptide
from scripts import ProteinWeightDict
import IonType

aa_table = ProteinWeightDict()
class IonSelector(object):
    '''selects significant ion types using offset frequency function(OFF).
       created on Sep 10, 2015 by mht.'''
    min_ion_num = 10 # minimum ion number
    sig_ion_prob = 0.15 # the significant prob. to select ions

    def __init__(self,tol,max_ion_num):
        self.sig_ions = dict() # dict{ion:intensity}
        self.tol = tol
        self.max_ion_num = max_ion_num

    def getSigIonDict(self):return self.sig_ions
    def findSigIons(self, charge,specs):
        '''returns the number of significant ions given the spectrum charge.
          created on Aug 6, 2015, need more and more revised.'''

        # (1) initialize variables

        temp_sii = dict() # temp sig_ion_intensity
        num_of_sii = 0
        poffsets_dict = dict() # dict{RelationBetweenPeaks,Integer}
        soffsets_dict = dict()
        roffsets_dict = dict() # precursor offsets

        prev_num = 0
        normalizer = 0.0

        # (2) process spectra and get offsets
        for spec in specs:
            for charge in range(1,spec.getCharge()+1):
                peak_gen = PeakGenerator(spec)
                compared_peaks = []
                compared_peaks += [peak for peak in spec.getPeaks() if \
                                   peak.getIntensity() >0]
                pbps = peak_gen.getTheoreticalPrefixPeaks(True,charge)
                sbps = peak_gen.getTheoreticalPrefixPeaks(False,charge)
                rbp = peak_gen.getTheoreticalPrecursorPeak(charge)

                poffsets,soffsets,roffsets = [],[],[]# RelationBetweenPeaks list
                for bp in pbps:
                    poffsets.extend(FeatureFrequencyFunction.getFeaturesBetweenPeaks(bp,compared_peaks,False))
                for bp in sbps:
                    soffsets.extend(FeatureFrequencyFunction.getFeaturesBetweenPeaks(bp,compared_peaks,False))
                roffsets.extend(FeatureFrequencyFunction.getFeaturesBetweenPeaks(rbp,compared_peaks,False))
                updateFeatures(poffsets_dict,poffsets)
                updateFeatures(soffsets_dict,soffsets)
                updateFeatures(roffsets_dict,roffsets)


        #(3) produce prefix and suffix ions present in spectrum
        temp_sii = self.getTempSII(poffsets_dict,soffsets_dict,roffsets_dict)

        #(4) get sig_ions dict
        self.sig_ions = self.getSigIons(charge,temp_sii)
        return self.sig_ions

    def getTempSII(self,poffsets_dict,soffsets_dict,roffsets_dict):
        temp_sii = {}
        pgof_peaks =  FeatureFrequencyFunction.getFeatureFrequencyPeak(poffsets_dict)
        for gof_peak in pgof_peaks:
            gof = gof_peak.getFeature()
            pion = IonType.PrefixIon('',gof.getBaseCharge(),gof.getOffset(),gof.getPosition())
            temp_sii[pion] = gof_peak.getFrequency()

        sgof_peaks = FeatureFrequencyFunction.getFeatureFrequencyPeak(soffsets_dict)
        for gof_peak in sgof_peaks:
            gof = gof_peak.getFeature()
            sion = \
            IonType.SuffixIon('',gof.getBaseCharge(),gof.getOffset(),gof.getPosition())
            temp_sii[sion] = gof_peak.getFrequency()
        
        rgof_peaks = FeatureFrequencyFunction.getFeatureFrequencyPeak(roffsets_dict)
        for gof_peak in rgof_peaks:
            gof = gof_peak.getFeature()
            rion = IonType.PrecursorIon(gof.getBaseCharge(),gof.getOffset(),gof.getPosition())
            temp_sii[rion] = gof_peak.getFrequency()
        return temp_sii

    def getSigIons(self,charge,temp_sii):
        sig_ions = {}
        known_ion_types = IonType.getAllKnownIonTypes(charge)

        for ion in temp_sii:
            if isinstance(ion, IonType.PrecursorIon):  continue
            is_known = False
            diff = 10000.0
            key_ion = None
            intensity = 0.0

            for kion in known_ion_types:
                if isinstance(kion, IonType.PrecursorIon): continue

                if (ion.isPrefixIon() and kion.isPrefixIon()) or (not
                ion.isPrefixIon() and not kion.isPrefixIon()):
                    tdiff = abs(ion.getOffset() - kion.getOffset())
                    if tdiff < diff:
                        diff = tdiff
                        if diff < 0.5:
                            if kion in sig_ions:
                                prev = sig_ions[kion]+1
                            else:
                                prev = 1
                            key_ion = kion
                            key_ion.setPosition(ion.getPosition())
                            frequency = max(prev,temp_sii[ion])
                            is_known = True
#            if not is_known:
 #               sig_ions[ion] = temp_sii[ion]
  #          else:
   #             sig_ions[key_ion] = frequency
            if is_known: sig_ions[key_ion] = frequency
        return sig_ions
    
def updateFeatures(feature_dict,features):
        if features == None: return
        for gof in features:
            if gof in feature_dict:
                freq = feature_dict[gof]                
            else: freq = 0
            freq += 1
            feature_dict[gof] = freq
        
if __name__ == '__main__':
    ion1 = Peak(2,274.112,40.1) #
    ion2 = Peak(2,361.121,80.1)
    spec = Spectrum()
    spec.addPeak(ion1)
    spec.addPeak(ion2)

    seq = 'SWR'
    acids = []
    for s in seq:
        acids.append(AminoAcid(s, '', aa_table[s]))
    pep = Peptide(acids)
    spec.setAnnotation(pep)
    specs = []
    specs.append(spec)
    ist = IonSelector(0.5,10)
    sig_ions =  ist.findSigIons(1,specs)
    for ion,freq in sig_ions.items(): print ion,':',ion.getPosition(),':',freq

#    for ion in IonType.getAllKnownIonTypes(spec.getCharge()):
 #       print ion


    
    
