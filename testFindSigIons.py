from Peak import Peak
from Spectrum import Spectrum
from AminoAcid import AminoAcid
from Peptide import Peptide
from PeakGenerator import PeakGenerator
import IonSelector
import IonType
from IonType import PrefixIon
from scripts import ProteinWeightDict
from SpectrumParser import SpectrumParser
import FeatureFrequencyFunction

aa_table = ProteinWeightDict()

def testFindSigIons():
#        file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
        file_name = 'data/SRW.mgf'
        parser = SpectrumParser()
        spec = parser.readSpectrum(file_name).next()
        print spec.getAnnotation()
        pg = PeakGenerator(spec)
        ppeaks = pg.getTheoreticalPrefixPeaks(True,1)
        print 'theoretical prefix peaks are: ---------'
        for peak in ppeaks: print peak
#        print len(ppeaks)
        print 'theoretical suffix peaks are:----------'
        speaks = pg.getTheoreticalPrefixPeaks(False,1)
        for peak in speaks: print peak
#        print len(speaks)
        print 'theoretical precursor peak is : -------'
        rpeak = pg.getTheoreticalPrecursorPeak(1)
        print rpeak

        compared_peaks = []
        compared_peaks += [peak for peak in spec.getPeaks() if
        peak.getIntensity() > 0]
        print 'compared peaks are: ---------'
        for peak in compared_peaks : print peak
        #print len(compared_peaks)

        print 'prefix offsets are:-----------'
        normalizer = len(ppeaks) # normalizer = 3
        normalizer_for_precursor = 1 
        poffsets = []
        for bp in ppeaks:
            poffsets.extend(\
                            FeatureFrequencyFunction.getFeaturesBetweenPeaks(bp,compared_peaks,False))
        for offset in poffsets: print offset
#        print len
        print 'suffix offsets are: -----------'
        soffsets = []
        for bp in speaks:
            soffsets.extend(\
                            FeatureFrequencyFunction.getFeaturesBetweenPeaks(bp,compared_peaks,False))
        for offset in soffsets:print offset
        print 'precursor offsets are: --------'
        roffsets = []
        roffsets.extend(FeatureFrequencyFunction.getFeaturesBetweenPeaks(rpeak,compared_peaks,False))
        for offset in roffsets: print offset

        # update offsets_dict
        poffsets_dict = {}
        IonSelector.updateFeatures(poffsets_dict,poffsets)
        print 'update prefix offsets dict: ------------'
        for key,val in  poffsets_dict.items(): print key, val
        soffsets_dict = {}
        IonSelector.updateFeatures(soffsets_dict,soffsets)
        print 'update suffix offsets dict:----------'
        for key,val in soffsets_dict.items():print key,val
        roffsets_dict = {}
        IonSelector.updateFeatures(roffsets_dict,roffsets)
        print 'precursor offsets dict: ',roffsets_dict

        print 'produce temp_sig_ions dict'
        temp_sii = {}
        pgof_peaks = \
                     FeatureFrequencyFunction.getFeatureFrequencyPeak(poffsets_dict)
        for gof_peak in pgof_peaks:
            gof =gof_peak.getFeature()
            pion = \
                   IonType.PrefixIon('',gof.getBaseCharge(),gof.getOffset())
            temp_sii[pion] = gof_peak.getFrequency()

        sgof_peaks = \
                     FeatureFrequencyFunction.getFeatureFrequencyPeak(soffsets_dict)
        for gof_peak in sgof_peaks:
            gof = gof_peak.getFeature()
            sion = \
                   IonType.SuffixIon('',gof.getBaseCharge(),gof.getOffset())
            temp_sii[sion] = gof_peak.getFrequency()
        rgof_peaks = \
                     FeatureFrequencyFunction.getFeatureFrequencyPeak(roffsets_dict)
      #  print rgof_peaks
        for key, val in temp_sii.items():
            print key, val
  #      print len(temp_sii)
        
        print 'produce significant ions dict--------------'
        sig_ions = {}
        known_ion_types = IonType.getAllKnownIonTypes(spec.getCharge())
#        for ion in  known_ion_types: print ion
        for ion in temp_sii:
            if isinstance(ion,IonType.PrecursorIon):continue
            for kion in known_ion_types:
                if (ion.isPrefixIon() and kion.isPrefixIon()) or (not ion.isPrefixIon()  and not kion.isPrefixIon()):
                    tdiff = abs(ion.getOffset()-kion.getOffset())
                    if tdiff < 0.5:
                        if kion in sig_ions:
                            sig_ions[kion] = sig_ions[kion]+1
                        else:
                            sig_ions[kion] = 1

        for ion,freq in sig_ions.items():print ion, freq

if __name__ == '__main__':
    testFindSigIons()
