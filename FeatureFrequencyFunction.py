from Peak import Peak
from RelationBetweenPeaks import RelationBetweenPeaks

max_mz_offset = 38
min_mz_offset = -38

class FeatureFrequencyFunction(object):
    '''represents the  frequency of a feature.
       created on Jul 27, 2015 by mht.
    '''
    pass

class FeatureFrequencyPeak(object):
    ''' represents peak of feature frequency.'''

    def __init__(self, feature, freq):
        self.feature = feature  # feature , also a RelationBetweenPeaks
        self.freq = freq # frequency

    def __str__(self):
        return '[%s], freq: %g' % (str(self.feature), self.freq)

    def __cmp__(self, other):
        if self.freq > other.freq: return 1
        if self.freq < other.freq: return -1
        return 0

    def getFeature(self): return self.feature
    def getFrequency(self):return self.freq

def getFeaturesBetweenPeaks(peak, peaks, is_complementary):
    features = [] # RelationBetweenPeaks
    for cp in peaks: # current peak
        feature = getRelationBetweenPeaks(peak, cp,is_complementary)
        if feature:
            features.append(feature)
    return features

def getRelationBetweenPeaks(peak, cpeak,is_complementary):
    offset = (cpeak.getMz()-peak.getMz())*peak.getCharge()
    if (offset > max_mz_offset or offset < min_mz_offset):
        return None
    return RelationBetweenPeaks(peak.getCharge(),offset,is_complementary,peak.getPosition())

def getFeatureFrequencyPeak(feature_freqs):
    '''returns the feature frequency peak, such as {feature,freq}
      args:
        feature_freqs- feature frequences, dict{RelationBetweenPeaks,int}
        threshold - a float threshold to select features
    '''
    
    num_peaks = 100 # max number of peaks considered
    fff_peaks = [] # FeatureFrequencyPeak list
    if feature_freqs is None or len(feature_freqs) == 0:
        return fff_peaks

    fff_peaks_tmp = []
    for feature in feature_freqs:
        numbers = (int)(feature_freqs[feature])
        fff_peaks_tmp.append(FeatureFrequencyPeak(feature,numbers))
    fff_peaks_tmp.reverse()
    
    for i in range(min(len(fff_peaks_tmp), num_peaks)):
        fff_peaks.append(fff_peaks_tmp[i])
    return fff_peaks

if __name__ == '__main__':
    peak = Peak(1,100.1, 50.1)
    peak1 = Peak(1,100.3,50.1)
    peak2 = Peak(1,110.2,80.1)
    peaks = [peak1,peak2]
    features =  getFeaturesBetweenPeaks(peak, peaks,False)
    for relation in features: print relation


    
