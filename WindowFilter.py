from SpectrumParser import SpectrumParser
import copy

class WindowFilter(object):
    '''  This filtering method guarantees that for any window of the determined size
  placed around a given peak, the peak is ranked better than a determined 
  parameter.
     created on Sep 15, 2015.
    '''
    def __init__(self,topk=6,window_size=50):
        self.topk = topk
        self.window_size = window_size

    def filtered(self,spec):
        '''select each peak if it is top n within window (-window,+window) around it.
        returns the filtered spectrum.'''
        result = copy.copy(spec) # shallow copy
        result.clearPeaks()
        
        peaks = spec.getPeaks()
        for i in range(len(peaks)):
            rank = 1
            peak = peaks[i]
            mass, intensity = peak.getMass(),peak.getIntensity()

            # move left
            prev = i - 1
            while (prev >= 0):
                prev_peak = peaks[prev]
                if (mass - prev_peak.getMass()) > self.window_size:  break
                if prev_peak.getIntensity() > intensity:
                    rank += 1
                prev -= 1

            # move right
            succ = i + 1
            while (succ < len(peaks)):
                next_peak = peaks[succ]
                if (next_peak.getMass()-mass > self.window_size):  break
                if (next_peak.getIntensity() > intensity):
                    rank += 1
                succ += 1

            if rank <= self.topk:
                result.addPeak(peak)
        return result

if __name__ == '__main__':
    filter = WindowFilter(3,10)
    file_name = 'data/AGGP.mgf'
    parser = SpectrumParser()
    spec = parser.readSpectrum(file_name).next()

    fspec = filter.filtered(spec)
    print len(fspec.getPeaks())
    #for peak in fspec.getPeaks(): print peak
    print fspec.getPrecusorPeak()

            
            
            
            
        
