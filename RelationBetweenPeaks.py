from Peak import Peak
import PeakGenerator
from Spectrum import Spectrum

class RelationBetweenPeaks(object):
    ''' represents the relation between two peaks considering m/z offset and
    complementary relation.
    This is used to define offsetFeature and for training.
      created on Sep 10, 2015 by mht.
    '''
    def __init__(self,base_charge,offset,is_complementary,pos=0):
        self.base_charge = base_charge
        self.offset = offset # m/z offset
        self.is_complementary = is_complementary
        self.pos = pos

    def __str__(self):
        return 'MzOff: %g Comp:%s ' % (self.offset, self.is_complementary)

    def getBaseCharge(self): return self.base_charge
    def getOffset(self): return self.offset
    def isComplementary(self): return self.is_complementary
    def getPosition(self): return self.pos

if __name__ == '__main__':
    relation = RelationBetweenPeaks(1,0.1,True)
    peak = Peak(1,87.1,1.2) # residule-S
    print peak.getMz()
    precursor_peak = Peak(1, 429.1,1.2)


