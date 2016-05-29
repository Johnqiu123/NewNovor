from scripts import Composition
#from Spectrum import Spectrum

class Peak(object):
    ''' repreasents a simple peak in spectrum.
       created on Sep 10, 2015 by mht.
    '''
    def __init__(self, charge=1, mz=0.0, intensity=0.0,pos=0):
        self.charge = charge  
        self.mz = mz
        self.intensity = intensity
        self.pos = pos # peak position, used for represent fragmentation

    def __str__(self):
        return 'Peak(mz:%s, intensity:%s)' % (self.mz, self.intensity)
    def __cmp__(self, other):
        if self.intensity > other.intensity:  return 1
        if self.intensity < other.intensity:  return -1
        if self.mz > other.mz:  return 1
        if self.mz < other.mz:  return -1
        return 0

    def __eq__(self,other,tol=0.5):
        return abs(self.mz-other.mz)<tol and abs(self.intensity - other.intensity) < tol
        
    def getAbsoluteMassDiff(self, other):
        ''' return the absolute mass difference between 2 peaks.'''
        return abs(self.mz - other.mz)

    def getMass(self):
        return (self.mz - float(Composition.PROTON)) * self.charge

    def getCharge(self): return self.charge
    def setCharge(self,charge): self.charge = charge
    def getMz(self):     return self.mz
    def getIntensity(self):  return self.intensity
    def getPosition(self): return self.pos

if __name__ == '__main__':
    p1 = Peak(1, 274.112,40.1)
    p2 = Peak(2,361.121,80.1)
    print p2
#    print p1.getAbsoluteMassDiff(p2)
    print p2.getMass()
    print p2.getPosition()
