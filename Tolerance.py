
class Tolerance(object):
    '''represents the tolerance of peptide masses.
       created on Jul 20, 2015.
    '''
    def __init__(self,value):
        self.value = value

    def __str__(self):
        return '%gDa' % self.value

    def __cmp__(self, other):
        pass

    def getToleranceAsDa(self,mass):
        return 1e-6*self.value*mass

if __name__ == '__main__':
    t = Tolerance(0.3)
    print t.getToleranceAsDa(57)
