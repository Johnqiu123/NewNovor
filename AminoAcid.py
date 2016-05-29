from scripts import ProteinWeightDict


INTEGER_MASS_SCALER = 0.999497
aa_table = ProteinWeightDict()

class AminoAcid(object):
    '''represents a stardard amino acid with a residue, name and mass.
       created on Jul 22, 2015 by mht.'''
    
    def __init__(self,residue,name,mass):
        self.residue = residue
        self.name = name
        self.mass = mass

    def __str__(self):
        return 'residue:%s, mass:%g' % (self.getResidue(), self.getMass())

    def getResidue(self): return self.residue
    def getName(self): return self.name
    def getMass(self): return self.mass
    def getNominalMass(self):
        return int(round(INTEGER_MASS_SCALER*self.getMass()))
#

if __name__ == '__main__':
    a = AminoAcid('Q','qusery',aa_table['Q'])
    print a
    print a.getNominalMass()

