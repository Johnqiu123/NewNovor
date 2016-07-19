from scripts import ProteinWeightDict
from scripts import Composition
from AminoAcid import AminoAcid

aa_table = ProteinWeightDict()

class Peptide(object):
    '''represents a peptide comsists a sequence of amino acid.
       created on Sep 22, 2015 by mht,Johnqiu
    '''    
    def __init__(self, acids):
        self.acids = acids

    def __str__(self):
        s = ''
        for acid in self.acids:
            s += 'residue: ' + acid.getResidue() + ' mass: ' + str(acid.getMass()) + '\n'
        return s

    def getAcids(self):        return self.acids
    def getMass(self):
        return sum([acid.getMass() for acid in  self.getAcids()])
    def getNominalMass(self):
        return sum([aa.getNominalMass() for aa in self.getAcids()])

    def getParentMass(self):
        return self.getMass() + Composition.H2O
    
    def getPrefixMasses(self,is_prefix):
        '''return prefix masses or suffix masses and their positions.'''
        masses = {}
        for pos in range(len(self.acids)-1):
            if is_prefix:
                mass = Peptide(self.acids[:pos+1]).getMass()
                masses[mass] = pos+1
            else:
                mass = Peptide(self.acids[pos+1:]).getMass()
                masses[mass] = len(self.acids)-pos-1
        return masses

    def getMassList(self):
        '''return masses list.'''
        masses = []
        for pos in range(len(self.acids)):
            mass = Peptide(self.acids[:pos+1]).getMass()
            masses.append(mass)
        return masses        
    

    def getBIonPeaks(self):
        bions = []
        bmass = 1 # for mass of b-ion is 1
        for aa in self.getAcids():
            bmass += aa.getNominalMass()
            bions.append(bmass)
        return bions[:-1] # b-ion not contain the whole peptide

    def getYIonPeaks(self):
        yions = []
        ymass = 19
        for aa in list(reversed(self.getAcids())):
            ymass += aa.getNominalMass()
            yions.append(ymass)
        return yions[:-1] # y-ion not contain the whole peptide                       
            
if __name__ == '__main__':
    seq = 'SWR'
    acids = []
    acids += [AminoAcid(s,'',aa_table[s]) for s in seq if s in aa_table]    
    pep = Peptide(acids)
    pepmass = pep.getPrefixMasses(True)
    listmass = pep.getMassList();
    s = ''
    for acid in pep.getAcids():
        s = s + acid.getResidue()
    print listmass[1]
    print pep.getMass()
    print s
    for mass in listmass:
        print mass
    for i in range(len(listmass)):
        print listmass[i]
    for m,pos in pepmass.items(): # Iterator
        print m,pos
#    print pep.getParentMass()
#$    print pep.getBIonPeaks(),pep.getYIonPeaks()
    masses =  pep.getPrefixMasses(True)
#    for mass, pos in masses.items(): print mass, pos
    aseq = 'AGGPLER'
    aacids = []
    aacids += [AminoAcid(s,'',aa_table[s]) for s in aseq if s in aa_table]
    apep = Peptide(aacids)
 #   print apep
    amasses = apep.getPrefixMasses(False)
#    for m,pos in amasses.items():
#        print m,pos
    



    

        
