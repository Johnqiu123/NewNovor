from scripts import ProteinDictDNA

aa_table = ProteinWeightDict()

class Peptide(object):
    '''represents a peptide comsists a sequence of amino acid.
       created on Jul 22, 2015 by mht.
    '''
    
    def __init__(self, acids):
        self.acids = []
        self.acids += [acid for acid in acids]

    def __str__(self):
        for acid in self.acids:
            s += acid
        return s

if __name__ == '__main__':
    pep = Peptide('QAKCR')
    print pep

    

        
