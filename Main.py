from SpectrumParser import SpectrumParser
from Tolerance import Tolerance
from AminoAcid import AminoAcid
from Peptide import Peptide
from IonSelector import IonSelector
from WindowFilter import WindowFilter
from scripts import ProteinWeightDict

''' Main: test ion selector method on real spectrum file.
   created on Sep 15, 2015 by mht.
'''
aa_table = ProteinWeightDict()

def parse(file_name):
    parser = SpectrumParser()
    return parser.readSpectrum(file_name).next()

def setPeptide(sequence):
    acids = []
    for s in sequence:
        acids.append(AminoAcid(s, s, aa_table[s]))
    peptide = Peptide(acids)
    return peptide

def setIonSelector():
    tol = Tolerance(0.5)
    max_ion_num = 10
    return IonSelector(tol,max_ion_num)
    
    
def main():
#    file_name = 'data/AGGPGLER.mgf'
 #   file_name2 = 'data/HHAISAK.mgf'
    file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
   # spec1 = parse(file_name1)
    #spec2 = parse(file_name2)
#    sequence1 = 'AGGPGLER'
 #   sequence2 = 'HHAISAK'
  #  peptide1 = setPeptide(sequence1)
   # peptide2 = setPeptide(sequence2)
    #spec1.setAnnotation(peptide1)
#    spec2.setAnnotation(peptide2)
 #   print spec1
 #   filter = WindowFilter(6,40)
#    fspec = filter.filtered(spec1)
  #  print fspec
    parser = SpectrumParser()
    specs = []
    parser = parser.readSpectrum(file_name)
    for i in range(1):
        spec = None
        spec =  parser.next()
        print spec.getTitle()
        specs.append(spec)
#    specs = list(parser.readSpectrum(file_name))
#    charge = spec1.getCharge()
    ist = setIonSelector()
    sig_ions =  ist.findSigIons(2,specs)
    for ion,freq in sig_ions.items():
        print ion,':',ion.getPosition(),':',freq

if __name__ == '__main__':
    main()
