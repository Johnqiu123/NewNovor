# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:32:46 2016

@author: Johnqiu
"""
"""
问题1：真实的数据是按大小的顺序排列，模拟数据需不要排序？

"""
from scripts import ProteinWeightDict, IonTypeDict
import random
from operator import itemgetter

aa_table = ProteinWeightDict()
ion_table = IonTypeDict()

def simulatePeptide(pep_len):
    acids = [k for k in aa_table]
    peptide=[random.choice(acids) for i in range(pep_len)]
    peptide = ''.join(peptide)  # change list to String
    return peptide
    
def generateSpectrum(peptide, ion_table,intensity = 100):
    """
    Args:
       -peptide:   a peptide string
       -iontables:   {ion:(offset,prob)}
    """
    spectrum = []
    prefix_mass = 0
    all_mass = 0
    
    for a in peptide:
        all_mass = all_mass + float(aa_table[a])
        
    for acid in peptide:
        prefix_mass = prefix_mass + float(aa_table[acid])
        i = 0
        for key in ion_table:
            if i < 5: # N-term
                position = prefix_mass + float(ion_table[key][0])
            else: # C-term
                position = all_mass + 20 - (prefix_mass + 1) + float(ion_table[key][0])
            
            i += 1 
            prob = random.random()
            expect_prob = float(ion_table[key][1])
            if(prob < expect_prob):
                spectrum.append((round(position,5), intensity * expect_prob))
    return spectrum
    
def generateNoiseSpectrum(peptide, noise_num, intensity = 100):
    pep_mass = 0
    for amino in peptide:
        pep_mass = pep_mass + aa_table[amino]
        
    noise_spectrum = []
    for i in range(noise_num):
        prob = random.random()
        noise_spectrum.append((round(pep_mass*prob,5), round(intensity * prob,1)))
    return noise_spectrum
    
    
def generateSpectra(ion_table, pep_len, num):
    spectra = {}
    for i in range(num):
        peptide = simulatePeptide(pep_len)
        spectrum = generateSpectrum(peptide, ion_table, 100)
        noise_spectrum = generateNoiseSpectrum(peptide, pep_len, 100)
        spectrum.extend(noise_spectrum)
        spectrum = sorted(spectrum, key=itemgetter(0))
        spectra[peptide] = spectrum
    return spectra
    

def WriteToFile(spectra, ion_table, filename):
    output_file = "data/" + filename
    with open(output_file,'w') as output_data:
        i = 0
        for pep in spectra:
            pep_mass = 0
            for acid in pep:
                pep_mass = pep_mass + aa_table[acid]
            output_data.write('BEGIN IONS'+'\n')
            output_data.write('TITLE='+filename+'_'+str(i)+'\n')
            output_data.write('SEQ='+pep+'\n')
            output_data.write('PEPMASS='+str(pep_mass)+'\n')
            output_data.write('CHARGE=' + '1+'+'\n')
            for poi,intensity in spectra[pep]:
                output_data.write(str(poi)+' '+str(intensity)+'\n')
            output_data.write('END IONS'+'\n')              
            i += 1

if __name__=="__main__":
    spectra = generateSpectra(ion_table, 7, 2000)
    WriteToFile(spectra,ion_table,"SimulateData")