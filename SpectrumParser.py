from Spectrum import Spectrum
from Peak import Peak
import time
from Paint_File import Paint_File 
''' parse spectrum file with mgf format.
   created on Jul 20, 2015.
'''
class SpectrumParser(object):
    def readSpectrum(self,input_file):
        '''parses a spectrum file with mgf format.
           created on Aug 31, 2015 by mht.'''
     #   spec = None
      #  title = None
        is_parser = False
        with open(input_file) as input_data:
            for line in input_data:
                if len(line) == 0: continue
#                if line.startswith('MASS'): continue
                if line.startswith('BEGIN IONS'):
                    is_parser = True
                    spec = Spectrum()
                elif(is_parser):
                    if line.startswith('TITLE'):
                        title = line[line.index('=')+1:].strip()
                        spec.setTitle(title)
                    elif line.startswith('SEQ'):
                        annotation = line[line.index('=')+1:].strip()
                        if spec.getAnnotation() is None:
                            spec.setAnnotation(annotation)
                    elif line.startswith('PEPMASS'):
                        pep_str = line[line.index('=')+1:].strip()
                        pep_str =  pep_str.split(' ')
                        pre_mass = float(pep_str[0])
                    elif line.startswith('CHARGE'):
                        charge = line[line.index('=')+1:line.index('+')].strip()
                        pre_charge = int(charge)
                    elif line[0].isdigit():
                        mass,intensity = map(float,line.split(' '))
                        spec.addPeak(Peak(1,mass,intensity))
                    
                    elif line.startswith('END IONS'):
                        assert(spec is not None)
                        spec.setPrecursor(Peak(pre_charge,pre_mass*pre_charge,1))
                           # spec.sortPeaks()                        
                        yield spec # return spec, replace return
#        return None

if __name__ == '__main__':
    start = time.clock()
##    file_name = 'data/SBQ_IP_HeLa818_100per_load.raw' #
##    len(specs)=24694,time=69.08 seconds
##    file_name = 'data/AGGP.mgf'
#    file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
##    len(specs) = 18051,time = 32.14 seconds
#    parser = SpectrumParser()
#    specs = list(parser.readSpectrum(file_name))
#    k = parser.readSpectrum(file_name)
#    print specs[0]
#    print k.next()
#    end = time.clock()
#    print len(specs)
#    print 'time consuming %s seconds.' % (end-start)


################################ Test 1#####################################
    file_name = 'data/1_3M_JD_sample1_A.mgf'
    parser = SpectrumParser()
    specs = list(parser.readSpectrum(file_name))
#    print specs[0].getPeaks()
 
    xaix = []
    yaix = []   
    for peak in specs[0].getPeaks():
        xaix.append(peak.getMz())
        yaix.append(peak.getIntensity())

    pt = Paint_File()
    pt.paint(xaix,yaix)


    end = time.clock()
#    print len(specs)
    print 'time consuming %s seconds.' % (end-start)

        

    

