from SubSpectrumGenerator import SubSpectrumGenerator
from NoiseSubSpectrumParser import NoiseSubSpectrumParser
from SpectrumParser import SpectrumParser
from SubSpectrum import SubSpectrum
import time
import numpy as np
'''
  Process raw data by adding their corresponding peptide annotations.
  created on Sep 22, 2015 by mht,Johnqiu'''
  
class write_file(object):
    def rewriteFile(self):
       ## all peptide in simple.peplist
      file_name = 'data/CHPP_LM3_RP3_2.mgf'
      parser = SpectrumParser()
      specs = list(parser.readSpectrum(file_name))
      with open('data/simple.peplist') as pep_file:
          title_peptide = {line.strip().split(' ')[0]:line.strip().split(' ')[1] \
          for line in pep_file.readlines()}
      # rewrite data
      with open('output','w') as output_data:
          for spec in specs:
              title = spec.getTitle()
              if title in title_peptide:
                  output_data.write('BEGIN IONS'+'\n')
                  output_data.write('TITLE='+title+'\n')
                  output_data.write('SEQ='+title_peptide[title]+'\n')
                  output_data.write('PEPMASS='+str(spec.getPrecursorPeak().getMass())+'\n')
                  output_data.write('CHARGE='+str(spec.getCharge())+'+'+'\n')
                  for peak in spec.getPeaks():
                      if peak.getIntensity() != 0:
                          output_data.write(str(peak.getMz())+' '+str(peak.getIntensity())+'\n')
                          output_data.write('END IONS'+'\n')
                          
    def writeSubSepc(self, binlen=0.5, arealen=50):
      file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
      parser = SpectrumParser()
      specs = list(parser.readSpectrum(file_name))
      subparser = SubSpectrumGenerator()
      subspecs = list(subparser.generateSubSpectra(specs, binlen, arealen))
      filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2"
#      allbins = np.zeros(int(arealen*2/binlen))
      with open(filename,'w')  as  data:
              for subspec in subspecs:
                  data.write('[Title='+ subspec.getTitle() + ','\
                             +'POSITION='+ str(subspec.getPosition())+ ','\
                             +'LENGHT='+str(len(subspec.getAnnotation().getAcids()))+','\
                             +'LeftAA='+subspec.getLA()+','\
                             +'RightAA='+subspec.getRA()+','\
                             +'BIN_LENGTH='+str(subspec.getBinLength())+','\
                             +'BIN_RANGE='+str(subspec.getBinArea())+','\
                             +'NTermBIN= '+str(subspec.getNtermBins())+','\
                             +'CTermBIN='+str(subspec.getCtermBins())+']'+'\n')
#                  allbins += np.array(subspec.getBins())
#              data.write('[allbins='+str(allbins)+']'+'\n')
    
    def writeNoiseSubSepc(self, num=3, binlen=0.5, arealen=50):
      file_name = 'data/new_CHPP_LM3_RP3_2.mgf'
      parser = SpectrumParser()
      specs = list(parser.readSpectrum(file_name))
      noisesubparser = SubSpectrumGenerator()
      noisesubspecs = list(noisesubparser.generateNoiseSubSpectra(specs, num, binlen, arealen))
      print(len(noisesubspecs))
      filename = "SubSpectrumData/"+"new_CHPP_LM3_RP3_2_Noise"
#      allbins = np.zeros(int(arealen*2/binlen))
      with open(filename,'w')  as  data:
              for noisesubspec in noisesubspecs:
                  data.write('[TITLE='+ noisesubspec.getTitle() + ','\
                             +'PositionMZ='+ str(noisesubspec.getPositionmz())+ ','\
                             +'ParentMass='+ str(noisesubspec.getAnnotation().getParentMass())+ ','\
                             +'NTermBIN= '+str(noisesubspec.getNtermBins())+','\
                             +'CTermBIN='+str(noisesubspec.getCtermBins())+']'+'\n')
#                  allbins += np.array(subspec.getBins())
#              data.write('[allbins='+str(allbins)+']'+'\n')
 
#with open('output','r') as final_data:
 #   lines = [line.strip() for line in final_data.readlines()]
  #  for line in lines: print line

if __name__ == '__main__':
    fw = write_file()
#    fw.rewriteFile()
    start = time.clock()
#    fw.writeSubSepc()
    fw.writeNoiseSubSepc(8)
    end = time.clock()
    print 'time consuming %s seconds.' % (end-start)

