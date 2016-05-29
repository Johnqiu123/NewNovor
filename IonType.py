from scripts import Composition
import re

class IonType(object):
    '''IonType class is the abstract of all ion types.
       created on Sep 9, 2015 by mht.
    '''    
    def __init__(self, name='', charge=0, offset=1.0,pos=0):
        self.name = name
        self.charge = charge
        self.offset = offset
        self.pos = pos
    def __str__(self):
        return '%s/%s/%s' % (self.name,self.charge,self.offset)

    def __cmp__(self, other):
        if self.getCharge() < other.getCharge():
            return -1
        elif self.getCharge() > other.getCharge():
            return 1
        elif self.getOffset() < other.getOffset():
            return -1
        elif self.getOffset() > other.getOffset():
            return 1
        return 0

    def getName(self): return self.name
    def getCharge(self): return self.charge
    def getOffset(self): return self.offset
    def getPosition(self): return self.pos

    def getMass(self,mz):
        return (mz-self.offset)*self.charge
    def getMz(self,mass):
        return mass/self.charge+self.offset
    def isPrefixIon(self): return isinstance(self, PrefixIon)    
    def isSuffixIon(self): return isinstance(self,SuffixIon)
    def setPosition(self,pos): self.pos = pos
    
class PrefixIon(IonType):
    def __init__(self,name='',charge=0, offset=0.0,pos=0):
        if name == '':
            IonType.__init__(self,'p',charge,offset,pos)
        else:
            IonType.__init__(self,name,charge,offset,pos)
#    def isPrefixIon(self): return True

class SuffixIon(IonType):
    def __init__(self, name='', charge=0, offset=0.0,pos=0):
        if name == '':
            IonType.__init__(self,'s',charge,offset,pos)
        else:
            IonType.__init__(self,name,charge,offset,pos)

#   def isPrefixIon(self):return False

class PrecursorIon(IonType):
    def __init__(self, charge, offset,pos=0):
        IonType.__init__(self,'r_',charge,offset,pos=0)

def getIonType(name):
    '''returns ion type from a string such as follows,
      Ion name format: t/c/o, t=[sp] (s:SuffixIon, p:PrefixIon, i:internalIon,r:PrecursorIon), c=charge,o=offset
      or
      Ion name format: [abcxyz][+-]c, (c=[H,H2,H2O,NH3,NH] or c=offset)
      examples: y2-12.02, a+1.002-H2O, i/2/+1.23, s/1/-22.11, b-H2O-NH3
      returns None if format is not valid or ion does not exist.
    '''
    if name is None or len(name) == 0:  return None
    if name.startswith('s/') or name.startswith('p/') or name.startswith('r/'):
        t,charge,offset = name.split('/')
        if name.startswith('s'):
            it = SuffixIon('',charge,offset)
        elif name.startswith('p'):
            it = SuffixIon('',charge,offset)
        else:
            it = PrecursorIon('',charge,offset)
        return it
    
    # or format a-NH3-H20
    if len(name) <= 2: # b, or y,b2,y2
        base, offs = name, 0.0
    else:
        base,offs = re.split('[+-]',name,1)
        offs = float(compositionOffsetTable.get(offs))
    base = ionTable.get(base)
    if '+' in name:
        sign = 1
    else:
        sign = -1
    offs = sign*offs

    if isinstance(base, PrefixIon):
        it = PrefixIon(name,base.getCharge(),base.getOffset()+offs/base.getCharge())
    elif isinstance(base, SuffixIon):
        it =  SuffixIon(name,base.getCharge(),base.getOffset()+offs/base.getCharge())
    else : it = None
    return it

def getAllKnownIonTypes(max_charge):
    base = list('yb')
#    print base
    extension = ['','-H2O','-NH3','-H2O-H2O','-H2O-NH3']
    ions = []
    for charge in range(1,max_charge+1):
        for b in base:
            for e in extension:
                if charge == 1:
                    ion = getIonType(b+''+e)
                else:
                    ion = getIonType(b+str(charge)+e)            
                if ion is not None:
                    ions.append(ion)
    ions = sorted(ions)
    return ions
                
B = PrefixIon('b',1,float(Composition.OFFSET_B))
Y = SuffixIon('y',1,Composition.OFFSET_Y)
NOISE = PrefixIon('noise',0,0)
#print isinstance(B, PrefixIon)

ionTable = {}
ionTable = dict(zip('yb',[Y,B]))
for charge in range(2,3): # charge=2+
    ionTable['y'+str(charge)] = \
        SuffixIon('y'+str(charge),charge,(float)((Y.offset+Composition.PROTON*(charge-1))/charge))
    ionTable['b'+str(charge)] = \
        PrefixIon('b'+str(charge),charge,(float)((B.offset+Composition.PROTON*(charge-1))/charge))
    
compositionOffsetTable = dict()
compositionOffsetTable['H2O']=Composition.H2O
compositionOffsetTable['NH3'] = Composition.NH3
compositionOffsetTable['H2O-H2O'] = Composition.H2O+Composition.H2O
compositionOffsetTable['H2O-NH3'] = Composition.H2O+Composition.NH3

if __name__ == '__main__':
    #for key, value in compositionOffsetTable.items():
    #    print key, value
    y =  getIonType('y-NH3')
    b =  getIonType('b')
    ions =  getAllKnownIonTypes(2)
    print 'ion type', '\t', 'ion offset'
    for ion in ions:
        print ion.getName(), '\t', ion.getOffset()
    print len(ions)
 
