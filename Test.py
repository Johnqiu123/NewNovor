# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:57:00 2016

@author: Johnqiu
"""
import numpy as np
import random
filename = "D:/Pythonlab/NewNovo/data/1_3M_JD_sample1_A.mgf"
print type(filename)
file_name = filename.split('/')[-1]
new_file = "data/"+file_name

fname = file_name.split('.')[0]
print file_name 
print new_file
print fname

label = []
for i in range(10):
    label += [i]*10 
label = np.array(label)

a = np.array([[0,1,1,4],[2,3,4,5],[1,34,5,5]])
b = np.array([0,1,4,2])
k3 = np.array([[2],[0],[3]])
c = b[np.nonzero(a!=b)[0]]

k4 = [13,4,5]
np.random.permutation(a)
print a,b,c,k4

d = np.vstack((a,b))
print d 

k2 = np.hstack((a,k3))
print k2

T=0
if 1<=T<4:
    print True
    
result = [[28.816901408450704, 4.183098591549296], \
          [33.183098591549296, 4.816901408450704]]
k = 0
for i in result:
    print i
    if 1 <= i and i < 5 :
        k += 1
print "k:",k
dd = len(filter(lambda x: x>6, result))
print dd

import math
print math.fabs(2)

####################################################
import pandas as pd

Adict = {}
B = [1,2,3,4]
A = Adict.fromkeys(B,(0,0)) 
print "A:",A
A[1] = (0,3)
A[2] = (3,5)
A[3] = (9,1)
print A
poiA = pd.DataFrame(A)
print poiA
TabelA = np.array([poiA.iloc[0],poiA.iloc[1]])
print TabelA

for i in range(9):
    print i
    
print int(1.5)