# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:30:54 2015

@author: mht , Johnqiu

note ： 计算结果有可能会超出int的范围，改成长整型计算
"""
import numpy as np 
from fractions import Fraction 
import time
import itertools
import math as math

class ChiSquared(object):
    
    def ComputeRow(self,n,rows, columns):
        '''compute a row of a table.
        Args:
          -ni : the sum of the elements in the i-th row
          -probs : list of probabilities
        '''
        row = [ni*nj/n for ni in rows for nj in columns]
        return row
        
    def AddList(self,xs,ys):
        '''add tow list of values'''
        t = []
        t += [x+y for x,y in zip(xs,ys)]
        return t

    def ComputeColumns(self,rows):
        '''compute the sum of rows of lists'''
#        nr = np.array(rows)
        return list(reduce(lambda u,v:u+v,rows))
        
    def ComputeExpRow(self,n,ni,nj):
        '''compute the prob p_ij given the sum of i-th row and j-th column.
        args:
          ni-the sum of i-th row
          nj-the sum of j-th column
        returns:
        the prob p_ij of the table.
        '''
        return float(ni)*nj / n
    
    def ComputeExpRows(self,n,row,column):
        result = []
        for i in row:
            result.append([self.ComputeExpRow(n,i,j) for j in column])
        return result
    
    def ComputeCS(self,expected,observed):
        '''compute the chi-squared statistic for two lists.
        args:
           -expected: list of lists of values
           -observed: list of lists of values
        returns:
           float, chi-squared statistic.
        '''
#        print "ComputeCS"
        it = zip(itertools.chain(*expected),itertools.chain(*observed))
        t = [(obs-exp)**2 / exp for exp,obs in it]
        return sum(t)

    def ComputeAdjustCS(self,expected,observed):
        '''compute the adjusted chi-squared statistic for two lists.
           chi-squared = (math.fabs(A-T)-0.5)**2 / T
        args:
           -expected: list of lists of values
           -observed: list of lists of values
        returns:
           float, chi-squared statistic.
        '''
        print "ComputeAdjustCS"
        it = zip(itertools.chain(*expected),itertools.chain(*observed))
        t = [(math.fabs(obs-exp)-0.5)**2 / exp for exp,obs in it]
        return sum(t)
 
    def CompBorderval(self, obs):
        """
        compute border values
        
        Args:
          -obs : ndarray of observers
         
        return:
           - xbor : a ndarray of sum of columns' values
           - ybor : a ndarray of sum of rows' values
           
        """
        xbor = np.array([obs[:,i].sum() for i in range(obs.shape[1])])
        ybor = np.array([obs[i].sum() for i in range(obs.shape[0])])
        
#        print xbor,ybor
        
        return xbor,ybor
    
    def JudgeTN(self, expect, N):
        """
        Judge the range of expected values and number N
        """
        newexpect =  np.ravel(expect)
        T1 = len(filter(lambda x: 1<=x<5, newexpect)) 
        T0 = len(filter(lambda x: x<1, newexpect)) 
        
#        print T1,T0
        if N < 40 or T0 > 0:
            print "please input more data"
            return 2
        if N >= 40:
            if T1 > 0:
                return 1
            return 0
        
    
    def OrgainChiSquard(self,obs):
        """
        compute a table
        Args:
          -obs: ndarray of observers

        Return:
          -CHiValues: the value of ChiSquared
        """
        expected =[]
        row = [sum(xs) for xs in obs]
        n = obs.sum()
        column = self.ComputeColumns(obs)
#        print row,column,n
        expected = self.ComputeExpRows(n,row,column)
#        print expected
        flag = self.JudgeTN(expected, n)
        ChiValues = 0 
        if flag == 0:
            ChiValues =  self.ComputeCS(expected,obs)
        elif flag == 1:
            ChiValues =  self.ComputeAdjustCS(expected,obs)
#        else:
#            ChiValues =  self.ComputeAdjustCS(expected,obs)
        
#        print row
#        print column
#        print expected
#        print ChiValues
        return ChiValues
    
    def EasyChiSquared(self, obs):
        """
        compute a table which rows = columns = 2
        Args:
           -obs: ndarray of observers
        
        Return:
          -CHiValues: the value of ChiSquared
        """
        if (2,2) != obs.shape:
            print("you can not use this function")
            return
        sumValues = obs.sum()
        crossValues = (obs[0][0] * obs[1][1]-obs[0][1] * obs[1][0])**2
        xbor,ybor = self.CompBorderval(obs)
        print xbor,ybor
        
        borderValues = 1L
        for x in np.hstack((xbor,ybor)):
            borderValues = borderValues * x

        ChiValues = float(Fraction(long(sumValues) * long(crossValues), borderValues))
        
#        ChiValues = float(Fraction(1000 * 284934400,480*520*956*44)) 
        
        
        
        print sumValues,crossValues,borderValues
#        print 480*520*956*44
               
#        borderValues = np.hstack((xbor,ybor)).cumprod()[-1] 
#        borderValues = obs[0].sum() * obs[1].sum() * obs[:,0].sum() * obs[:,1].sum() 
#        print obs[0].sum(),obs[1].sum(),obs[:,0].sum(),obs[:,1].sum(),borderValues
#        print obs[0][0],obs[1][1],obs[0][1],obs[1][0],crossValues,sumValues
        return ChiValues
    
    def ComplexChiSquared(self, obs):
        """
        compute a table
        Args:
          -obs: ndarray of observers

        Return:
          -CHiValues: the value of ChiSquared
        """
        sumValues = obs.sum()
        xbor,ybor = self.CompBorderval(obs)
        temp = [Fraction(long(obs[i][j]**2),long((xbor[j]*ybor[i]))) for j in range(obs.shape[1]) \
               for i in range(obs.shape[0])]             
        ChiValues = float(sumValues * (sum(temp) -1))
        return ChiValues

if __name__ == '__main__':
    chi = ChiSquared()
    table = np.array([[18,12],[4,78]])
    table2 = np.array([[24,23,12],[24,14,10],[17,8,13],[27,19,9]])
    table3 = np.array([[442,514],[38,6]])
    
    table4 = np.array([[52553,87025],[3687,3628]])
    table5 = np.array([[8888,5,444,7887,998],[3687,0,123,1222,3444]])
    
    table8 = np.array([[52,19],[39,3]])
    table9 = np.array([[26,7],[36,2]])
    
    table10 = np.array([[180,14,120,65],[200,16,84,33]])
    table10 = np.array([[180,14,120,65],[1,0,1,0]])
    
#    result = chi.EasyChiSquared(table)
#    result2 = chi.ComplexChiSquared(table2)
#    result3 = chi.EasyChiSquared(table3)
#    result4 = chi.ComplexChiSquared(table3)
#    result5 = chi.EasyChiSquared(table4)
#    result6 = chi.ComplexChiSquared(table4)
#    result7 = chi.OrgainChiSquard(table5)
#    result8 = chi.OrgainChiSquard(table8)
#    result9 = chi.OrgainChiSquard(table9)
    result10 = chi.OrgainChiSquard(table10)
#    start = time.clock()
#    result3 = chi.EasyChiSquared(table3)
#    end = time.clock()
#    print 'time consuming %s seconds.' % (end-start)
#    
#    start2 = time.clock()
#    result4 = chi.ComplexChiSquared(table3)
#    end2 = time.clock()
#    print 'time consuming %s seconds.' % (end2-start2)
    
    
#    print table2.shape
#    print result
#    print result2
#    print result3
#    print result4
#    print result5
#    print result6
#    print result7
#    print result8
#    print result9
    print result10
    