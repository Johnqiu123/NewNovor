# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:42:32 2016

@author: Johnqiu
"""
"""
采用Grid布局

好处：容易设计，方便快捷
坏处：除非你已经有完整的设计，否则容易牵一发而动全身，修改很麻烦

每个模块对应着自己独立的标签

传入参数方法
command=lambda: sockOpen(root)

问题：卡方检验的计算
"""
import Tkinter as tk
import tkFileDialog

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from write_file import write_file
from SubSpectrumProcessor import SubSpectrumProcessor
from SubSpectrumGenerator import SubSpectrumGenerator
from SpectrumParser import SpectrumParser
from PeptideProcessor import PeptideProcessor
from IonGroupLearner import IonGroupLearner
from SpectrumProcessor import SpectrumProcessor
import os
fw = write_file()
class showproject(object):
    
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Show Project")
#        self.root.geometry('470x320')
        
        # menu
        self.menu = tk.Menu(self.root)
        # menu initial
        self.menuTest() 
        self.root.config(menu=self.menu)
        
        # drawpiture
        self.dw_f = Figure(figsize=(5,4), dpi=80) #create canvas
        self.dw_canvas = FigureCanvasTkAgg(self.dw_f, master=self.root)
        self.dw_canvas.get_tk_widget().grid(row=0, columnspan=5)
        

    def menuTest(self):
        """
          define menu 'Test'
          
          问题：添加命令时，会自动调用方法
        """
        test_menu = tk.Menu(self.menu, tearoff=0)
        test_menu.add_command(label="rewrite file", command= self.rewrite)
        test_menu.add_command(label="Test 1", command= self.Test1)
        test_menu.add_command(label="Test 2", command= self.Test2)
        test_menu.add_command(label="Test 3", command=self.Test3)
        test_menu.add_command(label="Test 4", command=self.Test4)
        test_menu.add_command(label="Test 5", command=self.Test5)
        test_menu.add_command(label="Test 6", command=self.Test6)
        test_menu.add_command(label="Test 7", command=self.Test7)
        test_menu.add_command(label="Test 8", command=self.Test8)
        test_menu.add_command(label="Test 9", command=self.Test9)
#        test_menu.add_command(label="Test 10", command=self.Test10)
#        test_menu.add_command(label="Test 11", command=self.Test11)
        test_menu.add_command(label="Test 12", command=self.Test12)
#        test_menu.add_command(label="Test 13", command=self.Test13)
        self.menu.add_cascade(label="Test", menu=test_menu)
    
    ########################################### Graph ###################################
    def rewrite(self):
        self.rebuildroot()
        print "rewrite"
        tk.Label(self.root, text="-"*40+"rewrite file"+"-"*40)\
                .grid(row=1, column=0, columnspan=5)
        self.rw_IPFbutton = tk.Button(self.root, text="dataFile", command=self.rw_loadIPFile)
        self.rw_PLFbutton = tk.Button(self.root, text="peplistFile", command=self.rw_loadPLFile)
        tk.Label(self.root, text="outputFile :").grid(row=3, column=1)
        self.rw_OpFEntry = tk.Entry(self.root) 

        self.rw_Actbutton = tk.Button(self.root, text="action", command=self.rw_action)
        
        self.rw_IPFbutton.grid(row=2,column=1)
        self.rw_PLFbutton.grid(row=2,column=3)       
        self.rw_OpFEntry.grid(row=3, column=2, columnspan=2)
        self.rw_Actbutton.grid(row=3, column=4)
        
    def Test1(self):
        self.rebuildroot()
#        print dir(self)
        
        print "Test1"
        # test 1
        tk.Label(self.root, text="-"*40+"   Test 1 Generate SubSpectrum   "+"-"*40)\
                .grid(row=1, column=0, columnspan=5)
        tk.Label(self.root, text="binlen :").grid(row=2, column=0)
        tk.Label(self.root, text="arealen :").grid(row=2, column=2)
        tk.Label(self.root, text="flag :").grid(row=3, column=0)

        
        self.t1_BlEntry = tk.Entry(self.root)
        self.t1_AlEntry = tk.Entry(self.root) 
        self.t1_FlagEntry = tk.Entry(self.root) 
        self.t1_Fnbutton = tk.Button(self.root, text="filename", command=self.t1_loadFile)
        self.t1_Actbutton = tk.Button(self.root, text="action", command=self.t1_action)
        
        self.t1_BlEntry.grid(row=2, column=1)
        self.t1_AlEntry.grid(row=2, column=3)
        self.t1_FlagEntry.grid(row=3, column=1)
        self.t1_Fnbutton.grid(row=3, column=2)
        self.t1_Actbutton.grid(row=3, column=4)

        tk.Label(self.root, text="Noise:")\
                .grid(row=4, column=0, columnspan=5)
        tk.Label(self.root, text="Noibinlen :").grid(row=5, column=0)
        tk.Label(self.root, text="Noiarealen :").grid(row=5, column=2)
        tk.Label(self.root, text="Noiflag :").grid(row=6, column=0)
        tk.Label(self.root, text="Noinum :").grid(row=6, column=2)
        
        self.t1_NBlEntry = tk.Entry(self.root) 
        self.t1_NAlEntry = tk.Entry(self.root) 
        self.t1_NFlagEntry = tk.Entry(self.root) 
        self.t1_NnumEntry = tk.Entry(self.root)
        self.t1_NFnbutton = tk.Button(self.root, text="Noifilename", command=self.t1_loadNFile)
        self.t1_NActbutton = tk.Button(self.root, text="action", command=self.t1_noiaction)
        
        self.t1_NBlEntry.grid(row=5, column=1)
        self.t1_NAlEntry.grid(row=5, column=3)
        self.t1_NFlagEntry.grid(row=6, column=1)
        self.t1_NnumEntry.grid(row=6, column=3)
        self.t1_NFnbutton.grid(row=7, column=0)
        self.t1_NActbutton.grid(row=7, column=4)
    
    def Test2(self):    
        """
          Test 2: calculate Bins
        """
        self.rebuildroot()

        print "Test2"
        tk.Label(self.root, text="-"*40+"   Test 2  calculateBins  "+"-"*40)\
                .grid(row=1, column=0, columnspan=5)
        tk.Label(self.root, text="calculate type:").grid(row=2, column=1)
        self.v = tk.StringVar()
        self.v.set('num')
        self.t2_01Ratio = tk.Radiobutton(self.root, variable = self.v, value = 'num', text = '01')
        self.t2_IntRatio = tk.Radiobutton(self.root, variable = self.v, value = 'intensity', text = 'Intensity')  
        
        tk.Label(self.root, text="subspectrum file:").grid(row=3, column=1)
        self.t2_Fnbutton = tk.Button(self.root, text="subspectFile", command=self.t2_loadFile)
        
        tk.Label(self.root, text="show picture:").grid(row=4, column=1)
        self.pic = tk.IntVar()
        self.pic.set(0)
        self.t2_NRatio = tk.Radiobutton(self.root, variable = self.pic, value = 0, text = 'allNtermbins')
        self.t2_CRatio = tk.Radiobutton(self.root, variable = self.pic, value = 1, text = 'allCtermbins')  
        self.t2_AllRatio = tk.Radiobutton(self.root, variable = self.pic, value = 2, text = 'allSubbins')

        
        self.t2_Actbutton = tk.Button(self.root, text="action",command=self.t2_action) 
#        tk.Label(self.root, text="calculate type:").grid(row=2, column=1)
        
        self.t2_01Ratio.grid(row=2, column=2)
        self.t2_IntRatio.grid(row=2, column=3)
        self.t2_Fnbutton.grid(row=3, column=2)
        self.t2_NRatio.grid(row=4, column=2)
        self.t2_CRatio.grid(row=4, column=3)
        self.t2_AllRatio.grid(row=4, column=4)
        self.t2_Actbutton.grid(row=5, column=2)
    
    def Test3(self):
        self.rebuildroot()
        
        print "Test3"
        tk.Label(self.root, text="-"*40+"   Test 3  TypeandBreakPoint "+"-"*40)\
                .grid(row=1, column=0, columnspan=5)
                
        tk.Label(self.root, text="subspectrum file(01):").grid(row=2, column=1)
        self.t3_Fnbutton = tk.Button(self.root, text="subspectFile", command=self.t3_loadFile)
        self.t3_NFnbutton = tk.Button(self.root, text="NoisubspectFile", command=self.t3_loadNFile)
        
        tk.Label(self.root, text="show picture:").grid(row=3, column=1)
        self.pic = tk.IntVar()
        self.pic.set(0)
        self.t3_NRatio = tk.Radiobutton(self.root, variable = self.pic, value = 0, text = 'NchiValues')
        self.t3_CRatio = tk.Radiobutton(self.root, variable = self.pic, value = 1, text = 'CchiValues')  
        self.t3_AllRatio = tk.Radiobutton(self.root, variable = self.pic, value = 2, text = 'chiValues')

        
        self.t3_Actbutton = tk.Button(self.root, text="action",command=self.t3_action) 
#        tk.Label(self.root, text="calculate type:").grid(row=2, column=1)
        
        self.t3_Fnbutton.grid(row=2, column=2)
        self.t3_NFnbutton.grid(row=2, column=3)
        self.t3_NRatio.grid(row=3, column=2)
        self.t3_CRatio.grid(row=3, column=3)
        self.t3_AllRatio.grid(row=3, column=4)
        self.t3_Actbutton.grid(row=4, column=2)
    

    def Test4(self):
        self.rebuildroot()
        print "Test4"
        tk.Label(self.root, text="-"*40+"   Test 4  TypeandAminoPairs "+"-"*40)\
                .grid(row=1, column=0, columnspan=5)
                
        tk.Label(self.root, text="subspectrum file(01):").grid(row=2, column=1)
        self.t4_Fnbutton = tk.Button(self.root, text="subspectFile", command=self.t4_loadFile)
        self.t4_NFnbutton = tk.Button(self.root, text="NoisubspectFile", command=self.t4_loadNFile)
    
        
        self.t4_Actbutton = tk.Button(self.root, text="action",command=self.t4_action) 
#        tk.Label(self.root, text="calculate type:").grid(row=2, column=1)
        
        self.t4_Fnbutton.grid(row=2, column=2)
        self.t4_NFnbutton.grid(row=2, column=3)
        self.t4_Actbutton.grid(row=3, column=2)

    def Test5(self):
        self.rebuildroot()
        print "Test5"
        tk.Label(self.root, text="-"*40+"   Test 5  PepLenTable  "+"-"*40)\
                .grid(row=1, column=0, columnspan=5)
        
        self.t5_Fnbutton = tk.Button(self.root, text="originalFile", command=self.t5_loadFile)
        self.t5_Actbutton = tk.Button(self.root, text="action",command=self.t5_action) 
        
        self.t5_Fnbutton.grid(row=2, column=2)
        self.t5_Actbutton.grid(row=2, column=3)
        

    def Test6(self):
        self.rebuildroot()
        print "Test6"
        tk.Label(self.root, text="-"*40+"   Test 6  Pepbondpoi "+"-"*40)\
                .grid(row=1, column=0, columnspan=5)
        tk.Label(self.root, text="subspectrum file(01):").grid(row=2, column=1)
        self.t6_Fnbutton = tk.Button(self.root, text="subspectFile", command=self.t6_loadFile)
        tk.Label(self.root, text="splitnorm:").grid(row=3, column=1)
        self.splitflag = tk.StringVar()
        self.splitflag.set("length")
        self.t6_LRatio = tk.Radiobutton(self.root, variable = self.splitflag, value = "length", text = "length")
        self.t6_MRatio = tk.Radiobutton(self.root, variable = self.splitflag, value = "mass", text = "mass")  
        
        self.t6_Actbutton = tk.Button(self.root, text="action",command=self.t6_action) 
        
        self.t6_Fnbutton.grid(row=2, column=2)
        self.t6_LRatio.grid(row=3, column=2)
        self.t6_MRatio.grid(row=3, column=3)
        self.t6_Actbutton.grid(row=4, column=2)
    
    def Test7(self):
        self.rebuildroot()
        print "Test7"
        tk.Label(self.root, text="-"*40+"   Test 7  TypeandType  "+"-"*40)\
                .grid(row=1, column=0, columnspan=5)
        
        self.t7_Fnbutton = tk.Button(self.root, text="subspectrumFile(01)", command=self.t7_loadFile)
        self.t7_Actbutton = tk.Button(self.root, text="action",command=self.t7_action) 
        
        self.t7_Fnbutton.grid(row=2, column=2)
        self.t7_Actbutton.grid(row=2, column=3)   

    def Test8(self):
        self.rebuildroot()
        print "Test8"
        tk.Label(self.root, text="-"*40+"   Test 8  Specical subspectrum  "+"-"*40)\
                .grid(row=1, column=0, columnspan=5)
        
        self.t8_Fnbutton = tk.Button(self.root, text="spectrumFile", command=self.t8_loadFile)
        self.t8_Actbutton = tk.Button(self.root, text="action",command=self.t8_action) 
        
        self.stype = tk.StringVar()
        self.stype.set("dual")
        self.t8_dualRatio = tk.Radiobutton(self.root, variable = self.stype, value = "dual", text = "dual")
        self.t8_atypeRatio = tk.Radiobutton(self.root, variable = self.stype, value = "atype", text = "atype")  
        self.t8_yNH3Ratio = tk.Radiobutton(self.root, variable = self.stype, value = "yNH3", text = "yNH3")
        self.t8_yH2ORatio = tk.Radiobutton(self.root, variable = self.stype, value = "yH2O", text = "yH2O")  
        self.t8_bH2ORatio = tk.Radiobutton(self.root, variable = self.stype, value = "bH2O", text = "bH2O")
        self.t8_bNH3Ratio = tk.Radiobutton(self.root, variable = self.stype, value = "bNH3", text = "bNH3")  
        self.t8_y46_Ratio = tk.Radiobutton(self.root, variable = self.stype, value = "y46-", text = "y46-")
        self.t8_y45_Ratio = tk.Radiobutton(self.root, variable = self.stype, value = "y45-", text = "y45-")  
        self.t8_y10_Ratio = tk.Radiobutton(self.root, variable = self.stype, value = "y10+", text = "y10+")  
        tk.Label(self.root, text="binlen :").grid(row=6, column=0)
        tk.Label(self.root, text="arealen :").grid(row=6, column=2)
        self.t8_BlEntry = tk.Entry(self.root) 
        self.t8_AlEntry = tk.Entry(self.root) 
  
        self.t8_Fnbutton.grid(row=2, column=2)
        self.t8_dualRatio.grid(row=3, column=1)
        self.t8_atypeRatio.grid(row=3, column=2)
        self.t8_yNH3Ratio.grid(row=3, column=3)
        self.t8_yH2ORatio.grid(row=4, column=1)
        self.t8_bH2ORatio.grid(row=4, column=2)
        self.t8_bNH3Ratio.grid(row=4, column=3)
        self.t8_y46_Ratio.grid(row=5, column=1)
        self.t8_y45_Ratio.grid(row=5, column=2)
        self.t8_y10_Ratio.grid(row=5, column=3)
        self.t8_BlEntry.grid(row=6, column=1)
        self.t8_AlEntry.grid(row=6, column=3)  
        self.t8_Actbutton.grid(row=6, column=4)  
     

    def Test9(self):
        self.rebuildroot()
        print "Test9"
        tk.Label(self.root, text="-"*40+"   Test 9  Ion groups  "+"-"*40)\
                .grid(row=1, column=0, columnspan=5)
        
        tk.Label(self.root, text="SpectrumMaxIntentity：").grid(row=2, column=1)
        self.t9_SMFnbutton = tk.Button(self.root, text="spectraFile", command=self.t9_loadSMFile)
        self.t9_SMActbutton = tk.Button(self.root, text="action",command=self.t9_smaction) 
        
        tk.Label(self.root, text="SubspectrumFile(Intentiey)：").grid(row=3, column=1)
        self.t9_Fnbutton = tk.Button(self.root, text="subspectrumFile", command=self.t9_loadFile)   
        self.t9_Actbutton = tk.Button(self.root, text="action",command=self.t9_action) 
        
        self.t9_SMFnbutton.grid(row=2, column=2)
        self.t9_SMActbutton.grid(row=2, column=3) 
        self.t9_Fnbutton.grid(row=3, column=2)
        self.t9_Actbutton.grid(row=3, column=3) 
    
    def Test12(self):
        self.rebuildroot()
        print "Test12"
        tk.Label(self.root, text="-"*40+"  Test 12 Spectrum Sample  "+"-"*40)\
                   .grid(row=1, column=0, columnspan=5)
        tk.Label(self.root, text="Window number:").grid(row=2, column=1)
        tk.Label(self.root, text="Peak number:").grid(row=2, column=3)
        
        self.t12_WNEntry = tk.Entry(self.root) 
        self.t12_PNEntry = tk.Entry(self.root) 
        self.t12_SMFnbutton = tk.Button(self.root, text="dataFile", command=self.t12_loadSMFile)
        self.t12_Actbutton = tk.Button(self.root, text="action", command=self.t12_action)
       
        self.t12_WNEntry.grid(row=2, column=2)
        self.t12_PNEntry.grid(row=2, column=4)  
        self.t12_SMFnbutton.grid(row=3,column=1)       
        self.t12_Actbutton.grid(row=3, column=4)


    
    def rebuildroot(self):
        self.root.destroy()
        self.root = tk.Tk()
        self.root.title("Show Project")
        # menu
        self.menu = tk.Menu(self.root)
        # menu initial
        self.menuTest() 
        self.root.config(menu=self.menu)
    
        # drawpiture
        self.dw_f = Figure(figsize=(5,4), dpi=100) #create canvas
        self.dw_canvas = FigureCanvasTkAgg(self.dw_f, master=self.root)
        self.dw_canvas.show()
        self.dw_canvas.get_tk_widget().grid(row=0, columnspan=5)
        
    ########################################### Method ###################################    
    ################################## rewrite file #########################
    def rw_loadIPFile(self):
        """
        load the origin file
        """
        print "loadFile"
        filename = tkFileDialog.askopenfilename()
        filename = self.filenameparser(filename)
        self.rw_IPFile = filename
    
    def rw_loadPLFile(self):
        """
        load the origin file
        """
        print "loadFile"
        filename = tkFileDialog.askopenfilename()
        filename = self.filenameparser(filename)
        self.rw_PLFile = filename
    
    def rw_action(self):
        self.rw_OUTFile = self.rw_OpFEntry.get()
        if(hasattr(self,'rw_IPFile') and hasattr(self,'rw_PLFile') and self.rw_OUTFile!=""):
#            print self.rw_IPFile,self.rw_PLFile,self.rw_OUTFile
            mark = fw.rewriteFile(self.rw_IPFile,self.rw_PLFile,self.rw_OUTFile)
            if not mark:
                print "please input again"
            print "rewrite file finish      "
        
    ################################## Test 1 Generate SubSpectrum #########################
    def t1_loadFile(self):
        """
        load the subspectrum file
        """
        print "subspectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = self.filenameparser(filename)
        self.t1_filename = filename
        
    def t1_action(self):
        try:
            self.t1_binlen = float(self.t1_BlEntry.get())
            self.t1_arealen = int(self.t1_AlEntry.get())
        except:
            print "input number"
            return
            
        self.t1_flag = self.t1_FlagEntry.get()
        if(not hasattr(self,'t1_filename')):
            print "please choose file"
            return
        print self.t1_binlen,self.t1_arealen,self.t1_flag,self.t1_filename
        
        flags = ['num','intensity']
        
        if self.t1_flag in flags :
            fw.writeSubSepc(self.t1_filename,self.t1_binlen,self.t1_arealen,self.t1_flag)
            print "subspectrum file finish      "
        else:
            print "please input again"
 
            

    def t1_loadNFile(self):
        """
        load the noise subspectrum file
        """
        print "noisubspectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = self.filenameparser(filename)
        self.t1_noifilename = filename
        
    def t1_noiaction(self):
        try:
            self.t1_noibinlen = float(self.t1_NBlEntry.get())
            self.t1_noiarealen = int(self.t1_NAlEntry.get())
            self.t1_noinum = int(self.t1_NnumEntry.get())
        except:
            print "input number"
            return
            
        self.t1_noiflag = self.t1_NFlagEntry.get() 
        
        if(not hasattr(self,'t1_noifilename')):
            print "please choose file"
            return
        
        print self.t1_noibinlen,self.t1_noiarealen,self.t1_noiflag,self.t1_noinum,self.t1_noifilename
        
        flags = ['num','intensity']
        
        if self.t1_noiflag in flags :
            fw.writeNoiseSubSepc(self.t1_noifilename,self.t1_noinum,self.t1_noibinlen,self.t1_noiarealen,self.t1_noiflag)
            print "noise subspectrum file finish      "
        else:
            print "please input again"
        
    ################################## Test 2  calculateBins #########################
    def t2_loadFile(self):
        """
        load the subspectrum file
        """
        print "subspectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = "SubSpectrumData/" + self.filenameparser(filename)
        self.t2_filename = filename
    
    def t2_action(self):
        print "t2_action"
        if(not hasattr(self,'t2_filename')):
            print "please choose file"
            return
        
        subparser = SubSpectrumGenerator()
        subspects = list(subparser.generateSubSpecfile(self.t2_filename, self.v.get()))
        
        subprocessor = SubSpectrumProcessor()
        allNtermbins,allCtermbins,allSubbins,subNum = subprocessor.calculateBins(subspects)
        
        showpics = []
        showpics.append(allNtermbins)
        showpics.append(allCtermbins)
        showpics.append(allSubbins)
        
        subprocessor.paintSubSpects(showpics[self.pic.get()])
        
        print self.v.get()
        print self.pic.get()
        print self.t2_filename
        
    ################################## Test 3  TypeandBreakPoint #########################
    def t3_loadFile(self):
        """
        load the subspectrum file
        """
        print "subspectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = "SubSpectrumData/" + self.filenameparser(filename)
        self.t3_filename = filename

    def t3_loadNFile(self):
        """
        load the noisubspectrum file
        """
        print "noisubspectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = "SubSpectrumData/" + self.filenameparser(filename)
        self.t3_noifilename = filename
    
    def t3_action(self):
        print "t3_action"
        if(not hasattr(self,'t3_filename') and not hasattr(self,'t3_noifilename')):
            print "please choose file"
            return
        
        subparser = SubSpectrumGenerator()
        subspects = list(subparser.generateSubSpecfile(self.t3_filename))
        noisubspects = list(subparser.generateNoiSubfile(self.t3_noifilename))
        
        subprocessor = SubSpectrumProcessor()
        allNtermbins,allCtermbins,allSubbins,subNum = subprocessor.calculateBins(subspects)
        allNOiNtermbins,allNoiCtermbins,allNoibins,noiNum = subprocessor.calculateBins(noisubspects)
        
        chiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allSubbins,allNoibins)
        poiChiValues,poichiV = subprocessor.sortChiValues(chiValues)
        orginalpois = [poiChiValues[i][1] for i in range(len(poiChiValues))][0:21] # get top 21 chivalues
        # store orginalpois        
        fw.writeIonPoi(orginalpois,self.t3_filename)
        
        #     #n-term
        if self.pic.get() == 0:
            NchiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allNtermbins,allNOiNtermbins)
            subprocessor.paintChiValues(NchiValues)
        elif self.pic.get() == 1:
            #c-term
            CchiValues = subprocessor.ChiSquared_TypeandBreakPoint(subNum,noiNum,allCtermbins,allNoiCtermbins)
            subprocessor.paintChiValues(CchiValues)
        else:
            #all  
            subprocessor.paintChiValues(chiValues)
    

        print self.pic.get()
        print self.t3_filename
        print self.t3_noifilename
    
    ################################## Test 4  TypeandAminoPairs #########################
    def t4_loadFile(self):
        """
        load the subspectrum file
        """
        print "subspectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = "SubSpectrumData/" + self.filenameparser(filename)
        self.t4_filename = filename

    def t4_loadNFile(self):
        """
        load the noisubspectrum file
        """
        print "noisubspectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = "SubSpectrumData/" + self.filenameparser(filename)
        self.t4_noifilename = filename
    
    def t4_action(self):
        print "t4_action"
        if(not hasattr(self,'t4_filename') and not hasattr(self,'t4_noifilename')):
            print "please choose file"
            return
        
        peppro = PeptideProcessor()
        if(not hasattr(self,'orginalpois')):
            filename = self.t4_filename+"_IonPostion" 
            if not os.path.exists(filename):
                print "please operate Test3 Frist"
                return 
            self.orginalpois = peppro.generateIonPoitionFile(filename)
        
        subparser = SubSpectrumGenerator()
        subspects = list(subparser.generateSubSpecfile(self.t4_filename))
        
        subprocessor = SubSpectrumProcessor()
        ionapChiValues = subprocessor.ChiSquared_TypeandAminoPairs(subspects,self.orginalpois)
        fileName = self.t4_filename + "_typeAPChi"
        fw.writeFile_cp(fileName, ionapChiValues)
        
        print self.t4_filename
        print self.t4_noifilename
        
        
#        subprocessor.paintSubSpects(ionapChiValues)
        
    ################################## Test 5  PepLenTable #########################
    def t5_loadFile(self):
        """
        load the orginal file
        """
        print "OriginFile"
        filename = tkFileDialog.askopenfilename()
        filename = "data/" + self.filenameparser(filename)
        self.t5_filename = filename

    def t5_action(self):
        print "t5_action"
        if(not hasattr(self,'t5_filename')):
            print "please choose file"
            return
        
        parser = SpectrumParser()
        spects = list(parser.readSpectrum(self.t5_filename))
        peppro = PeptideProcessor()
       
        pepLendf = peppro.generatePepLenTable(spects)     
        peppro.paintPeplen(pepLendf)
#        print pepLendf     
        print self.t5_filename
    ################################## Test 6  Pepbondpoi #########################
    def t6_loadFile(self):
        """
        load the subspectrum file
        """
        print "subspectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = "SubSpectrumData/" + self.filenameparser(filename)
        self.t6_filename = filename

    def t6_action(self):
        print "t6_action"
        if(not hasattr(self,'t6_filename')):
            print "please choose file"
            return
        
        peppro = PeptideProcessor()
        if(not hasattr(self,'orginalpois')):
            filename = self.t6_filename+"_IonPostion" 
            if not os.path.exists(filename):
                print "please operate Test3 Frist"
                return 
            self.orginalpois = peppro.generateIonPoitionFile(filename)
        
        print self.orginalpois
        
        subprocessor = SubSpectrumProcessor()

        subparser = SubSpectrumGenerator()
        subspects = list(subparser.generateSubSpecfile(self.t6_filename))
        
        ionpbptables = subprocessor.generateIonPepbondpoiTable(subspects,self.orginalpois,self.splitflag.get())
#        ionpbpchiValues = subprocessor.ChiSquared_TypeandPepbondPoi(subspects,self.orginalpois,self.splitflag.get())
        subprocessor.paintionpbpTable(ionpbptables)
#        subprocessor.paintChiValues(ionpbpchiValues)
#        print pepLendf     
        print self.t6_filename
        print self.splitflag.get()
    
    ################################## Test 7  TypeandType #########################
    def t7_loadFile(self):
        """
        load the subspectrum file
        """
        print "subspectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = "SubSpectrumData/" + self.filenameparser(filename)
        self.t7_filename = filename

    def t7_action(self):
        print "t7_action"
        if(not hasattr(self,'t7_filename')):
            print "please choose file"
            return
        
        ionLearner = IonGroupLearner()
        if(not hasattr(self,'orginalpois')):
            filename = self.t7_filename+"_IonPostion" 
            if not os.path.exists(filename):
                print "please operate Test3 Frist"
                return 
            self.orginalpois = ionLearner.generateIonPoitionFile(filename)

        subprocessor = SubSpectrumProcessor()
        subparser = SubSpectrumGenerator()
        subspects = list(subparser.generateSubSpecfile(self.t7_filename))
        ionchiValues = subprocessor.ChiSquared_TypeandType(subspects,self.orginalpois)
        fileName = self.t7_filename + "_typetypeChi"
        fw.writeFile_cp(fileName, ionchiValues)
#        subprocessor.paintChiValues(ionchiValues)
   
        print self.t7_filename

    ################################## Test 8  Specical subspectrum #########################
    def t8_loadFile(self):
        """
        load the spectrum file
        """
        print "spectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = self.filenameparser(filename)
        self.t8_filename = filename

    def t8_action(self):
        print "t8_action"
        if(not hasattr(self,'t8_filename')):
            print "please choose file" 
            return
        
        try:
            self.t8_binlen = float(self.t8_BlEntry.get())
            self.t8_arealen = int(self.t8_AlEntry.get())
        except:
            print "input number"
            return

        fw.writeSpecialSubSepc(self.t8_filename, self.t8_binlen, self.t8_arealen, 'intensity', self.stype.get())
   
        print self.t8_filename
        print self.stype.get()
    ################################## Test 9  Ion groups #########################
    def t9_loadSMFile(self):
        """
        load the spectrum file
        """
        print "spectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = "data/" + self.filenameparser(filename)
        orgin_file = self.filenameparser(filename).split('.')[0]
        self.t9_smfilename = filename
        self.t9_orginfilename = orgin_file

    def t9_smaction(self):
        """
          generate spectMaxInt File
        """
        print "t9_smaction"
        if(not hasattr(self,'t9_smfilename')):
            print "please choose file"
            return
        
        iglearner = IonGroupLearner()
        parser = SpectrumParser()
        specs = parser.readSpectrum(self.t9_smfilename)# orignal datas file    
        spectMaxInt = iglearner.generateMaxIntentity(specs)
        fw.writeSpectMaxInt(spectMaxInt, self.t9_orginfilename)
        
        print self.t9_smfilename
        print self.t9_orginfilename
    
    def t9_loadFile(self):
        """
        load the subspectrum file
        """
        print "subspectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = "SubSpectrumData/" + self.filenameparser(filename)
        orgin_file = self.filenameparser(filename)
        self.t9_filename = filename
        self.t9_orginfilename = orgin_file
    
    def t9_action(self):
        print "t9_action"
        if(not hasattr(self,'t9_filename')):
            print "please choose file"
            return
        
        ionLearner = IonGroupLearner()
        if(not hasattr(self,'spectMaxInt')):
            print self.t9_orginfilename.split("_")[-2]
            filename = "SubSpectrumData/"+self.t9_orginfilename.split("_")[0]+"_SpectMaxInt" 
            print filename
            if not os.path.exists(filename):
                print "please operate spectMaxInt Frist"
                return 
            self.spectMaxInt = ionLearner.generateMaxIntentityFile(filename)
        
        if(not hasattr(self,'orginalpois')):
            filename = "SubSpectrumData/"+self.t9_orginfilename.split("_")[0]+"_IonPostion" 
            if not os.path.exists(filename):
                print "please operate Test3 Frist"
                return 
            self.orginalpois = ionLearner.generateIonPoitionFile(filename)
            
        subparser = SubSpectrumGenerator()    
        
        if(self.t9_orginfilename.split("_")[-2]=='Noise'):
            subspects = subparser.generateNoiSubfile(self.t9_filename,'intensity')
        else:
            subspects = subparser.generateSubSpecfile(self.t9_filename,'intensity')
        ionLists = ionLearner.generateIonGroup_Int(subspects, self.orginalpois, self.spectMaxInt)
        file_name = "SubSpectrumData/"+self.t9_orginfilename+"_iongroup"
        fw.writeIonGroups(ionLists, file_name)
        
        print file_name
        
    ################################## Test 10 Ionclassifying#########################         
    ################################## Test 11 BYclassifying#########################    
    ################################## Test 12 Spectrum Sample#########################
    def t12_loadSMFile(self):
        """
        load the spectrum file
        """
        print "spectrumFile"
        filename = tkFileDialog.askopenfilename()
        filename = "data/" + self.filenameparser(filename)
        orgin_file = self.filenameparser(filename)
        self.t12_smfilename = filename
        self.t12_orginfilename = orgin_file
    
    def t12_action(self):
        print "t12_action"
        if(not hasattr(self,'t12_orginfilename')):
            print "please choose file"
            return
        
        try:
            self.t12_winnum = int(self.t12_WNEntry.get())
            self.t12_peaknum = int(self.t12_PNEntry.get())
        except:
            print "input number"
            return 
        
        parser = SpectrumParser()
        spectprocer = SpectrumProcessor()
        specs = parser.readSpectrum(self.t12_smfilename)   
        sampleDict = spectprocer.preprocessing(specs,self.t12_winnum, self.t12_peaknum)
        file_name = "data/"+ self.t12_orginfilename + "_sample"
        spectprocer.writetoFile_cp(sampleDict,file_name)
        print "spectrum sample finish"
    ################################## Test 13 Use Model#########################
    ################################## Test 14 Iontransforming#########################    
    ################################## Test 15 Spectrum Graph#########################    
        
        


    def filenameparser(self, filename):
        file_name = filename.split('/')[-1]
        return file_name

if __name__=='__main__':
    showP = showproject()
    tk.mainloop()