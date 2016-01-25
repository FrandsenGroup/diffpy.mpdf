#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import necessary functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize.minpack import leastsq

from diffpy.Structure import loadStructure

import sys
sys.path.append('/home/ben/mPDFmodules/mpdfcalculator')
from mcalculator import *


mnostructure=loadStructure('MnO_cubic.cif')

mnmag=magSpecies(mnostructure,'Mn')
mnmag.magIdxs=[0,1,2,3]
mnmag.kvecs=np.array([[0.5,0.5,0.5]])
mnmag.basisvecs=np.array([[1,-1,0]])

magstruc=magStructure(mnostructure)
magstruc.addSpecies(mnmag)

magstruc.makeAtoms()
magstruc.makeSpins()


