#!/usr/bin/env python

"""Unit tests for various mpdf classes and functions.
"""


import unittest
import sys
import os
import diffpy.mpdf
from diffpy.Structure import loadStructure
import numpy as np

##############################################################################
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

class mPDFfromCIFtest(unittest.TestCase):
    def test(self):
        path = os.path.dirname(os.path.abspath(__file__))
        struc=loadStructure(find('MnO_cubic.cif',path))
        msp=diffpy.mpdf.MagSpecies(struc=struc)
        msp.magIdxs=[0,1,2,3]
        msp.basisvecs=np.array([[1,-1,0]])
        msp.kvecs=np.array([[0.5,0.5,0.5]])
        msp.ffparamkey='Mn2'
        mstr=diffpy.mpdf.MagStructure()
        mstr.loadSpecies(msp)
        mstr.makeAll()
        mc=diffpy.mpdf.MPDFcalculator(magstruc=mstr)
        r,fr,dr=mc.calc(both=True)
        testval=np.round(dr[100],decimals=4)
        self.assertEqual(testval,10.9411)

# End of class mPDFfromCIFtest

if __name__ == '__main__':
    unittest.main()

# End of file
