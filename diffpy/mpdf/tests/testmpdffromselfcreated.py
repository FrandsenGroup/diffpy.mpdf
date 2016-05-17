#!/usr/bin/env python

"""Unit tests for creating magnetic structures.
"""


import unittest
import diffpy.mpdf
import numpy as np

##############################################################################
class mPDFfromSelftest(unittest.TestCase):
    def test(self):
        msp=diffpy.mpdf.MagSpecies(useDiffpyStruc=False)
        msp.latVecs=np.array([[4,0,0],[0,4,0],[0,0,4]])
        msp.atomBasis=np.array([[0,0,0],[0.5,0.5,0.5]])
        msp.spinBasis=np.array([[0,0,1],[0,0,-1]])
        mstr=diffpy.mpdf.MagStructure()
        mstr.loadSpecies(msp)
        mstr.makeAtoms()
        mstr.makeSpins()

        # Set up the mPDF calculator
        mc=diffpy.mpdf.MPDFcalculator(magstruc=mstr)
        r,fr=mc.calc()
        testval=np.round(fr.max(),decimals=4)
        self.assertEqual(testval,3.3996)

# End of class mPDFfromSelftest

if __name__ == '__main__':
    unittest.main()

# End of file
