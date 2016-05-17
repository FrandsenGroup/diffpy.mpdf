#!/usr/bin/env python

"""Unit tests for the spinsFromAtoms and atomsFromSpins functions.
"""


import unittest
import diffpy.mpdf
import numpy as np

##############################################################################
class spinsAtomsTest(unittest.TestCase):
    def testSpinsFromAtoms(self):
        atoms=np.array([[0,0,0],[1,0,0],[2,0,0]])
        spins=np.array([[0,0,1],[0,0,-1],[0,0,1]])
        mstr=diffpy.mpdf.MagStructure()
        mstr.atoms=atoms
        mstr.spins=spins
        testval=diffpy.mpdf.spinsFromAtoms(mstr,[1,0,0],fractional=False)[0][2]
        self.assertEqual(testval,-1)

    def testAtomsFromSpins(self):
        atoms=np.array([[0,0,0],[1,0,0],[2,0,0]])
        spins=np.array([[0,0,1],[0,0,-1],[0,0,1]])
        mstr=diffpy.mpdf.MagStructure()
        mstr.atoms=atoms
        mstr.spins=spins
        testval=diffpy.mpdf.atomsFromSpins(mstr,[0,0,-1],fractional=False)[0][0][0]
        self.assertEqual(testval,1)

# End of class spinsAtomsTest

if __name__ == '__main__':
    unittest.main()

# End of file
