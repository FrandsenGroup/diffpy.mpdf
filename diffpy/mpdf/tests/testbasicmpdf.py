#!/usr/bin/env python

"""Unit tests basic mpdf functionalities.
"""


import unittest
import diffpy.mpdf
import numpy as np

##############################################################################
class basicmPDFtest(unittest.TestCase):
    def test(self):
        ms=diffpy.mpdf.MagStructure()
        ms.atoms=np.array([[0,0,0],[1,0,0]])
        ms.spins=np.array([[0,0,1],[0,0,-1]])
        mc=diffpy.mpdf.MPDFcalculator(magstruc=ms)
        r,fr=mc.calc()
        testval=np.round(fr[100],decimals=4)
        self.assertEqual(testval,-1.4000)

# End of class basicmPDFtest

if __name__ == '__main__':
    unittest.main()

# End of file
