#!/usr/bin/env python

"""Unit tests for magnetic form factor table.
"""


import unittest
import diffpy.mpdf
import numpy as np

##############################################################################
class formFactorTest(unittest.TestCase):
    def test(self):
        q=np.arange(0,10,0.01)
        fq=diffpy.mpdf.jCalc(q,diffpy.mpdf.getFFparams('Mn2'))
        testval=np.round(fq[100],decimals=4)
        self.assertEqual(testval,0.9323)

# End of class formFactorTest

if __name__ == '__main__':
    unittest.main()

# End of file
