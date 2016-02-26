#!/usr/bin/env python

"""Unit tests for the AtomRadiiTable class.
"""


import unittest
import sys, os
import diffpy.magpdf
import numpy as np

##############################################################################
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

class basicmPDFtest(unittest.TestCase):
    def test(self):
        path = os.path.dirname(os.path.abspath(__file__))
        r,dr=diffpy.magpdf.getDiffData([find('testdata.fgr',path)])
        testval=dr.max()
        self.assertEqual(testval,107.42)

# End of class TestAtomRadiiTable

if __name__ == '__main__':
    unittest.main()

# End of file
