#!/usr/bin/env python

"""Unit tests for the AtomRadiiTable class.
"""


import unittest
from diffpy.magpdf import jCalc

##############################################################################
def fun(x):
    return x + 1

class basicTest(unittest.TestCase):
    def test(self):
        self.assertEqual(fun(3),4)

# End of class TestAtomRadiiTable

if __name__ == '__main__':
    unittest.main()

# End of file
