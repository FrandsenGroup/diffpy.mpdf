#!/usr/bin/env python
##############################################################################
#
# diffpy.magpdf     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2016 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Benjamin Frandsen
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################

"""Unit tests for diffpy.magpdf.
"""


# create logger instance for the tests subpackage
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
del logging


def testsuite():
    '''Build a unit tests suite for the diffpy.magpdf package.
    Return a unittest.TestSuite object.
    '''
    import unittest
    modulenames = '''
        diffpy.magpdf.tests.testbasicmpdf
        diffpy.magpdf.tests.testmpdffromcif
        diffpy.magpdf.tests.testmpdffromselfcreated
        diffpy.magpdf.tests.testformfactor
        diffpy.magpdf.tests.testgetdiffdata
    '''.split()
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    mobj = None
    for mname in modulenames:
        exec ('import %s as mobj' % mname)
        suite.addTests(loader.loadTestsFromModule(mobj))
    return suite


def test():
    '''Execute all unit tests for the diffpy.magpdf package.
    Return a unittest TestResult object.
    '''
    import unittest
    suite = testsuite()
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    return result


# End of file
