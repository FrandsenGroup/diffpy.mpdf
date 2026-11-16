#!/usr/bin/env python

"""Unit tests for the Q-space termination / window convolution applied to
the calculated mPDF (calculatemPDF with qmin, qmax or qwindow/qgrid).
"""


import unittest
import numpy as np
from diffpy.structure import Structure, Atom, Lattice
from diffpy.mpdf import MagSpecies, MagStructure, MPDFcalculator

##############################################################################
def makeMagStructure():
    """Simple-cubic G-type antiferromagnet with one Mn2+ per cell."""
    struc = Structure([Atom('Mn', [0, 0, 0])],
                      lattice=Lattice(4, 4, 4, 90, 90, 90))
    spec = MagSpecies(struc=struc, label='Mn', strucIdxs=[0],
                      basisvecs=np.array([[0, 0, 1.]]),
                      kvecs=np.array([[0.5, 0.5, 0.5]]), S=2.5,
                      ffparamkey='Mn2', rmaxAtoms=30)
    mstruc = MagStructure(rmaxAtoms=30)
    mstruc.loadSpecies(spec)
    mstruc.makeAll()
    return mstruc


def calc(mstruc, **kwargs):
    mc = MPDFcalculator(magstruc=mstruc, rmin=0.1, rmax=20, **kwargs)
    return mc.calc()


class terminationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mstruc = makeMagStructure()
        cls.r, cls.f0 = calc(cls.mstruc, qmin=0.0, qmax=6.0)

    def test_qmin_does_not_change_scale(self):
        """A low-Q cutoff only adds weak ripples; it must not rescale the
        mPDF or flip its sign (previously off by 10-100x for qmin > 0)."""
        mask = self.r > 2.0
        for qmin in [0.3, 0.7, 1.0]:
            r, f = calc(self.mstruc, qmin=qmin, qmax=6.0)
            ratio = np.abs(f[mask]).max() / np.abs(self.f0[mask]).max()
            self.assertTrue(0.8 < ratio < 1.25,
                            msg='qmin=%.1f: amplitude ratio %.3f' % (qmin, ratio))
            corr = np.corrcoef(f[mask], self.f0[mask])[0, 1]
            self.assertGreater(corr, 0.9, msg='qmin=%.1f: corr %.3f' % (qmin, corr))

    def test_window_matches_qmin_qmax(self):
        """A box window on [qmin, qmax] must reproduce the qmin/qmax path."""
        q = np.arange(0, 6.0 + 1e-9, 0.005)
        mask = (self.r > 2.0) & (self.r < 18.0)
        for qmin in [0.0, 0.7]:
            _, fq = calc(self.mstruc, qmin=qmin, qmax=6.0)
            W = ((q >= qmin) & (q <= 6.0)).astype(float)
            _, fw = calc(self.mstruc, qwindow=W, qgrid=q)
            err = np.abs(fw[mask] - fq[mask]).max() / np.abs(fq[mask]).max()
            self.assertLess(err, 0.05, msg='qmin=%.1f: rel. diff %.3f' % (qmin, err))

    def test_large_qmax_recovers_untruncated(self):
        """With qmin = 0 and a large qmax, termination effects vanish."""
        _, fu = calc(self.mstruc, qmin=-1, qmax=-1)
        _, ft = calc(self.mstruc, qmin=0.0, qmax=40.0)
        mask = (self.r > 2.0) & (self.r < 18.0)
        err = np.abs(ft[mask] - fu[mask]).max() / np.abs(fu[mask]).max()
        self.assertLess(err, 0.05, msg='rel. diff %.3f' % err)

# End of class terminationTest

if __name__ == '__main__':
    unittest.main()

# End of file
