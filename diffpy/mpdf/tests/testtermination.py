#!/usr/bin/env python

"""Unit tests for the Q-space termination / window convolution applied to
the calculated mPDF (calculatemPDF with qmin, qmax or qwindow/qgrid).
"""


import unittest
import numpy as np
from diffpy.structure import Structure, Atom, Lattice
from diffpy.mpdf import MagSpecies, MagStructure, MPDFcalculator
from diffpy.mpdf.magutils import calculatemPDF

trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # numpy < 2.0

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

    def test_qmin_does_not_change_scale_unnormalized(self):
        """Same check for the unnormalized mPDF D(r), which is built from the
        terminated f(r) in calculateDr and therefore inherited the error."""
        def calcD(qmin):
            mc = MPDFcalculator(magstruc=self.mstruc, rmin=0.1, rmax=20,
                                qmin=qmin, qmax=6.0)
            r, fr, dr = mc.calc(both=True)
            return r, dr
        r, d0 = calcD(0.0)
        mask = r > 2.0
        for qmin in [0.3, 0.7, 1.0]:
            _, d = calcD(qmin)
            ratio = np.abs(d[mask]).max() / np.abs(d0[mask]).max()
            self.assertTrue(0.8 < ratio < 1.25,
                            msg='qmin=%.1f: D(r) amplitude ratio %.3f' % (qmin, ratio))
            corr = np.corrcoef(d[mask], d0[mask])[0, 1]
            self.assertGreater(corr, 0.9, msg='qmin=%.1f: D(r) corr %.3f' % (qmin, corr))

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


class terminationExactTest(unittest.TestCase):
    """Compare the terminated mPDF of an isolated spin cluster with an exact
    reference. For a finite cluster f(r) vanishes beyond a few Angstroms, so
    the effect of a finite Q-range can be computed directly: sine-transform
    f(r) to F(Q), then transform back over [qmin, qmax] only. This checks the
    odd extension of f(r) and the kernel range (previously off by 4-8%)."""

    @classmethod
    def setUpClass(cls):
        d = 3.5
        cls.xyz = np.array([[0, 0, 0], [d, 0, 0], [d / 2, d * np.sqrt(3) / 2, 0],
                            [d / 2, d * np.sqrt(3) / 6, d * np.sqrt(2.0 / 3.0)]])
        cls.sxyz = np.array([[0, 0, 1.], [0, 0, -1.], [0, 1., 0], [1., 0, 0]])
        cls.kw = dict(calcIdxs=[0, 1, 2, 3], rstep=0.01, rmin=0.0, rmax=20.0)
        cls.r, f0 = calculatemPDF(cls.xyz, cls.sxyz, qmin=-1, qmax=-1, **cls.kw)
        cls.Q = np.arange(0, 30, 0.002)
        cls.FQ = trapezoid(f0[None, :] * np.sin(np.outer(cls.Q, cls.r)), cls.r, axis=1)
        cls.qmax = 6.0

    def reference(self, qmin):
        m = (self.Q >= qmin) & (self.Q <= self.qmax)
        return (2 / np.pi) * trapezoid(self.FQ[m][None, :] * np.sin(np.outer(self.r, self.Q[m])),
                                          self.Q[m], axis=1)

    def test_qmin_qmax_matches_exact(self):
        for qmin in [0.0, 0.4, 0.7, 1.0]:
            ref = self.reference(qmin)
            r, f = calculatemPDF(self.xyz, self.sxyz, qmin=qmin, qmax=self.qmax, **self.kw)
            self.assertTrue(np.allclose(r, self.r, rtol=0, atol=1e-8))
            err = np.abs(f - ref).max() / np.abs(ref).max()
            self.assertLess(err, 0.005, msg='qmin=%.1f: rel. err %.4f' % (qmin, err))

    def test_window_matches_exact(self):
        qgrid = np.arange(0, self.qmax + 1e-9, 0.005)
        for qmin in [0.0, 0.4, 0.7, 1.0]:
            ref = self.reference(qmin)
            W = ((qgrid >= qmin) & (qgrid <= self.qmax)).astype(float)
            r, f = calculatemPDF(self.xyz, self.sxyz, qwindow=W, qgrid=qgrid, **self.kw)
            self.assertTrue(np.allclose(r, self.r, rtol=0, atol=1e-8))
            err = np.abs(f - ref).max() / np.abs(ref).max()
            self.assertLess(err, 0.005, msg='qmin=%.1f: rel. err %.4f' % (qmin, err))

# End of class terminationExactTest

if __name__ == '__main__':
    unittest.main()

# End of file
