#!/usr/bin/env python
##############################################################################
#
# diffpy.mpdf       by Billinge Group
#                     Simon J. L. Billinge sb2896@columbia.edu
#                     (c) 2016 trustees of Columbia University in the City of
#                           New York.
#                      All rights reserved
#
# File coded by:    Benjamin Frandsen
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""functions and classes to perform mPDF calculations"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

def jCalc(q, params=[0.2394, 26.038, 0.4727, 12.1375, 0.3065, 3.0939, -0.01906],
          j2=False):
    """Calculate the magnetic form factor j0.

    This method for calculating the magnetic form factor is based on the
    approximate empirical forms based on the tables by Brown, consisting of
    the sum of 3 Gaussians and a constant.

    Args:
        q (numpy array): 1-d grid of momentum transfer on which the form
            factor is to be computed
        params (python list): provides the 7 numerical coefficients. The
            default is an average form factor of 3d j0 approximations.
        j2 (boolean): if True, calculate the j2 approximation for orbital
            angular momentum contributions

    Returns:
        numpy array with same shape as q giving the magnetic form factor j0 or j2.
    """
    [A, a, B, b, C, c, D] = params
    if j2:
        return (A*np.exp(-a*(q/4/np.pi)**2)+B*np.exp(-b*(q/4/np.pi)**2)+C*np.exp(-c*(q/4/np.pi)**2)+D)*(q/4.0/np.pi)**2
    else:
        return A*np.exp(-a*(q/4/np.pi)**2)+B*np.exp(-b*(q/4/np.pi)**2)+C*np.exp(-c*(q/4/np.pi)**2)+D

def cv(x1, y1, x2, y2):
    """Perform the convolution of two functions and give the correct output.


    Args:
        x1 (numpy array): independent variable of first function; must be in
            ascending order
        y1 (numpy array): dependent variable of first function
        x2 (numpy array): independent variable of second function; must have
            same grid spacing as x1
        y2 (numpy array): dependent variable of second function

    Returns:
        xcv (numpy array): independent variable of convoluted function, has
            dimension len(x1) + len(x2) - 1

        ycv (numpy array): convolution of y1 and y2, same shape as xcv

    """
    dx = x1[1]-x1[0]
    ycv = dx*np.convolve(y1, y2, 'full')
    xcv = np.linspace(x1[0]+x2[0], x1[-1]+x2[-1], len(ycv))
    return xcv, ycv

def cosTransform(q, fq, rmin=0.0, rmax=50.0, rstep=0.1): # does not require even q-grid
    """Compute the cosine Fourier transform of a function.

    This method uses direct integration rather than an FFT and doesn't require
    an even grid. The grid for the Fourier transform is even and specifiable.

    Args:
        q (numpy array): independent variable for function to be transformed
        fq (numpy array): dependent variable for function to be transformed
        rmin (float, default = 0.0): min value of conjugate independent variable
            grid
        rmax (float, default = 50.0): maximum value of conjugate independent
            variable grid
        rstep (float, default = 0.1): grid spacing for conjugate independent
            variable

    Returns:
        r (numpy array): independent variable grid for transformed quantity

        fr (numpy array): cosine Fourier transform of fq
    """
    lostep = int(np.ceil((rmin - 1e-8) / rstep))
    histep = int(np.floor((rmax + 1e-8) / rstep)) + 1
    r = np.arange(lostep, histep)*rstep
    qrmat = np.outer(r, q)
    integrand = fq*np.cos(qrmat)
    fr = np.sqrt(2.0/np.pi)*np.trapz(integrand, q)
    return r, fr


def getDiffData(fileNames=[], fmt='pdfgui', writedata=False):
    """Extract the fit residual from a structural PDF fit.

    Args:
        fileNames (python list): list of paths to the files containing the
            fit information (e.g. calculated and experimental PDF, as in the
            .fgr files from PDFgui exported fits)
        fmt (string): string identifying the format of the file(s). Options
            are currently just 'pdfgui'.
        writedata (boolean): whether or not the output should be saved to file

    Returns:
        r (numpy array): same r-grid as contained in the fit file

        diff (numpy array): the structural PDF fit residual (i.e. the mPDF)
    """
    for name in fileNames:
        if fmt == 'pdfgui':
            allcols = np.loadtxt(name, unpack=True, comments='#', skiprows=14)
            r, diff = allcols[0], allcols[4]
            if writedata:
                np.savetxt(name[:-4]+'.diff', np.transpose((r, diff)))
            else:
                return r, diff
        else:
            print 'This format is not currently supported.'

def calculatemPDF(xyz, sxyz, gfactors=np.array([2.0]), calcList=np.array([0]),
                  rstep=0.01, rmin=0.0, rmax=20.0, psigma=0.1, qmin=0,
                  qmax=-1, dampRate=0.0, dampPower=2.0, maxextension=10.0,
                  orderedScale=1.0/np.sqrt(2*np.pi)):
    """Calculate the normalized mPDF.

    At minimum, this module requires input lists of atomic positions and spins.

    Args:
        xyz (numpy array): list of atomic coordinates of all the magnetic
            atoms in the structure.
        sxyz (numpy array): triplets giving the spin vectors of all the
            atoms, in the same order as the xyz array provided as input.
        gfactors (numpy array): Lande g-factors of spins in same order as
            spin array.
        calcList (python list): list giving the indices of the atoms array
            specifying the atoms to be used as the origin when calculating
            the mPDF.
        rstep (float): step size for r-grid of calculated mPDF.
        rmin (float): minimum value of r for which mPDF should be calculated.
        rmax (float): maximum value of r for which mPDF should be calculated.
        psigma(float): std deviation (in Angstroms) of Gaussian peak
            to be convoluted with the calculated mPDF to simulate thermal
            motion.
        qmin (float): minimum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.
        qmax (float): maximum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.
        dampRate (float): generalized ("stretched") exponential damping rate
                of the mPDF.
        dampPower (float): power of the generalized exponential function.
        maxextension (float): extension of the r-grid on which the mPDF is
            calculated to properly account for contribution of pairs just
            outside the boundary.
        ordScale (float): overall scale factor for the mPDF function f(r).

    Returns: numpy arrays for r and the mPDF fr.
        """
    # check if g-factors have been provided
    if sxyz.shape[0] != gfactors.shape[0]:
        gfactors = 2.0*np.ones(sxyz.shape[0])

    # calculate s1, s2
    r = np.arange(rmin, rmax+maxextension+rstep, rstep)
    rbin = np.concatenate([r-rstep/2, [r[-1]+rstep/2]])

    s1 = np.zeros(len(r))
    s2 = np.zeros(len(r))

    for uu in calcList:
        ri = xyz[uu]
        rj = xyz
        si = sxyz[uu]
        sj = sxyz
        gi = gfactors[uu]
        gj = gfactors

        dxyz = rj-ri
        d2xyz = np.sum((dxyz)**2, axis=1).reshape(dxyz.shape[0], 1)
        d1xyz = np.sqrt(d2xyz)
        d1xyz[uu] = 1e-6 ### avoid divide by zero problem
        d1xyzr = d1xyz.ravel()

        xh = dxyz / d1xyz
        yh = si - xh * np.sum(si*xh, axis=1).reshape(dxyz.shape[0], 1)
        yh_dis = np.sum(yh**2, axis=1)
        yh_ind = np.nonzero(np.abs(yh_dis) < 1e-10)
        yh[yh_ind] = [0, 0, 0]
        yh_dis[yh_ind] = 1e-6 ### avoid divide by zero problem

        aij = np.sum(si * yh, axis=1) * np.sum(sj * yh, axis=1) / yh_dis
        aij[uu] = 0
        bij = 2 * np.sum(si * xh, axis=1) * np.sum(sj * xh, axis=1) - aij

        w2 = bij / d1xyzr**3

        d1xyzr[uu] = 0.0
        s1 += np.histogram(d1xyzr, bins=rbin, weights=gi*gj*aij)[0]
        s2 += np.histogram(d1xyzr, bins=rbin, weights=gi*gj*w2)[0]

    # apply Gaussian shape function
    if psigma != None:
        x = np.arange(-3, 3, rstep)
        y = np.exp(-x**2 / psigma**2 / 2) * (1 / np.sqrt(2*np.pi) / psigma)

        s1[0] = 0
        s1 = fftconvolve(s1, y)
        s1 = s1[len(x)/2: -len(x)/2+1]

        s2 = fftconvolve(s2, y) * rstep
        s2 = s2[len(x)/2: -len(x)/2+1]

    ss2 = np.cumsum(s2)

    if rmin == 0:
        r[0] = 1e-4*rstep # avoid infinities at r = 0
    fr = s1 / r + r * (ss2[-1] - ss2)
    r[0] = rmin
    fr /= len(calcList)*np.mean(gfactors)**2

    fr *= orderedScale*np.exp(-1.0*(dampRate*r/dampPower)**dampPower)
    # Do the convolution with the termination function if qmin/qmax have been given
    if qmin >= 0 and qmax > qmin:
        rth = np.arange(0.0, rmax+maxextension+rstep, rstep)
        rth[0] = 1e-4*rstep # avoid infinities at r = 0
        th = (np.sin(qmax*rth)-np.sin(qmin*rth))/np.pi/rth
        rth[0] = 0.0
        rcv, frcv = cv(r, fr, rth, th)
    else:
        rcv, frcv = r, fr

    return rcv[np.logical_and(rcv >= r[0]-0.5*rstep, rcv <= rmax+0.5*rstep)], \
           frcv[np.logical_and(rcv >= r[0]-0.5*rstep, rcv <= rmax+0.5*rstep)]


def calculateDr(r, fr, q, ff, paraScale=1.0, rmintr=-5.0, rmaxtr=5.0,
                drtr=0.01, qmin=0, qmax=-1):
    """Calculate the unnormalized mPDF quantity D(r).

    This module requires a normalized mPDF as an input, as well as a magnetic
    form factor and associated q grid.

    Args:
        r (numpy array): r grid for the properly normalized mPDF.
        fr (numpy array): the properly normalized mPDF.
        q (numpy array): grid of momentum transfer values used for calculating
            the magnetic form factor.
        ff (numpy array): magnetic form factor. Same shape as ffqgrid.
        paraScale (float): scale factor for the paramagnetic part of the
            unnormalized mPDF function D(r).
        rmintr (float): minimum value of r for the Fourier transform of the
            magnetic form factor required for unnormalized mPDF.
        rmaxtr (float): maximum value of r for the Fourier transform of the
            magnetic form factor required for unnormalized mPDF.
        drtr (float): step size for r-grid used for calculating Fourier
            transform of magnetic form mactor.
        qmin (float): minimum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.
        qmax (float): maximum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.

    Returns: numpy array for the unnormalized mPDF Dr.
    """

    rsr, sr = cosTransform(q, ff, rmintr, rmaxtr, drtr)
    sr = np.sqrt(np.pi/2.0)*sr
    rSr, Sr = cv(rsr, sr, rsr, sr)
    para = -1.0*np.sqrt(2.0*np.pi)*np.gradient(Sr, rSr[1]-rSr[0]) ### paramagnetic term in d(r)
    rDr, Dr = cv(r, fr, rSr, Sr)
    if qmin >= 0 and qmax > qmin:
        rstep = r[1]-r[0]
        rth = np.arange(0.0, r.max()+rstep, rstep)
        rth[0] = 1e-4*rstep # avoid infinities at r = 0
        th = (np.sin(qmax*rth)-np.sin(qmin*rth))/np.pi/rth
        rth[0] = 0.0
        rpara, para = cv(rSr, para, rth, th)
    else:
        rpara, para = rSr, para
    # make sure para and Dr match up in shape
    dr = r[1]-r[0]
    goodslice = np.logical_and(rDr >= r.min() - 0.5*dr, rDr <= r.max()+0.5*dr)
    Dr = Dr[goodslice]
    para = para[rpara >= r.min()]
    if para.shape[0] < Dr.shape[0]:
        para = np.concatenate((para, np.zeros(Dr.shape[0]-para.shape[0])))
    else:
        para = para[:Dr.shape[0]]
    Dr += paraScale*para
    return Dr

class MPDFcalculator:
    """Create an MPDFcalculator object to help calculate mPDF functions.

    This class is loosely modelled after the PDFcalculator class in diffpy.
    At minimum, it requires a magnetic structure with atoms and spins, and
    it calculates the mPDF from that. Various other options can be specified
    for the calculated mPDF.

    Args:
        magstruc (MagStructure object): provides information about the
            magnetic structure. Must have arrays of atoms and spins.
        calcList (python list): list giving the indices of the atoms array
            specifying the atoms to be used as the origin when calculating
            the mPDF.
        maxextension (float): extension of the r-grid on which the mPDF is
            calculated to properly account for contribution of pairs just
            outside the boundary.
        gaussPeakWidth(float): std deviation (in Angstroms) of Gaussian peak
            to be convoluted with the calculated mPDF to simulate thermal
            motion.
        dampRate (float): generalized ("stretched") exponential damping rate
                of the mPDF.
        dampPower (float): power of the generalized exponential function.
        qmin (float): minimum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.
        qmax (float): maximum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.
        rmin (float): minimum value of r for which mPDF should be calculated.
        rmax (float): maximum value of r for which mPDF should be calculated.
        rstep (float): step size for r-grid of calculated mPDF.
        ordScale (float): overall scale factor for the mPDF function f(r).
        paraScale (float): scale factor for the paramagnetic part of the
            unnormalized mPDF function D(r).
        rmintr (float): minimum value of r for the Fourier transform of the
            magnetic form factor required for unnormalized mPDF.
        rmaxtr (float): maximum value of r for the Fourier transform of the
            magnetic form factor required for unnormalized mPDF.
        drtr (float): step size for r-grid used for calculating Fourier
            transform of magnetic form mactor.
        label (string): Optional descriptive string for the MPDFcalculator.
        """
    def __init__(self, magstruc=None, calcList=[0], maxextension=10.0,
                 gaussPeakWidth=0.1, dampRate=0.0, dampPower=2.0, qmin=-1.0,
                 qmax=-1.0, rmin=0.0, rmax=20.0, rstep=0.01,
                 ordScale=1.0/np.sqrt(2*np.pi), paraScale=1.0, rmintr=-5.0,
                 rmaxtr=5.0, label=''):
        if magstruc is None:
            self.magstruc = []
        else:
            self.magstruc = magstruc
            if magstruc.rmaxAtoms < rmax:
                print 'Warning: Your structure may not be big enough for your'
                print 'desired calculation range.'
        self.calcList = calcList
        self.maxextension = maxextension
        self.gaussPeakWidth = gaussPeakWidth
        self.dampRate = dampRate
        self.dampPower = dampPower
        self.qmin = qmin
        self.qmax = qmax
        self.rmin = rmin
        self.rmax = rmax
        self.rstep = rstep
        self.ordScale = ordScale
        self.paraScale = paraScale
        self.rmintr = rmintr
        self.rmaxtr = rmaxtr
        self.label = label

    def __repr__(self):
        if self.label == '':
            return 'MPDFcalculator() object'
        else:
            return self.label+': MPDFcalculator() object'

    def calc(self, normalized=True, both=False):
        """Calculate the magnetic PDF.

        Args:
            normalized (boolean): indicates whether or not the normalized mPDF
                should be returned.
            both (boolean): indicates whether or not both normalized and
                unnormalized mPDF quantities should be returned.

        Returns: numpy array giving the r grid of the calculation, as well as
            one or both the mPDF quantities.
        """
        if normalized and not both:
            rcalc, frcalc = calculatemPDF(self.magstruc.atoms, self.magstruc.spins,
                                          self.magstruc.gfactors, self.calcList,
                                          self.rstep, self.rmin, self.rmax,
                                          self.gaussPeakWidth, self.qmin, self.qmax,
                                          self.dampRate, self.dampPower,
                                          self.maxextension, self.ordScale)
            return rcalc, frcalc
        elif not normalized and not both:
            rcalc, frcalc = calculatemPDF(self.magstruc.atoms, self.magstruc.spins,
                                          self.magstruc.gfactors, self.calcList,
                                          self.rstep, self.rmin, self.rmax+self.maxextension,
                                          self.gaussPeakWidth, self.qmin, self.qmax,
                                          self.dampRate, self.dampPower,
                                          self.maxextension, self.ordScale)
            Drcalc = calculateDr(rcalc, frcalc, self.magstruc.ffqgrid,
                                 self.magstruc.ff, self.paraScale, self.rmintr,
                                 self.rmaxtr, self.rstep, self.qmin, self.qmax)
            mask = (rcalc <= self.rmax + 0.5*self.rstep)
            return rcalc[mask], Drcalc[mask]
        else:
            rcalc, frcalc = calculatemPDF(self.magstruc.atoms, self.magstruc.spins,
                                          self.magstruc.gfactors, self.calcList,
                                          self.rstep, self.rmin, self.rmax+self.maxextension,
                                          self.gaussPeakWidth, self.qmin, self.qmax,
                                          self.dampRate, self.dampPower,
                                          self.maxextension, self.ordScale)
            Drcalc = calculateDr(rcalc, frcalc, self.magstruc.ffqgrid,
                                 self.magstruc.ff, self.paraScale, self.rmintr,
                                 self.rmaxtr, self.rstep, self.qmin, self.qmax)
            mask = (rcalc <= self.rmax + 0.5*self.rstep)
            return rcalc[mask], frcalc[mask], Drcalc[mask]

    def plot(self, normalized=True, both=False):
        """Plot the magnetic PDF.

        Args:
            normalized (boolean): indicates whether or not the normalized mPDF
                should be plotted.
            both (boolean): indicates whether or not both normalized and
                unnormalized mPDF quantities should be plotted.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'r ($\AA$)')
        ax.set_xlim(xmin=self.rmin, xmax=self.rmax)
        if normalized and not both:
            rcalc, frcalc = calculatemPDF(self.magstruc.atoms, self.magstruc.spins,
                                          self.magstruc.gfactors, self.calcList,
                                          self.rstep, self.rmin, self.rmax,
                                          self.gaussPeakWidth, self.qmin, self.qmax,
                                          self.dampRate, self.dampPower,
                                          self.maxextension, self.ordScale)
            ax.plot(rcalc, frcalc) 
            ax.set_ylabel(r'f ($\AA^{-2}$)')
        elif not normalized and not both:
            rcalc, frcalc = calculatemPDF(self.magstruc.atoms, self.magstruc.spins,
                                          self.magstruc.gfactors, self.calcList,
                                          self.rstep, self.rmin, self.rmax+self.maxextension,
                                          self.gaussPeakWidth, self.qmin, self.qmax,
                                          self.dampRate, self.dampPower,
                                          self.maxextension, self.ordScale)
            Drcalc = calculateDr(rcalc, frcalc, self.magstruc.ffqgrid,
                                 self.magstruc.ff, self.paraScale, self.rmintr,
                                 self.rmaxtr, self.rstep, self.qmin, self.qmax)
            mask = (rcalc <= self.rmax + 0.5*self.rstep)
            ax.plot(rcalc[mask], Drcalc[mask])
            ax.set_ylabel(r'd ($\AA^{-2}$)')
        else:
            rcalc, frcalc = calculatemPDF(self.magstruc.atoms, self.magstruc.spins,
                                          self.magstruc.gfactors, self.calcList,
                                          self.rstep, self.rmin, self.rmax+self.maxextension,
                                          self.gaussPeakWidth, self.qmin, self.qmax,
                                          self.dampRate, self.dampPower,
                                          self.maxextension, self.ordScale)
            Drcalc = calculateDr(rcalc, frcalc, self.magstruc.ffqgrid,
                                 self.magstruc.ff, self.paraScale, self.rmintr,
                                 self.rmaxtr, self.rstep, self.qmin, self.qmax)
            mask = (rcalc <= self.rmax + 0.5*self.rstep)
            ax.plot(rcalc[mask], frcalc[mask]*np.sqrt(2*np.pi), 'b-', label='f(r)') # extra factor makes it easier to compare
            ax.plot(rcalc[mask], Drcalc[mask], 'r-', label='d(r)')
            ax.set_ylabel(r'f, d ($\AA^{-2}$)')
            plt.legend(loc='best')
        plt.show()

    def runChecks(self):
        """Run some quick checks to help with troubleshooting.
        """
        print 'Running checks on MPDFcalculator...\n'

        flagCount = 0
        flag = False

        ### check if number of spins and atoms do not match
        if self.magstruc.atoms.shape[0] != self.magstruc.spins.shape[0]:
            flag = True
        if flag:
            flagCount += 1
            print 'Number of atoms and spins do not match; try calling'
            print 'makeAtoms() and makeSpins() again on your MagStructure.\n'
        flag = False

        ### check for nan values in spin array
        if np.any(np.isnan(self.magstruc.spins)):
            flag = True
        if flag:
            flagCount += 1
            print 'Spin array contains nan values ("not a number").\n'
        flag = False

        ### check if rmax is too big for rmaxAtoms in structure
        for key in self.magstruc.species:
            if self.magstruc.species[key].rmaxAtoms < self.rmax:
                flag = True
        if flag:
            flagCount += 1
            print 'Warning: the atoms in your MagStructure may not fill a'
            print 'volume large enough for the desired rmax for the mPDF'
            print 'calculation. Adjust rmax and/or rmaxAtoms in the'
            print 'MagSpecies or MagStructure objects.\n'
        flag = False

        ### check if calcList may not be representative of all MagSpecies.
        if len(self.calcList) < len(self.magstruc.species):
            flag = True
        if flag:
            flagCount += 1
            print 'Warning: your calcList may not be representative of all'
            print 'the magnetic species. calcList should have the index of'
            print 'at least one spin from each species. Use'
            print 'magStruc.getSpeciesIdxs() to see starting indices for'
            print 'each species.\n'
        flag = False

        ### check if calcList has indices that exceed the spin array
        if (np.array(self.calcList).max()+1) > self.magstruc.atoms.shape[0]:
            flag = True
        if flag:
            flagCount += 1
            print 'calcList contains indices that are too large for the'
            print 'arrays of atoms and spins contained in the MagStructure.'
        flag = False

        ### check for unphysical parameters like negative scale factors
        if np.min((self.paraScale, self.ordScale)) < 0:
            flag = True
        if flag:
            flagCount += 1
            print 'Warning: you have a negative scale factor.'
            print 'Paramagnetic scale = ', self.paraScale
            print 'Ordered scale = ', self.ordScale
        flag = False

        if flagCount == 0:
            print 'All checks passed. No obvious problems found.\n'

    def rgrid(self):
        """Return the current r grid of the mPDF calculator."""
        return np.arange(self.rmin, self.rmax+self.rstep, self.rstep)

    def copy(self):
        """Return a deep copy of the MPDFcalculator object."""
        return copy.deepcopy(self)

