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

"""class to perform mPDF calculations"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, fftconvolve
from diffpy.mpdf.magutils import calculatemPDF, calculateDr

class MPDFcalculator:
    """Create an MPDFcalculator object to help calculate mPDF functions.

    This class is loosely modelled after the PDFcalculator class in diffpy.
    At minimum, it requires a magnetic structure with atoms and spins, and
    it calculates the mPDF from that. Various other options can be specified
    for the calculated mPDF.

    Args:
        magstruc (MagStructure object): provides information about the
            magnetic structure. Must have arrays of atoms and spins.
        extendedrmin (float): extension of the r-grid on which the mPDF is
            calculated to properly account for contribution of pairs just
            before the boundary. 4 A by default.
        extendedrmax (float): extension of the r-grid on which the mPDF is
            calculated to properly account for contribution of pairs just
            outside the boundary. 4 A by default.
        qdamp (float): usual PDF qdamp parameter. Turned off if set to zero.
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
    def __init__(self, magstruc=None, extendedrmax=4.0,
                 extendedrmin=4.0, qdamp=0.0, qmin=0.0,
                 qmax=-1.0, rmin=0.0, rmax=20.0, rstep=0.01,
                 ordScale=1.0, paraScale=1.0, rmintr=-5.0,
                 rmaxtr=5.0, label=''):
        if magstruc is None:
            self.magstruc = []
        else:
            self.magstruc = magstruc
            if magstruc.rmaxAtoms < rmax:
                print('Warning: Your structure may not be big enough for your')
                print('desired calculation range.')
        self.extendedrmin = extendedrmin
        self.extendedrmax = extendedrmax
        self.qdamp = qdamp
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
            one or both of the mPDF quantities.
        """
        peakWidth = np.sqrt(self.magstruc.Uiso)
        xi = self.magstruc.corrLength
        if xi==0:
            calcIdxs = self.magstruc.calcIdxs
            rcalc, frcalc = calculatemPDF(self.magstruc.atoms, self.magstruc.spins,
                                          self.magstruc.gfactors, calcIdxs,
                                          self.rstep, self.rmin, self.rmax,
                                          peakWidth, self.qmin, self.qmax,
                                          self.qdamp, self.extendedrmin,
                                          self.extendedrmax, self.ordScale,
                                          self.magstruc.K1)
        else:
            originalSpins = 1.0*self.magstruc.spins
            for i, currentIdx in enumerate(self.magstruc.calcIdxs):
                distanceVecs = self.magstruc.atoms - self.magstruc.atoms[currentIdx]
                distances = np.apply_along_axis(np.linalg.norm, 1, distanceVecs)
                rescale = np.exp(-distances/xi)[:,np.newaxis] 
                self.magstruc.spins *= rescale 
                rcalc, frtemp = calculatemPDF(self.magstruc.atoms, self.magstruc.spins,
                                              self.magstruc.gfactors, [currentIdx],
                                              self.rstep, self.rmin, self.rmax,
                                              peakWidth, self.qmin, self.qmax,
                                              self.qdamp, self.extendedrmin,
                                              self.extendedrmax, self.ordScale,
                                              self.magstruc.K1)
                if i==0:
                    frcalc = 1.0*frtemp
                else:
                    frcalc += frtemp
                self.magstruc.spins = 1.0*originalSpins
            frcalc /= len(self.magstruc.calcIdxs)
        # create a mask to put the calculation on the desired grid
        mask = np.logical_and(rcalc > self.rmin - 0.5*self.rstep,
                              rcalc < self.rmax + 0.5*self.rstep)
        if normalized and not both:
            return rcalc[mask], frcalc[mask]
        elif not normalized and not both:
            Drcalc = calculateDr(rcalc, frcalc, self.magstruc.ffqgrid,
                                 self.magstruc.ff, self.paraScale, self.rmintr,
                                 self.rmaxtr, self.rstep, self.qmin, self.qmax,
                                 self.magstruc.K1, self.magstruc.K2)
            return rcalc[mask], Drcalc[mask]
        else:
            Drcalc = calculateDr(rcalc, frcalc, self.magstruc.ffqgrid,
                                 self.magstruc.ff, self.paraScale, self.rmintr,
                                 self.rmaxtr, self.rstep, self.qmin, self.qmax,
                                 self.magstruc.K1, self.magstruc.K2)
            return rcalc[mask], frcalc[mask], Drcalc[mask]

    def plot(self, normalized=True, both=False, scaled=True):
        """Plot the magnetic PDF.

        Args:
            normalized (boolean): indicates whether or not the normalized mPDF
                should be plotted.
            both (boolean): indicates whether or not both normalized and
                unnormalized mPDF quantities should be plotted.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'r ($\mathdefault{\AA}$)')
        ax.set_xlim(xmin=self.rmin, xmax=self.rmax)
        if normalized and not both:
            rcalc, frcalc = self.calc(normalized, both)
            ax.plot(rcalc, frcalc) 
            ax.set_ylabel(r'f ($\mathdefault{\AA^{-2}}$)')
        elif not normalized and not both:
            rcalc, Drcalc = self.calc(normalized, both)
            ax.plot(rcalc, Drcalc)
            ax.set_ylabel(r'd ($\mathdefault{\AA^{-2}}$)')
        else:
            rcalc, frcalc, Drcalc = self.calc(normalized, both)
            if scaled:
                frscl = np.max(np.abs(frcalc))
                drscl = np.max(np.abs(Drcalc[rcalc>1.5]))
                scl = frscl / drscl
            else:
                scl = 1.0
            ax.plot(rcalc, frcalc, 'b-', label='f(r)')
            ax.plot(rcalc, scl * Drcalc, 'r-', label='d(r)')
            ax.set_ylabel(r'f, d ($\mathdefault{\AA^{-2}}$)')
            plt.legend(loc='best')
        plt.show()

    def runChecks(self):
        """Run some quick checks to help with troubleshooting.
        """
        print('Running checks on MPDFcalculator...\n')

        flagCount = 0
        flag = False

        ### check if number of spins and atoms do not match
        if self.magstruc.atoms.shape[0] != self.magstruc.spins.shape[0]:
            flag = True
        if flag:
            flagCount += 1
            print('Number of atoms and spins do not match; try calling')
            print('makeAtoms() and makeSpins() again on your MagStructure.\n')
        flag = False

        ### check for nan values in spin array
        if np.any(np.isnan(self.magstruc.spins)):
            flag = True
        if flag:
            flagCount += 1
            print('Spin array contains nan values ("not a number").\n')
        flag = False

        ### check if rmax is too big for rmaxAtoms in structure
        for key in self.magstruc.species:
            if self.magstruc.species[key].rmaxAtoms < self.rmax:
                flag = True
        if flag:
            flagCount += 1
            print('Warning: the atoms in your MagStructure may not fill a')
            print('volume large enough for the desired rmax for the mPDF')
            print('calculation. Adjust rmax and/or rmaxAtoms in the')
            print('MagSpecies or MagStructure objects.\n')
        flag = False

        ### check for unphysical parameters like negative scale factors
        if np.min((self.paraScale, self.ordScale)) < 0:
            flag = True
        if flag:
            flagCount += 1
            print('Warning: you have a negative scale factor.')
            print(('Paramagnetic scale = ', self.paraScale))
            print(('Ordered scale = ', self.ordScale))
        flag = False

        if flagCount == 0:
            print('All checks passed. No obvious problems found.\n')

    def rgrid(self):
        """Return the current r grid of the mPDF calculator."""
        return np.arange(self.rmin, self.rmax+self.rstep, self.rstep)

    def copy(self):
        """Return a deep copy of the MPDFcalculator object."""
        return copy.deepcopy(self)

