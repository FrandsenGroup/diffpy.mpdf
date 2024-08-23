#!/usr/bin/env python
##############################################################################
#
# diffpy.mpdf         by Frandsen Group
#                     Benjamin A. Frandsen benfrandsen@byu.edu
#                     (c) 2022 Benjamin Allen Frandsen
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
from diffpy.mpdf.magutils import calculatemPDF, calculateDr, estimate_effective_xi

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
        qwindow (numpy array): Q-space window function applied to the data
            prior to Fourier transformation. Not used by default.
        qgrid (numpy array): Q-space grid on which the window function is
            defined.
        """
    def __init__(self, magstruc=None, extendedrmax=4.0,
                 extendedrmin=4.0, qdamp=0.0, qmin=0.0,
                 qmax=-1.0, rmin=0.0, rmax=20.0, rstep=0.01,
                 ordScale=1.0, paraScale=1.0, rmintr=-5.0,
                 rmaxtr=5.0, label='', qwindow=None, qgrid=None):
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
        if qwindow is None:
            self.qwindow = np.array([0])
        else:
            self.qwindow = qwindow
        if qgrid is None:
            self.qgrid = np.array([0])
        else:
            self.qgrid = qgrid

    def __repr__(self):
        if self.label == '':
            return 'MPDFcalculator() object'
        else:
            return self.label+': MPDFcalculator() object'

    def calc(self, normalized=True, both=False, correlationMethod='simple',
             linearTermMethod='exact'):
        """Calculate the magnetic PDF.

        Args:
            normalized (boolean): indicates whether or not the normalized mPDF
                should be returned.
            both (boolean): indicates whether or not both normalized and
                unnormalized mPDF quantities should be returned.
            correlationMethod (string): determines how the calculation should
                be done if the correlation length is finite. Options are:
                'simple'; exponential envelope is applied to the mPDF
                'full'; actual spin magnitudes are adjusted according to the
                        correlation length; more accurate (especially for very
                        short correlation lengths) but slower (especially if
                        rmax is beyond ~30 A)
                'auto'; simple method is chosen if xi <= 5 A, full otherwise
                Note that any other option will be converted to 'simple'
            linearTermMethod (string): determines how the calculation will
                handle the linear term present for magnetic structures with a net
                magnetization. Options are:
                'exact'; slope will be calculated from the values of
                    MagStructure.rho0 and MagStructure.netMag, damping set by
                    MagStructure.corrLength.
                'autoslope': slope will be determined by least-squares
                    minimization of the calculated mPDF, thereby ensuring that
                    the mPDF oscillates around zero, as it is supposed to.
                    Damping set by MagStructure.corrLength.
                'fullauto': slope and damping set by least-squares minimization.
                    This should only be used in the case of an anisotropic
                    correlation length.
                Note that any other option will be converted to 'exact'.
        Returns: numpy array giving the r grid of the calculation, as well as
            one or both of the mPDF quantities.
        """
        peakWidth = np.sqrt(self.magstruc.Uiso)
        if correlationMethod not in ['simple', 'full', 'auto']:
            correlationMethod = 'simple'  # convert non-standard inputs to simple
        if linearTermMethod not in ['exact', 'autoslope', 'fullauto']:
            linearTermMethod = 'exact'  # convert non-standard inputs to simple
        dampingMat = self.magstruc.dampingMat
        xi = self.magstruc.corrLength
        if correlationMethod == 'auto':  # convert to full or simple
            if xi <= 5.0:
                correlationMethod = 'full'
            else:
                correlationMethod = 'simple'

        if type(dampingMat) == np.ndarray and correlationMethod != 'full':
            print("Warning: correlationMethod should be set to 'full' when using")
            print("the damping matrix instead of a scalar correlation length.")
        if type(dampingMat) != np.ndarray and linearTermMethod == 'fullauto':
            print("Warning: 'fullauto' should only be used with an anisotropic")
            print("correlation length as encoded in the damping matrix.")
        if (xi==0) and (type(dampingMat) != np.ndarray):
            calcIdxs = self.magstruc.calcIdxs
            rcalc, frcalc = calculatemPDF(self.magstruc.atoms, self.magstruc.spins,
                                          self.magstruc.gfactors, calcIdxs,
                                          self.rstep, self.rmin, self.rmax,
                                          peakWidth, self.qmin, self.qmax,
                                          self.qdamp, self.extendedrmin,
                                          self.extendedrmax, self.ordScale,
                                          self.magstruc.K1, self.magstruc.rho0,
                                          self.magstruc.netMag, xi,
                                          linearTermMethod, False, self.qwindow,
                                          self.qgrid)
        elif correlationMethod == 'full':  # change magnitudes of the spins
            originalSpins = 1.0*self.magstruc.spins
            for i, currentIdx in enumerate(self.magstruc.calcIdxs):
                self.magstruc.spins = self.magstruc.generateScaledSpins(currentIdx)
                rcalc, frtemp = calculatemPDF(self.magstruc.atoms, self.magstruc.spins,
                                              self.magstruc.gfactors, [currentIdx],
                                              self.rstep, self.rmin, self.rmax,
                                              peakWidth, self.qmin, self.qmax,
                                              self.qdamp, self.extendedrmin,
                                              self.extendedrmax, self.ordScale,
                                              self.magstruc.K1, self.magstruc.rho0,
                                              self.magstruc.netMag, xi,
                                              linearTermMethod, False, self.qwindow,
                                              self.qgrid)
                if i==0:
                    frcalc = 1.0*frtemp
                else:
                    frcalc += frtemp
                self.magstruc.spins = 1.0*originalSpins
            frcalc /= len(self.magstruc.calcIdxs)
        else:  # simple method: apply exponential envelope
            calcIdxs = self.magstruc.calcIdxs
            rcalc, frcalc = calculatemPDF(self.magstruc.atoms, self.magstruc.spins,
                                          self.magstruc.gfactors, calcIdxs,
                                          self.rstep, self.rmin, self.rmax,
                                          peakWidth, self.qmin, self.qmax,
                                          self.qdamp, self.extendedrmin,
                                          self.extendedrmax, self.ordScale,
                                          self.magstruc.K1, self.magstruc.rho0,
                                          self.magstruc.netMag, xi,
                                          linearTermMethod, True, self.qwindow,
                                          self.qgrid)
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

    def plot(self, normalized=True, both=False, scaled=True,
             correlationMethod='simple', linearTermMethod='exact'):
        """Plot the magnetic PDF.

        Args:
            normalized (boolean): indicates whether or not the normalized mPDF
                should be plotted.
            both (boolean): indicates whether or not both normalized and
                unnormalized mPDF quantities should be plotted.
            correlationMethod (string): determines how the calculation should
                be done if the correlation length is finite. Options are:
                'simple'; exponential envelope is applied to the mPDF
                'full'; actual spin magnitudes are adjusted according to the
                        correlation length; more accurate (especially for very
                        short correlation lengths) but slower (especially if
                        rmax is beyond ~30 A)
                'auto'; simple method is chosen if xi <= 5 A, full otherwise
                Note that any other option will be converted to 'simple'
            linearTermMethod (string): determines how the calculation will
                handle the linear term present for magnetic structures with a net
                magnetization. Options are:
                'exact'; slope will be calculated from the values of
                    MagStructure.rho0 and MagStructure.netMag, damping set by
                    MagStructure.corrLength.
                'autoslope': slope will be determined by least-squares
                    minimization of the calculated mPDF, thereby ensuring that
                    the mPDF oscillates around zero, as it is supposed to.
                    Damping set by MagStructure.corrLength.
                'fullauto': slope and damping set by least-squares minimization.
                    This should only be used in the case of an anisotropic
                    correlation length.
                Note that any other option will be converted to 'exact'.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'r ($\mathdefault{\AA}$)')
        ax.set_xlim(xmin=self.rmin, xmax=self.rmax)
        if normalized and not both:
            rcalc, frcalc = self.calc(normalized, both, correlationMethod, linearTermMethod)
            ax.plot(rcalc, frcalc) 
            ax.set_ylabel(r'f ($\mathdefault{\AA^{-2}}$)')
        elif not normalized and not both:
            rcalc, Drcalc = self.calc(normalized, both, correlationMethod, linearTermMethod)
            ax.plot(rcalc, Drcalc)
            ax.set_ylabel(r'd ($\mathdefault{\AA^{-2}}$)')
        else:
            rcalc, frcalc, Drcalc = self.calc(normalized, both, correlationMethod, linearTermMethod)
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
        r = np.arange(0,self.rmax+10,self.rstep)        
        mask = np.logical_and(r > self.rmin - 0.5*self.rstep,
                              r < self.rmax + 0.5*self.rstep)
        return r[mask]

    def copy(self):
        """Return a deep copy of the MPDFcalculator object."""
        return copy.deepcopy(self)

