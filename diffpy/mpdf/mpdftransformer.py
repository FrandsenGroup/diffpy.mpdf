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


"""class to transform magnetic scattering data into mPDF data."""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval
from diffpy.mpdf.magutils import sinTransformDirectIntegration

def resid1(p, ff, iq, diq):
    """Scale the magnetic form factor to the data."""
    return (iq - (p[0] * ff)**2)/diq

def bkgfunc(p, q):
    """Generate polynomial to fit to F(Q)."""
    return q * polyval(q, p)

def resid2(p, q, fqm, dfqm):
    """Fit polynomial to F(Q)."""
    return (fqm - bkgfunc(p, q))/dfqm

def window_FD(q, qmaxwindow, width):
    """
    Window function based on the Fermi-Dirac function.
    Goes to zero at qmaxwindow.
    """
    window = 0.0 * q
    mask = (q <= qmaxwindow)
    window[mask] = 2.0/(np.exp((q[mask]-qmaxwindow)/width) + 1) - 1
    return window

def window_Lorch(q, qmaxwindow):
    """
    Lorch window function. Goes to zero at qmaxwindow.
    """
    window = 0.0 * q
    mask = (q <= qmaxwindow)
    window[mask] = (qmaxwindow/np.pi/q[mask]) * \
                   np.sin(np.pi*q[mask]/qmaxwindow)
    return window

def getmPDF_unnormalized(q, iq, qmin, qmax, rmin=0.05, rmax=100, rstep=0.01):
    """Sine Fourier transform to generate the unnormalized mPDF.

    This function is called by an instance of mpdftransformer to generate
    the unnormalized mPDF.

    Args:
        q (np array): Momentum transfer for which scattered intensities
            have been measured. Need not be uniform.
        iq (np array): Scattered intensity.
        qmin (float): Minimum q value included in the Fourier transform.
            In inverse Angstroms.
        qmax (float): Maximum q value included in the Fourier transform.
            In inverse Angstroms.
        rmin (float): Minimum r value for the mPDF data. In Angstroms.
        rmax (float): Maximum r value for the mPDF data. In Angstroms.
        rstep (float): Step size for the r-grid for the mPDF data. In
            Angstroms.

    Returns:
        r (np array): r grid of the unnormalized mPDF.
        dmag (np array): Unnormalized mPDF.
    """
    mask = np.logical_and(q>=qmin, q<=qmax)
    q = q[mask]
    iq = iq[mask]
    r, dmag = sinTransformDirectIntegration(q, q*iq, rmin=rmin, rmax=rmax,
                                            rstep=rstep)
    return r, dmag

def getmPDF_normalized(q, iq, ff, qmin, qmax, qmaxinst, rpoly,
                  diq=None, qstart=3.0, rmin=0.05, rmax=100, rstep=0.01,
                  window='None', qmaxwindow=0, windowedgewidth=0.1):
    """PDFgetX3-style processing to produce the normalized mPDF.
    
    This function is called by an instance of MPDFtransformer to generate
    the normalized mPDF and intermediate functions.

    Args:
        q (np array): Momentum transfer for which scattered intensities
            have been measured. Need not be uniform.
        iq (np array): Scattered intensity.
        ff (np array): Magnetic form factor on the same grid as q.
        qmin (float): Minimum q value included in the Fourier transform.
            In inverse Angstroms.
        qmax (float): Maximum q value included in the Fourier transform.
            In inverse Angstroms.
        qmaxinst (float): Maximum q value included in the background fit.
            In inverse Angstroms.
        rpoly (float): real-space distance below which artifacts may be
            introduced; used with qmaxinst to determin polynomial degree.
            In Angstroms.
        diq (np array): Error bars corresponding to the scattered intensity.
        qstart (float): Starting q-value for which the squared form factor
            will be fit to the data. Fitting range is [qstart, qmaxinst].
            In inverse Angstroms.
        rmin (float): Minimum r value for the mPDF data. In Angstroms.
        rmax (float): Maximum r value for the mPDF data. In Angstroms.
        rstep (float): Step size for the r-grid for the mPDF data. In
            Angstroms.
        window (str): specifies what type of window function to use. Default
            is 'None'. Other options are 'FD' for the Fermi-Dirac window
            and 'Lorch' for the Lorch window.
        qmaxwindow (float): Sets the q value where the window goes to 0.
        windowedgewidth (float): approximate width of Fermi-Dirac window.
            Not applicable to 'Lorch' or 'None' window options.

    Returns:
        Dictionary with the following keys and values:
        'r' : r-grid of the resulting mPDF data.
        'gmag': the resulting normalized mPDF data.
        'sqm': magnetic structure function (no corrections applied)
        'dsqm': estimated uncertainties for sqm.
        'fqc': corrected reduced magnetic structure function (this is what
            gets Fourier transformed).
        'fqc_prewindow': corrected reduced magnetic structure function
            before any window function is applied.
        'fqm': measured (i.e. uncorrected) reduced magnetic structure function.
        'dfqm': estimated uncertainties for fqm.
        'windowFunc': the window function used.
        'ff2': the scaled, squared magnetic form factor.
        'bkg': the final polynomial background used.
        'bkga': the lower-degree polynomial background.
        'bkgb': the higher-degree polynomial background.
        'mask1': mask used to scale the squared form factor.
        'mask2': mask used to fit the polynomial background.
        'mask3': mask used when computing the Fourier transform.
    """
    
    ### Step 1: Fit squared form factor to asymptotic behavior of I(Q)
    
    # set uncertainties to unity if not provided
    if diq is None:
        diq = np.ones_like(q)
    if (diq == np.array([0])).all():
        diq = np.ones_like(q)
    
    # set nans and zeros in diq to large numbers so they have limited weight
    # in the fits
    diq[diq==0.0] = 1000*np.nanmax(diq)
    diq = np.nan_to_num(diq, nan=1000*np.nanmax(diq))

    mask1 = np.logical_and(q>=qstart, q<=qmaxinst)
    opt1 = least_squares(resid1, [1], args=(ff[mask1], iq[mask1], diq[mask1]))

    scl = opt1.x[0]

    ff = scl * ff

    # create a dictionary containing the items to be returned and add the
    # scaled squared form factor and initial mask to it.
    rv = {}
    rv['ff2'] = ff**2
    rv['mask1'] = mask1

    ### Step 2: Obtain the measured F(Q), labeled fqm
    sqm = iq / ff**2
    dsqm = diq / ff**2
    fqm = q * (sqm - 1)
    dfqm = q * dsqm
    rv['sqm'] = sqm
    rv['dsqm'] = dsqm
    rv['fqm'] = fqm
    rv['dfqm'] = dfqm


    ### Step 3: Fit polynomial background to fqm
    nr = rpoly * qmaxinst / np.pi # non-integer polynomial degree
    nlow = int(np.floor(nr))
    nhigh = int(np.ceil(nr))
    weight_low = nr - nlow
    weight_high = nhigh - nr
    
    mask2 = np.logical_and(q>=qmin, q<=qmaxinst)
    
    # fit with lower degree
    p02a = np.ones(nlow+1)
    opt2a = least_squares(resid2, p02a, args=(q[mask2], fqm[mask2], dfqm[mask2]))

    bkga = bkgfunc(opt2a.x, q)

    # fit with higher degree
    p02b = np.ones(nhigh+1)
    opt2b = least_squares(resid2, p02b, args=(q[mask2], fqm[mask2], dfqm[mask2]))

    bkgb = bkgfunc(opt2b.x, q)

    # now generate the weighted background
    bkg = weight_low * bkga + weight_high * bkgb
    
    # final calculated F(Q)
    fqc = fqm - bkg

    # add a bunch of other stuff to the return dictionary.
    rv['mask2'] = mask2
    rv['bkga'] = bkga
    rv['bkgb'] = bkgb
    rv['bkg'] = bkg
    rv['fqc_prewindow'] = 1.0 * fqc

    if window == 'Lorch':
        windowFunc = window_Lorch(q, qmaxwindow)
        fqc *= windowFunc
    
    elif window == 'FD':
        windowFunc = window_FD(q, qmaxwindow, windowedgewidth)
        fqc *= windowFunc

    else:
        windowFunc = np.zeros_like(q) + 1.0

    rv['windowFunc'] = windowFunc
    rv['fqc'] = fqc
    

    ### Step 4: Compute the Fourier transform
    mask3 = np.logical_and(q>=qmin, q<=qmax)

    # normalized mPDF
    r, gmag = sinTransformDirectIntegration(q[mask3], fqc[mask3], rmin=rmin, rmax=rmax, rstep=rstep)

    rv['r'] = r
    rv['gmag'] = gmag
    rv['mask3'] = mask3

    return rv

class MPDFtransformer:
    """Control the Fourier transform parameters to produce the mPDF.


    This class is used to generate mPDF data (both normalized and unnormalized)
    from magnetic scattering data. For the normalized mPDF, the magnetic form
    factor must be provided, and an ad hoc polynomial background correction is
    applied to minimize slowly varying errors at high q. The algorithm closely
    follows that used in PDFgetX3 for x-ray PDF data.

    Args:
        q (np array): Momentum transfer for which scattered intensities
            have been measured. Need not be uniform.
        iq (np array): Scattered intensity.
        ff (np array): Magnetic form factor on the same grid as q.
        qmin (float): Minimum q value included in the Fourier transform.
            In inverse Angstroms.
        qmax (float): Maximum q value included in the Fourier transform.
            In inverse Angstroms.
        qmaxinst (float): Maximum q value included in the background fit.
            In inverse Angstroms.
        rpoly (float): real-space distance below which artifacts may be
            introduced; used with qmaxinst to determin polynomial degree.
            In Angstroms.
        diq (np array): Error bars corresponding to the scattered intensity.
        qstart (float): Starting q-value for which the squared form factor
            will be fit to the data. Fitting range is [qstart, qmaxinst].
            In inverse Angstroms.
        rmin (float): Minimum r value for the mPDF data. In Angstroms.
        rmax (float): Maximum r value for the mPDF data. In Angstroms.
        rstep (float): Step size for the r-grid for the mPDF data. In
            Angstroms.
        window (str): specifies what type of window function to use. Default
            is 'None'. Other options are 'FD' for the Fermi-Dirac window
            and 'Lorch' for the Lorch window.
        qmaxwindow (float): Sets the q value where the window goes to 0.
        windowedgewidth (float): approximate width of Fermi-Dirac window.
            Not applicable to 'Lorch' or 'None' window options.

    Properties:
        r (np array): r grid of the mPDF.
        dmag (np array): Unnormalized mPDF.
        gmag (np array): Normalized mPDF.
        sqm (np array): Magnetic structure function (no corrections applied)
        dsqm (np array): Estimated uncertainties for sqm.
        fqc (np array): Corrected reduced magnetic structure function
            (this is what gets Fourier transformed).
        fqc_prewindow (np array): Corrected reduced magnetic structure
            function before any window function is applied.
        fqm (np array): Measured (i.e. uncorrected) reduced magnetic structure
            function without any polynomial correction.
        dfqm (np array): Estimated uncertainties for fqm.
        windowFunc (np array): The window function used.
        ff2 (np array): The scaled, squared magnetic form factor.
        bkg (np array): The final polynomial background used.
        bkga (np array): The lower-degree polynomial background.
        bkgb (np array): The higher-degree polynomial background.
        mask1 (np array): Mask used to scale the squared form factor.
        mask2 (np array): Mask used to fit the polynomial background.
        mask3 (np array): Mask used when computing the Fourier transform.
        unnormalized_done (boolean): True if the unnormalized mPDF has
            been generated.
        normalized_done (boolean): True if the normalized mPDF has
            been generated.
    """
    def __init__(self, q=None, iq=None, ff=None, qmin=0.0, qmax=10.0,
                 qmaxinst=None, rpoly=1.8, diq=None, qstart=3.0, rmin=0.05,
                 rmax=100.0, rstep=0.01, window=None, qmaxwindow=None,
                 windowedgewidth=0.1):
        if q is None:
            self.q = np.array([0])
        else:
            self.q = q
        if iq is None:
            self.iq = np.array([0])
        else:
            self.iq = iq
        if ff is None:
            self.ff = np.array([0])
        else:
            self.ff = ff
        self.qmin = qmin
        self.qmax = qmax
        if qmaxinst is None:
            self.qmaxinst = 1.0*qmax
        else:
            self.qmaxinst = qmaxinst
        self.rpoly = rpoly
        if diq is None:
            self.diq = np.array([0])
        else:
            self.diq = diq
        self.qstart = qstart
        self.rmin = rmin
        self.rmax = rmax
        self.rstep = rstep
        if window is None:
            self.window = 'None'
        else:
            self.window = window
        if qmaxwindow is None:
            self.qmaxwindow = 1.0*qmax
        else:
            self.qmaxwindow = qmaxwindow
        self.windowedgewidth = windowedgewidth
        self._r = np.array([])
        self._dmag = np.array([])
        self._gmag = np.array([])
        self._sqm = np.array([])
        self._dsqm = np.array([])
        self._fqm = np.array([])
        self._fqc = np.array([])
        self._fqc_prewindow = np.array([])
        self._dfqm = np.array([])
        self._windowFunc = np.array([])
        self._ff2 = np.array([])
        self._bkg = np.array([])
        self._bkga = np.array([])
        self._bkgb = np.array([])
        self._mask1 = np.array([])
        self._mask2 = np.array([])
        self._mask3 = np.array([])
        self._unnormalized_done = False
        self._normalized_done = False

    def __repr__(self):
        info = 'MPDFtransformer instance: \n'
        info += 'qmin = ' + str(self.qmin) + '\n'
        info += 'qmax = ' + str(self.qmax) + '\n'
        info += 'qmaxinst = ' + str(self.qmaxinst) + '\n'
        info += 'rpoly = ' + str(self.rpoly) + '\n'
        info += 'rmin = ' + str(self.rmin) + '\n'
        info += 'rmax = ' + str(self.rmax) + '\n'
        info += 'rstep = ' + str(self.rstep) + '\n'
        info += 'window = ' + self.window + '\n'
        return info

    @property
    def r(self):
        """r grid of the mPDF."""
        return self._r

    @property
    def dmag(self):
        """Unnormalized mPDF."""
        return self._dmag

    @property
    def gmag(self):
        """Normalized mPDF."""
        return self._gmag

    @property
    def sqm(self):
        """Magnetic structure function."""
        return self._sqm

    @property
    def dsqm(self):
        """Estimated uncertainties for sqm."""
        return self._dsqm

    @property
    def fqm(self):
        """Uncorrected reduced magnetic structure function."""
        return self._fqm

    @property
    def dfqm(self):
        """Estimated uncertainties for fqm."""
        return self._dfqm

    @property
    def fqc(self):
        """Corrected reduced magnetic structure function."""
        return self._fqc

    @property
    def fqc_prewindow(self):
        """fqc but before any window function has been applied."""
        return self._fqc_prewindow

    @property
    def windowFunc(self):
        """The window function used for fqc."""
        return self._windowFunc

    @property
    def ff2(self):
        """Scaled, squared magnetic form factor."""
        return self._ff2

    @property
    def bkg(self):
        """Polynomial background used in the correction."""
        return self._bkg

    @property
    def bkga(self):
        """Lower-degree polynomial background."""
        return self._bkga

    @property
    def bkgb(self):
        """Higher-degree polynomial background."""
        return self._bkgb

    @property
    def mask1(self):
        """Mask used to scale the squared form factor."""
        return self._mask1

    @property
    def mask2(self):
        """Mask used to fit the polynomial background."""
        return self._mask2

    @property
    def mask3(self):
        """Mask used when computing the Fourier transform."""
        return self._mask3

    @property
    def unnormalized_done(self):
        """True if the unnormalized mPDF has been generated."""
        return self._unnormalized_done

    @property
    def normalized_done(self):
        """True if the normalized mPDF has been generated."""
        return self._normalized_done

    def getmPDF(self, type='normalized'):
        """Generate the magnetic PDF.

        Args:
            type (str): must be 'normalized' or 'unnormalized'.
                'normalized': will generate the normalized mPDF.
                'unnormalized': will generate the unnormalized mPDF.
        Returns: Doesn't return anything, but populates the relevant
            MPDFtransformer properties.
        """
        if type not in ['normalized', 'unnormalized']:
            print('Please choose normalized or unnormalized.')
        elif type == 'unnormalized':
            r, dmag = getmPDF_unnormalized(self.q, self.iq, self.qmin,
                                           self.qmax, self.rmin, self.rmax,
                                           self.rstep)
            self._r = r
            self._dmag = dmag
            self._unnormalized_done = True
        elif type == 'normalized':
            output = getmPDF_normalized(self.q, self.iq, self.ff, self.qmin,
                                        self.qmax, self.qmaxinst, self.rpoly,
                                        1.0*self.diq, self.qstart, self.rmin,
                                        self.rmax, self.rstep, self.window,
                                        self.qmaxwindow, self.windowedgewidth)
            self._r = output['r']
            self._gmag = output['gmag']
            self._sqm = output['sqm']
            self._dsqm = output['dsqm']
            self._fqc = output['fqc']
            self._fqc_prewindow = output['fqc_prewindow']
            self._fqm = output['fqm']
            self._dfqm = output['dfqm']
            self._windowFunc = output['windowFunc']
            self._ff2 = output['ff2']
            self._bkg = output['bkg']
            self._bkga = output['bkga']
            self._bkgb = output['bkgb']
            self._mask1 = output['mask1']
            self._mask2 = output['mask2']
            self._mask3 = output['mask3']
            self._normalized_done = True

    def makePlots(self, showuncertainty=True):
        """Generate plots from the data processing.
        
        showuncertainty (boolean): If true, the uncertainty associated with
            each data point in Q will be displayed."""
        w = 6
        h = 3

        plotsReady = False

        # plot of scattered intensity
        if len(self.q) == len(self.iq) > 1:
            fig = plt.figure(figsize=(w, h))
            ax = fig.add_subplot(111)
            ax.set_title('Intensity vs. Q')
            if (len(self.diq) == len(self.iq)) and showuncertainty:
                ax.errorbar(self.q, self.iq, yerr=self.diq,
                            marker='.', linestyle='-', color='k')
            else:
                ax.plot(self.q, self.iq, marker='.', linestyle='-', color='k')
            if len(self._ff2) == len(self.q):
                ax.plot(self.q, self._ff2, color='Gray', linestyle='--')
            ax.set_xlabel(r'Q ($\mathdefault{\AA^{-1}}$)')
            ax.set_ylabel(r' Intensity (arb. units)')
            plt.tight_layout()
            plt.draw()
            plotsReady = True

        # plots for the normalized mPDF
        if self._normalized_done:
            # plot of sqm
            fig = plt.figure(figsize=(w, h))
            ax = fig.add_subplot(111)
            ax.set_title(r'S$_{\mathdefault{m}}$ vs. Q')
            msk = self._mask3
            if showuncertainty:
                ax.errorbar(self.q[msk], self._sqm[msk], yerr=self._dsqm[msk],
                            marker='.', linestyle='-', color='k')
            else:
                ax.plot(self.q[msk], self._sqm[msk], marker='.',
                        linestyle='-', color='k')
            ax.set_xlabel(r'Q ($\mathdefault{\AA^{-1}}$)')
            ax.set_ylabel(r'S$_{\mathdefault{m}}$')
            plt.tight_layout()
            plt.draw()

            # plot of fqm with the polynomial background
            fig = plt.figure(figsize=(w, h))
            ax = fig.add_subplot(111)
            ax.set_title(r'F$_{\mathdefault{m}}$ vs. Q')
            msk = self._mask3
            if showuncertainty:
                ax.errorbar(self.q[msk], self._fqm[msk], yerr=self._dfqm[msk],
                            marker='.', linestyle='-', color='k')
            else:
                ax.plot(self.q[msk], self._fqm[msk],
                            marker='.', linestyle='-', color='k')
            ax.plot(self.q[msk], self._bkg[msk], color='Gray', linestyle='--')
            ax.set_xlabel(r'Q ($\mathdefault{\AA^{-1}}$)')
            ax.set_ylabel(r'F$_{\mathdefault{m}}$')
            plt.tight_layout()
            plt.draw()

            # plot of fqc
            fig = plt.figure(figsize=(w, h))
            ax = fig.add_subplot(111)
            ax.set_title(r'F$_{\mathdefault{c}}$ vs. Q')
            msk = self._mask3
            if showuncertainty:
                ax.errorbar(self.q[msk], self._fqc[msk], yerr=self._dfqm[msk],
                            marker='.', linestyle='-', color='k')
            else:
                ax.plot(self.q[msk], self._fqc[msk],
                            marker='.', linestyle='-', color='k')            
            if self.window in ['FD', 'Lorch']:
                ax.plot(self.q[msk], self._windowFunc[msk] * 0.8 * \
                        np.max(self._fqc[msk]), color='Gray', linestyle='--')
            ax.set_xlabel(r'Q ($\mathdefault{\AA^{-1}}$)')
            ax.set_ylabel(r'F$_{\mathdefault{c}}$')
            plt.tight_layout()
            plt.draw()

            # plot of gmag
            fig = plt.figure(figsize=(w, h))
            ax = fig.add_subplot(111)
            ax.set_title(r'G$_{\mathdefault{mag}}$ vs. r')
            ax.plot(self._r, self._gmag)
            ax.set_xlabel(r'r ($\mathdefault{\AA}$)')
            ax.set_ylabel(r'G$_{\mathdefault{mag}}$ ($\mathdefault{\AA^{-2}}$)')
            plt.tight_layout()
            plt.draw()
            plotsReady = True

        # plot of unnormalized mPDF
        if self._unnormalized_done:
            # plot of gmag
            fig = plt.figure(figsize=(w, h))
            ax = fig.add_subplot(111)
            ax.set_title(r'd$_{\mathdefault{mag}}$ vs. r')
            ax.plot(self._r, self._dmag)
            ax.set_xlabel(r'r ($\mathdefault{\AA}$)')
            ax.set_ylabel(r'd$_{\mathdefault{mag}}$ (arb. units)')
            plt.tight_layout()
            plt.draw()
            plotsReady = True

        if plotsReady:
            plt.show()

    def copy(self):
        """Return a deep copy of the MPDFtransformer object."""
        return copy.deepcopy(self)
