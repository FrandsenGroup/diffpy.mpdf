#!/usr/bin/env python
# -*- coding: utf-8 -*-


# We'll need numpy and pylab for plotting our results
import numpy as np
#import pylab
import matplotlib.pyplot as plt

# A least squares fitting algorithm from scipy
from scipy.optimize.minpack import leastsq

# DiffPy-CMI modules for building a fitting recipe
from diffpy.Structure.Parsers import getParser
from diffpy.srfit.pdf import PDFGenerator, PDFParser
from diffpy.srfit.fitbase import FitRecipe, FitResults
from diffpy.srfit.fitbase import Profile, FitContribution

# Load the mPDF calculator modules
from mcalculator import *


# Files containing our experimental data and structure file
dataFile = "mPDF_exampleFiles/npdf_07334.gr"
structureFile = "mPDF_exampleFiles/MnO_R-3m.cif"

# load structure and space group from the CIF file
pcif = getParser('cif')
mno2 = pcif.parseFile(structureFile)

# prepare profile object with experimental data
profile = Profile()
parser = PDFParser()
parser.parseFile(dataFile)
profile.loadParsedData(parser)

# define range for pdf calculation
rmin = 0.01
rmax = 20
rstep = 0.01

# setup calculation range for the PDF simulation
profile.setCalculationRange(xmin=rmin, xmax=rmax, dx=rstep)

# prepare nucpdf function that simulates the nuclear PDF
nucpdf = PDFGenerator("nucpdf")
nucpdf.setStructure(mno2)
# apply metadata from dataFile such as neutron/x-ray type of the PDF
nucpdf.setProfile(profile)

# prepare magpdf function that simulates the magnetic PDF

# Set up the mPDF calculator.  This is calculation engine for magpdf.
mc=mPDFcalculator(mno2, magIdxs=[0,1,2],
        rmin=rmin, rmax=rmax, rstep=rstep, gaussPeakWidth=0.2)
mc.svec=2.5*np.array([1.0,-1.0,0])/np.sqrt(2)
mc.kvec=np.array([0,0,1.5])
mc.spinOrigin=np.array([0,0,0])
mc.ffqgrid=np.arange(0,10,0.01)
mc.ff=jCalc(mc.ffqgrid,getFFparams('Mn2'))
mc.calcList=np.arange(1)
mc.makeAtoms()
mc.makeSpins()

def magpdf(parascale, ordscale):
    mc.paraScale = parascale
    mc.ordScale = ordscale
    mc.makeAtoms()
    mc.makeSpins()
    rv = mc.calc(both=True)[2]
    return rv

totpdf = FitContribution('totpdf')
totpdf.addProfileGenerator(nucpdf)
totpdf.registerFunction(magpdf)
totpdf.setProfile(profile)
totpdf.setEquation("nucscale * nucpdf + magpdf(parascale, ordscale)")

# Make magnetic PDF depend on any changes to the atomic structure.
# Cover your eyes, but a structure change will now behave as ordscale
# modification and will forget cached values of magpdf.
for o in totpdf.ordscale._observers:
    nucpdf.phase.addObserver(o)

# The FitRecipe does the work of calculating the PDF with the fit variable
# that we give it.
mnofit = FitRecipe()

# give the PDFContribution to the FitRecipe
mnofit.addContribution(totpdf)

# Configure the fit variables and give them to the recipe.  We can use the
# srfit function constrainAsSpaceGroup to constrain the lattice and ADP
# parameters according to the CIF-loaded space group.
from diffpy.srfit.structure import constrainAsSpaceGroup
sgpars = constrainAsSpaceGroup(nucpdf.phase, pcif.spacegroup.short_name)
print "Space group parameters are:",
print ', '.join([p.name for p in sgpars])
print

# We can now cycle through the parameters and activate them in the recipe as
# variables
for par in sgpars.latpars:
    mnofit.addVar(par)
# Set initial value for the ADP parameters, because CIF had no ADP data.
for par in sgpars.adppars:
    mnofit.addVar(par, value=0.003, fixed=True)

# As usual, we add variables for the overall scale of the PDF and a delta2
# parameter for correlated motion of neighboring atoms.
mnofit.addVar(totpdf.nucscale, 1)
mnofit.addVar(nucpdf.delta2, 1.5)

# We fix Qdamp based on prior information about our beamline.
mnofit.addVar(nucpdf.qdamp, 0.03, fixed=True)

# variables for magnetic scattering
mnofit.addVar(totpdf.parascale, 4)
mnofit.addVar(totpdf.ordscale, 1.5)

# Turn off printout of iteration number.
mnofit.clearFitHooks()

# Initial structural fit
print "Refine PDF using scipy's least-squares optimizer:"
print "  variables:", mnofit.names
print "  initial values:", mnofit.values
leastsq(mnofit.residual, mnofit.values)
print "  final values:", mnofit.values
print
# Obtain and display the fit results.
mnoresults = FitResults(mnofit)
print "FIT RESULTS\n"
print mnoresults

# Plot the observed and refined PDF.

# Get the experimental data from the recipe
r = mnofit.totpdf.profile.x
gobs = mnofit.totpdf.profile.y

# Get the calculated PDF and compute the difference between the calculated and
# measured PDF
gcalc = mnofit.totpdf.evaluate()
gnuc = mnofit.totpdf.evaluateEquation('nucscale * nucpdf')
gmag = mnofit.totpdf.evaluateEquation('magpdf')

baseline = 1.1 * gobs.min()
gdiff = gobs - gcalc
baseline2 = 1.1 * (gdiff+baseline).min()
magfit=mc.calc(both=True)[2]

# Plot!
ax=plt.figure().add_subplot(111)
ax.plot(r, gobs, 'bo', label="G(r) data",markerfacecolor='none', markeredgecolor='b')
ax.plot(r, gcalc, 'r-', lw=1.5, label="G(r) fit")
ax.plot(r, gnuc, 'k-', lw=1.5, label="G(r) nuclear")
ax.plot(r, gmag, 'g-', lw=1.5, label="G(r) magnetic")

#ax.plot(r, gdiff + baseline,mfc='none',mec='b',marker='o')
#ax.plot(r, gmag+baseline,'r-',lw=1.5)
#ax.plot(r,gdiff-magfit+baseline2,'g-')
#ax.plot(r, np.zeros_like(r) + baseline2, 'k:')
ax.set_xlabel(r"r ($\AA$)")
ax.set_ylabel(r"G ($\AA^{-2}$)")
plt.legend()

plt.show()
