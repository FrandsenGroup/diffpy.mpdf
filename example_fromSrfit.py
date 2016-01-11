#!/usr/bin/env python
# -*- coding: utf-8 -*-


# We'll need numpy and pylab for plotting our results
import numpy as np
#import pylab
import matplotlib.pyplot as plt

# A least squares fitting algorithm from scipy
from scipy.optimize.minpack import leastsq

# DiffPy-CMI modules for building a fitting recipe
from diffpy.Structure import loadStructure
from diffpy.srfit.pdf import PDFContribution
from diffpy.srfit.fitbase import FitRecipe, FitResults

# Load the mPDF calculator modules
from mcalculator import *


# Files containing our experimental data and structure file
dataFile = "mPDF_exampleFiles/npdf_07334.gr"
structureFile = "mPDF_exampleFiles/MnO_R-3m.cif"
spaceGroup = "R-3m"

# The first thing to construct is a contribution. Since this is a simple
# example, the contribution will simply contain our PDF data and an associated
# structure file. We'll give it the name "nickel"
MnOPDF = PDFContribution("MnO")

# Load the data and set the r-range over which we'll fit
MnOPDF.loadData(dataFile)
MnOPDF.setCalculationRange(xmin=0.01, xmax=20, dx=0.01)

# Add the structure from our cif file to the contribution
MnOStructure = loadStructure(structureFile)
MnOPDF.addStructure("MnO", MnOStructure)

# The FitRecipe does the work of calculating the PDF with the fit variable
# that we give it.
MnOFit = FitRecipe()

# give the PDFContribution to the FitRecipe
MnOFit.addContribution(MnOPDF)

# Configure the fit variables and give them to the recipe.  We can use the
# srfit function constrainAsSpaceGroup to constrain the lattice and ADP
# parameters according to the R-3m space group.
from diffpy.srfit.structure import constrainAsSpaceGroup
spaceGroupParams = constrainAsSpaceGroup(MnOPDF.MnO.phase, spaceGroup)
print "Space group parameters are:",
print ', '.join([p.name for p in spaceGroupParams])
print

# We can now cycle through the parameters and activate them in the recipe as
# variables
for par in spaceGroupParams.latpars:
    MnOFit.addVar(par)
# Set initial value for the ADP parameters, because CIF had no ADP data.
for par in spaceGroupParams.adppars:
    MnOFit.addVar(par, value=0.003,fixed=True)

# As usual, we add variables for the overall scale of the PDF and a delta2
# parameter for correlated motion of neighboring atoms.
MnOFit.addVar(MnOPDF.scale, 1)
MnOFit.addVar(MnOPDF.MnO.delta2, 1.5)

# We fix Qdamp based on prior information about our beamline.
MnOFit.addVar(MnOPDF.qdamp, 0.03, fixed=True)

# Turn off printout of iteration number.
MnOFit.clearFitHooks()

# We can now execute the fit using scipy's least square optimizer.
print "Refine PDF using scipy's least-squares optimizer:"
print "  variables:", MnOFit.names
print "  initial values:", MnOFit.values
leastsq(MnOFit.residual, MnOFit.values)
print "  final values:", MnOFit.values
print

# Obtain and display the fit results.
MnOResults = FitResults(MnOFit)
print "FIT RESULTS\n"
print MnOResults

# Plot the observed and refined PDF.

# Get the experimental data from the recipe
r = MnOFit.MnO.profile.x
gobs = MnOFit.MnO.profile.y

# Get the calculated PDF and compute the difference between the calculated and
# measured PDF
gcalc = MnOFit.MnO.evaluate()
baseline = 1.1 * gobs.min()
gdiff = gobs - gcalc
baseline2 = 1.1 * (gdiff+baseline).min()

# Do the mPDF fit

# Set up the mPDF calculator
mc=mPDFcalculator(struc=MnOStructure,magIdxs=[0,1,2],rmin=0.01,rmax=20.0,gaussPeakWidth=0.2)

mc.makeAtoms()
mc.svec=2.5*np.array([1.0,-1.0,0])/np.sqrt(2)
mc.kvec=np.array([0,0,1.5])
mc.spinOrigin=mc.atoms[0]
mc.makeSpins()
mc.ffqgrid=np.arange(0,10,0.01)
mc.ff=j0calc(mc.ffqgrid,[0.422,17.684,0.5948,6.005,0.0043,-0.609,-0.0219])
mc.calcList=np.arange(1)

# Do the refinement
mc.rmin=r.min()
mc.rmax=r.max()
def residual(p,yexp,mcalc):
    mcalc.paraScale,mcalc.ordScale=p
    return yexp-mcalc.calc(both=True)[2]

p0=[5.0,3.0]
pOpt=leastsq(residual,p0,args=(gdiff,mc))
print pOpt

fit=mc.calc(both=True)[2]

# Plot!
ax=plt.figure().add_subplot(111)
ax.plot(r, gobs, 'bo', label="G(r) data",markerfacecolor='none', markeredgecolor='b')
ax.plot(r, gcalc, 'r-', lw=1.5, label="G(r) fit")
ax.plot(r, gdiff + baseline,mfc='none',mec='b',marker='o')
ax.plot(r,fit+baseline,'r-',lw=1.5)
ax.plot(r,gdiff-fit+baseline2,'g-')
ax.plot(r, np.zeros_like(r) + baseline2, 'k:')
ax.set_xlabel(r"r ($\AA$)")
ax.set_ylabel(r"G ($\AA^{-2}$)")
plt.legend()

plt.show()
