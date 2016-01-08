import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from mcalculator import *
import diffpy as dp
from diffpy.Structure import loadStructure

# Create the structure from our cif file, update the lattice params
MnOrhomb = loadStructure("mPDF_exampleFiles/MnO_R-3m.cif")
MnOcubic = loadStructure("mPDF_exampleFiles/MnO_cubic.cif")
latCub=MnOcubic.lattice
latRhomb=MnOrhomb.lattice
latRhomb.a,latRhomb.b,latRhomb.c=latCub.a/np.sqrt(2),latCub.a/np.sqrt(2),np.sqrt(6)*latCub.a/np.sqrt(2)
#MnOrhomb.lattice=latRhomb

# Set up the mPDF calculators
mcCub=mPDFcalculator(MnOcubic)
mcCub.magIdxs=[0,1,2,3]
mcCub.makeAtoms()

mcRhomb=mPDFcalculator(MnOrhomb)
mcRhomb.magIdxs=[0,1,2]
mcRhomb.makeAtoms()

svec=2.5*np.array([1.0,-1.0,0])/np.sqrt(2)
mcCub.svec=svec
mcCub.kvec=np.array([0.5,0.5,0.5])
mcCub.spinOrigin=mcCub.atoms[0]
mcCub.makeSpins()

mcRhomb.svec=svec
mcRhomb.kvec=np.array([0,0,1.5])
mcRhomb.spinOrigin=mcRhomb.atoms[0]
mcRhomb.makeSpins()

mcCub.ffqgrid=np.arange(0,10,0.01)
mcCub.ff=j0calc(mcCub.ffqgrid,[0.422,17.684,0.5948,6.005,0.0043,-0.609,-0.0219])
mcRhomb.ffqgrid=np.arange(0,10,0.01)
mcRhomb.ff=j0calc(mcRhomb.ffqgrid,[0.422,17.684,0.5948,6.005,0.0043,-0.609,-0.0219])

rCub,frCub=mcCub.calc()
rRhomb,frRhomb=mcRhomb.calc()

# Plot the results
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(rCub,frCub,'r-')
ax.plot(rRhomb,frRhomb,'b.')
ax.set_xlim(xmin=mcCub.rmin,xmax=mcCub.rmax)
ax.set_xlabel('r ($\AA$)')
ax.set_ylabel('d(r) ($\AA^{-2}$)')

plt.show()

