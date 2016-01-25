import numpy as np
import matplotlib.pyplot as plt
import diffpy as dp
import sys

# DiffPy-CMI modules for building a fitting recipe
from diffpy.Structure import loadStructure

import sys
sys.path.append('/home/ben/mPDFmodules/mpdfcalculator')
from mcalculator import *

# Files containing our experimental data and structure file
structureFile = "MnO_cubic.cif"

# Create the structure from our cif file
#
MnOStructure = loadStructure(structureFile)
magIdxs=[0,1,2,3]
rmax=30.0
aXYZ=generateAtomsXYZ(MnOStructure,rmax,magIdxs)

svec=2.5*np.array([1.0,-1.0,0])/np.sqrt(2)
k=np.array([0.5,0.5,0.5])
sXYZ=generateSpinsXYZ(MnOStructure,aXYZ,aXYZ[0],k,svec)

# Calculate the mPDF
r,fr=calculatemPDF(aXYZ,sXYZ,rmax=30,psigma=0.15,qmin=0.1,qmax=30.)

# Calculate the unnormalized mPDF D(r)
q=np.arange(0,10,0.01)
ff=jCalc(q,getFFparams('Mn2'))
Dr=calculateDr(r,fr,q,ff)

# Plot the mPDF
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(r,fr,'r-',r,Dr,'b-')
ax.set_xlabel('r ($\AA$)')
ax.set_ylabel('f ($\AA^{-2}$)')

plt.show()


# Now an alternative way to do it using the mPDFcalculator class:
mc=mPDFcalculator(struc=MnOStructure,magIdxs=[0,1,2,3],rmax=30.0,svec=2.5*np.array([1.0,-1.0,0])/np.sqrt(2),kvec=np.array([0.5,0.5,0.5]),ffqgrid=np.arange(0,10,0.01),ff=jCalc(q,getFFparams('Mn2')),gaussPeakWidth=0.15)
mc.makeAtoms()
mc.spinOrigin=mc.atoms[0]
mc.makeSpins()

r,fr,Dr=mc.calc(both=True)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(r,fr,'r-',r,Dr,'b-')
ax.set_xlabel('r ($\AA$)')
ax.set_ylabel('f ($\AA^{-2}$)')

plt.show()

