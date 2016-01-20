import numpy as np
import matplotlib.pyplot as plt
from mcalculator import *
import diffpy as dp

# DiffPy-CMI modules for building a fitting recipe
from diffpy.Structure import loadStructure

# Files containing our experimental data and structure file
structureFile = "mPDF_exampleFiles/MnO_cubic.cif"

# Create the structure from our cif file
MnOStructure = loadStructure(structureFile)
magIdxs=[0,1,2,3]
rmax=25.0

svec=2.5*np.array([1.0,-1.0,0])/np.sqrt(2)
kvec=np.array([0.5,0.5,0.5])

# Calculate the unnormalized mPDF D(r)
q=np.arange(0,10,0.01)
ff=jCalc(q,getFFparams('Mn2'))

# Now an alternative way to do it using the mPDFcalculator class:
mc=mPDFcalculator(struc=MnOStructure,magIdxs=magIdxs,rmax=rmax,svec=svec,kvec=kvec,ffqgrid=q,ff=ff,gaussPeakWidth=0.15)
mc.atoms=generateAtomsXYZ(MnOStructure,rmax,magIdxs,square=True)
mc.spinOrigin=mc.atoms[0]
mc.makeSpins()

atoms=mc.atoms
spins=mc.spins

ax,ay,az=np.transpose(atoms)
xmin,xmax=np.floor(np.min(ax)/MnOStructure.lattice.a)*MnOStructure.lattice.a,(np.floor(np.max(ax)/MnOStructure.lattice.a)+1)*MnOStructure.lattice.a
ymin,ymax=np.floor(np.min(ay)/MnOStructure.lattice.b)*MnOStructure.lattice.b,(np.floor(np.max(ay)/MnOStructure.lattice.b)+1)*MnOStructure.lattice.b
zmin,zmax=np.floor(np.min(az)/MnOStructure.lattice.c)*MnOStructure.lattice.c,(np.floor(np.max(az)/MnOStructure.lattice.c)+1)*MnOStructure.lattice.c
ff=ff[q>0.2]
q=q[q>0.2]

q,iq=calculateIQinFold(atoms,spins,np.arange(len(atoms)),q,ff,xmin,xmax,ymin,ymax,zmin,zmax,0.02,1.0,0.4*(xmax-xmin))

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(q,iq)
ax.set_xlabel('q ($\AA^{-1}$)')
ax.set_ylabel('I ($\AA^{-2}$)')

plt.show()

