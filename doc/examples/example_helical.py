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
mnostructure = loadStructure(structureFile)
mnostructure.lattice.a=3.0
mnostructure.lattice.b=150.0
mnostructure.lattice.c=150.0

# Create a magnetic structure object
helixSpec=magSpecies(mnostructure)

Sk=0.5*(np.array([0,0,1])+0.5j*np.array([0,1,0]))
helixSpec.basisvecs=np.array([Sk,Sk.conj()])
helixSpec.kvecs=np.array([[np.sqrt(2)/100,0,0],[-1.0*np.sqrt(2)/100,0,0]])
helixSpec.makeAtoms()
helixSpec.makeSpins()

x,y,z=np.transpose(helixSpec.atoms)
sx,sy,sz=np.transpose(helixSpec.spins)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(x,sz,'r.',x,sy,'b.',x,sx,'k.')
plt.show()

