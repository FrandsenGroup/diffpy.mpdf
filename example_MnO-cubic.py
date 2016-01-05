import numpy as np
import matplotlib.pyplot as plt
from mcalculator import calculateMPDF, j0calc, cv, generateAtomsXYZ

# DiffPy-CMI modules for building a fitting recipe
from diffpy.Structure import loadStructure

# Files containing our experimental data and structure file
structureFile = "mPDF_exampleFiles/MnO_cubic.cif"

# Create the structure from our cif file
MnOStructure = loadStructure(structureFile)
magIdxs=[0,1,2,3]
rmax=30.0

aXYZ=generateAtomsXYZ(MnOStructure,rmax,magIdxs)


### create spins using propagation vector
astar,bstar,cstar=getRlat(cell[0],cell[1],cell[2])
Q=0.0*astar+1.0*bstar+1.0*cstar
signs=np.cos(np.dot(atoms-atoms[0],Q))
spins=S*svec*signs.reshape(-1,1)