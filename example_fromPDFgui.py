import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from mcalculator import *
import diffpy as dp
from diffpy.Structure import loadStructure

# Create the structure from our cif file, update the lattice params
structureFile = "mPDF_exampleFiles/MnO_R-3m.cif"
MnOStructure = loadStructure(structureFile)
lat=MnOStructure.lattice
lat.a,lat.b,lat.c=3.1505626,3.1505626,7.5936979
MnOStructure.lattice=lat

# Set up the mPDF calculator
mc=mPDFcalculator(struc=MnOStructure,magIdxs=[0,1,2],rmax=20.0,gaussPeakWidth=0.2)

mc.makeAtoms()
mc.svec=2.5*np.array([1.0,-1.0,0])/np.sqrt(2)
mc.kvec=np.array([0,0,1.5])
mc.spinOrigin=mc.atoms[0]
mc.makeSpins()
mc.ffqgrid=np.arange(0,10,0.01)
mc.ff=j0calc(mc.ffqgrid,[0.422,17.684,0.5948,6.005,0.0043,-0.609,-0.0219])
mc.calcList=np.arange(1)

# Load the data
PDFfitFile='mPDF_exampleFiles/MnOfit_PDFgui.fgr'
rexp,Drexp=getDiffData([PDFfitFile])
mc.rmin=rexp.min()
mc.rmax=rexp.max()

# Do the refinement
def residual(p,yexp,mcalc):
    mcalc.paraScale,mcalc.ordScale=p
    return yexp-mcalc.calc(both=True)[2]

p0=[5.0,3.0]
pOpt=leastsq(residual,p0,args=(Drexp,mc))
print pOpt

fit=mc.calc(both=True)[2]

# Plot the results
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(rexp,Drexp,marker='o',mfc='none',mec='b',linestyle='none')
ax.plot(rexp,fit,'r-',lw=2)
ax.set_xlim(xmin=mc.rmin,xmax=mc.rmax)
ax.set_xlabel('r ($\AA$)')
ax.set_ylabel('d(r) ($\AA^{-2}$)')

plt.show()


