#!/usr/bin/env python
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, convolve
import diffpy as dp
import copy

def j0calc(q,params):
    """Calculate the magnetic form factor j0.

    This method for calculating the magnetic form factor is based on the
    approximate empirical forms based on the tables by Brown, consisting of
    the sum of 3 Gaussians and a constant. 
    
    Args:
        q (numpy array): 1-d grid of momentum transfer on which the form
            factor is to be computed
        params (python list): provides the 7 numerical coefficients    
    
    Returns:
        numpy array with same shape as q giving the magnetic form factor j0.
    """
    [A,a,B,b,C,c,D] = params
    return A*np.exp(-a*(q/4/np.pi)**2)+B*np.exp(-b*(q/4/np.pi)**2)+C*np.exp(-c*(q/4/np.pi)**2)+D

def cv(x1,y1,x2,y2):
    """Perform the convolution of two functions and give the correct output.

    Args:    
        x1 (numpy array): independent variable of first function; must be in
            ascending order
        y1 (numpy array): dependent variable of first function
        x2 (numpy array): independent variable of second function; must have
            same grid spacing as x1
        y2 (numpy array): dependent variable of second function

    Returns:
        xcv (numpy array): independent variable of convoluted function, has
            dimension len(x1) + len(x2) - 1
        ycv (numpy array): convolution of y1 and y2, same shape as xcv
    
    Returns: arrays ycv and xcv giving the convolution.
    """
    dx=x1[1]-x1[0]
    ycv = dx*np.convolve(y1,y2,'full')
    xcv=np.linspace(x1[0]+x2[0],x1[-1]+x2[-1],len(ycv))
    return xcv,ycv
    
def costransform(q,fq,rmin=0.0,rmax=50.0,rstep=0.1): # does not require even q-grid
    """Compute the cosine Fourier transform of a function.

    This method uses direct integration rather than an FFT and doesn't require
    an even grid. The grid for the Fourier transform is even and specifiable.
    
    Args:
        q (numpy array): independent variable for function to be transformed
        fq (numpy array): dependent variable for function to be transformed
        rmin (float, default=0.0): min value of conjugate independent variable
            grid
        rmax (float, default=50.0): maximum value of conjugate independent
            variable grid
        rstep (float, default=0.1): grid spacing for conjugate independent
            variable
    
    Returns: 
        r (numpy array): independent variable grid for transformed quantity
        fr (numpy array): cosine Fourier transform of fq
    """
    lostep = int(np.ceil((rmin - 1e-8) / rstep))
    histep = int(np.floor((rmax + 1e-8) / rstep)) + 1
    r = np.arange(lostep,histep)*rstep
    qrmat=np.outer(r,q)
    integrand=fq*np.cos(qrmat)
    fr=np.sqrt(2.0/np.pi)*np.trapz(integrand,q)
    return r,fr

	
def getDiffData(fileNames=[],fmt='pdfgui',writedata=False):
    """Extract the fit residual from a structural PDF fit.

    Args:
        fileNames (python list): list of paths to the files containing the
            fit information (e.g. calculated and experimental PDF, as in the
            .fgr files from PDFgui exported fits)
        fmt (string): string identifying the format of the file(s). Options
            are currently just 'pdfgui'.
        writedata (boolean): whether or not the output should be saved to file

    Returns:
        r (numpy array): same r-grid as contained in the fit file
        diff (numpy array): the structural PDF fit residual (i.e. the mPDF)
    """
    for name in fileNames:
        if fmt=='pdfgui':
            allcols = np.loadtxt(name,unpack=True,comments='#',skiprows=14)
            r,grcalc,diff=allcols[0],allcols[1],allcols[4]
            grexp = grcalc+diff
            if writedata:
                np.savetxt(name[:-4]+'.diff',np.transpose((r,diff)))
            else:
                return r,diff
        else:
            print 'This format is not currently supported.'
	
def calculatemPDF(xyz, sxyz, calcList=np.array([0]), rstep=0.01, rmin=0.0, rmax=20.0, psigma=0.1,qmin=0,qmax=-1,dampRate=0.0,dampPower=2.0,maxextension=10.0):
    """Calculate the normalized mPDF.
    
    At minimum, this module requires input lists of atomic positions and spins.
    
    Args:
        xyz (numpy array): list of atomic coordinates of all the magnetic
            atoms in the structure.
        sxyz (numpy array): triplets giving the spin vectors of all the 
            atoms, in the same order as the xyz array provided as input.
        calcList (python list): list giving the indices of the atoms array
            specifying the atoms to be used as the origin when calculating
            the mPDF.
        rstep (float): step size for r-grid of calculated mPDF.
        rmin (float): minimum value of r for which mPDF should be calculated.
        rmax (float): maximum value of r for which mPDF should be calculated.
        psigma(float): std deviation (in Angstroms) of Gaussian peak
            to be convoluted with the calculated mPDF to simulate thermal
            motion.
        qmin (float): minimum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.
        qmax (float): maximum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.
        dampRate (float): generalized ("stretched") exponential damping rate
                of the mPDF.
        dampPower (float): power of the generalized exponential function.
        maxextension (float): extension of the r-grid on which the mPDF is
            calculated to properly account for contribution of pairs just
            outside the boundary.

    Returns: numpy arrays for r and the mPDF fr.
        """
    
    # calculate s1, s2
    r = np.arange(rmin, rmax+maxextension+rstep, rstep)
    rbin =  np.concatenate([r-rstep/2, [r[-1]+rstep/2]])
    
    s1 = np.zeros(len(r))
    s2 = np.zeros(len(r))
    
    for i in range(len(calcList)):
        uu = calcList[i]
        
        ri = xyz0 = xyz[uu]
        rj = xyz
        si = sxyz0 = sxyz[uu]
        sj = sxyz
        
        dxyz = rj-ri
        d2xyz = np.sum((dxyz)**2, axis=1).reshape(dxyz.shape[0], 1)
        d1xyz = np.sqrt(d2xyz)
        d1xyzr = d1xyz.ravel()
        
        xh = dxyz / d1xyz
        xh[uu] = 0
        yh = si - xh * np.sum(si*xh, axis=1).reshape(dxyz.shape[0], 1)
        yh_dis = np.sum(yh**2, axis = 1)
        yh_ind = np.nonzero(np.abs(yh_dis)<1e-10)
        yh[yh_ind] = [0,0,0]
        
        aij = np.sum(si * yh, axis=1) * np.sum(sj * yh, axis=1) / yh_dis
        aij[yh_ind] = 0
        bij = 2 * np.sum(si * xh, axis=1) * np.sum(sj * xh, axis=1) - aij
        bij[uu] = 0
        
        w2 = bij / d1xyzr**3
        w2[uu] = 0
        
        s1 += np.histogram(d1xyzr, bins=rbin, weights=aij)[0]
        s2 += np.histogram(d1xyzr, bins=rbin, weights=w2)[0]
    
    # apply Gaussian shape function
    if psigma != None:
        x = np.arange(-3, 3, rstep)
        y = np.exp(-x**2 / psigma**2 / 2) * (1 / np.sqrt(2*np.pi) / psigma)
    
        s1[0] = 0
        s1 = fftconvolve(s1, y)
        s1 = s1[len(x)/2: -len(x)/2+1]
        
        s2 = fftconvolve(s2, y) * rstep
        s2 = s2[len(x)/2: -len(x)/2+1]
        
    ss2 = np.cumsum(s2)

    if rmin==0:
        r[0]=1e-4*rstep # avoid infinities at r=0
    fr = s1 / r + r * (ss2[-1] - ss2)
    r[0]=rmin
    fr /= len(calcList)

    fr *= np.exp(-1.0*(dampRate*r)**dampPower)
    # Do the convolution with the termination function if qmin/qmax have been given
    if qmin >= 0 and qmax > qmin:
        rth=np.arange(0.0,rmax+maxextension+rstep,rstep)
        rth[0]=1e-4*rstep # avoid infinities at r=0
        th=(np.sin(qmax*rth)-np.sin(qmin*rth))/np.pi/rth
        rth[0]=0.0
        rcv,frcv=cv(r,fr,rth,th)
    else:
        rcv,frcv=r,fr

    return rcv[np.logical_and(rcv>=r[0]-0.5*rstep,rcv<=rmax+0.5*rstep)], frcv[np.logical_and(rcv>=r[0]-0.5*rstep,rcv<=rmax+0.5*rstep)]
    

def calculateDr(r,fr,q,ff,paraScale=1.0,orderedScale=1.0/np.sqrt(2*np.pi),rmintr=-5.0,rmaxtr=5.0,drtr=0.01,qmin=0,qmax=-1):
    """Calculate the unnormalized mPDF quantity D(r).
    
    This module requires a normalized mPDF as an input, as well as a magnetic
    form factor and associated q grid.

    Args:
        r (numpy array): r grid for the properly normalized mPDF.
        fr (numpy array): the properly normalized mPDF.
        q (numpy array): grid of momentum transfer values used for calculating
            the magnetic form factor.
        ff (numpy array): magnetic form factor. Same shape as ffqgrid.
        paraScale (float): scale factor for the paramagnetic part of the
            unnormalized mPDF function D(r).
        ordScale (float): scale factor for the ordered part of the
            unnormalized mPDF function D(r).
        rmintr (float): minimum value of r for the Fourier transform of the
            magnetic form factor required for unnormalized mPDF.
        rmaxtr (float): maximum value of r for the Fourier transform of the
            magnetic form factor required for unnormalized mPDF.
        drtr (float): step size for r-grid used for calculating Fourier
            transform of magnetic form mactor.
        qmin (float): minimum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.
        qmax (float): maximum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.

    Returns: numpy array for the unnormalized mPDF Dr.
    """

    rsr,sr=costransform(q,ff,rmintr,rmaxtr,drtr)
    sr=np.sqrt(np.pi/2.0)*sr
    rSr,Sr=cv(rsr,sr,rsr,sr)
    para=-1.0*np.sqrt(2.0*np.pi)*np.gradient(Sr,rSr[1]-rSr[0]) ### paramagnetic term in d(r)
    rDr,Dr=cv(r,fr,rSr,Sr)
    Dr*=orderedScale
    if qmin >= 0 and qmax > qmin:
        rstep=r[1]-r[0]        
        rth=np.arange(0.0,r.max()+rstep,rstep)
        rth[0]=1e-4*rstep # avoid infinities at r=0
        th=(np.sin(qmax*rth)-np.sin(qmin*rth))/np.pi/rth
        rth[0]=0.0
        rpara,para=cv(rSr,para,rth,th)
    else:
        rpara,para=rSr,para

    Dr[:np.min((len(para),len(Dr)))]+=para[:np.min((len(para),len(Dr)))]*paraScale
    dr=r[1]-r[0]
    return Dr[np.logical_and(rDr>=np.min(r)-0.5*dr,rDr<=np.max(r)+0.5*dr)]
    
def generateAtomsXYZ(struc,rmax=30.0,magIdxs=[0]):
    """Generate array of atomic Cartesian coordinates from a given structure.

    Args:
        struc (diffpy.Structure object): provides lattice parameters and unit
            cell of the desired structure
        rmax (float): largest distance from central atom that should be
            included
        magIdxs (python list): list of integers giving indices of magnetic
            atoms in the unit cell

    Returns:
        numpy array of triples giving the Cartesian coordinates of all the
            magnetic atoms. Atom closest to the origin placed first in array.
    
    Note: This will only work well for structures that can be expressed with a
        unit cell that is close to orthorhombic or higher symmetry.
    """
    lat=struc.lattice
    unitcell=lat.stdbase
    cellwithatoms=struc.xyz_cartn[np.array(magIdxs)]
    radius=rmax+15.0
    dim1=np.round(radius/np.linalg.norm(unitcell[0]))
    dim2=np.round(radius/np.linalg.norm(unitcell[1]))
    dim3=np.round(radius/np.linalg.norm(unitcell[2]))

    ### generate the coordinates of each unit cell 
    latos=np.dot(np.mgrid[-dim1:dim1+1,-dim2:dim2+1,-dim3:dim3+1].transpose().ravel().reshape((2*dim1+1)*(2*dim2+1)*(2*dim3+1),3),unitcell)

    ### select points within a desired radius from origin
    latos=latos[np.where(np.apply_along_axis(np.linalg.norm,1,latos)<=(rmax+np.linalg.norm(unitcell.sum(axis=1))))]

    ## rearrange latos array so that [0,0,0] is the first one (for convenience)
    latos[np.where(np.all(latos==[0,0,0],axis=1))]=latos[0]
    latos[0]=np.array([0,0,0])

    ### create list of all atomic positions
    atoms=np.empty([len(latos)*len(cellwithatoms),3])
    index=0
    for i in range(len(latos)):
        for j in range(len(cellwithatoms)):
            atoms[index]=latos[i]+cellwithatoms[j]
            index+=1

    return atoms

def generateSpinsXYZ(struc,atoms=np.array([]),spinOrigin=np.array([0,0,0]),kvec=np.array([0,0,0]),svec=np.array([0,0,1])):
    """Generate array of 3-vectors representing the spins in a structure.

    Args:
        struc (diffpy.Structure object): provides lattice parameters and unit
            cell of the desired structure
        atoms (numpy array): list of atomic coordinates of all the magnetic
            atoms in the structure; e.g. generated by generateAtomsXYZ()
        spinOrigin (numpy array): three-vector giving the origin for the phase
            determined by the propagation vector
        kvec (numpy array): three-vector giving the propagation vector of the
            magnetic structure
        svec (numpy array): three-vector describing the spin located at the
            spin origin.
            
    Returns:
        numpy array of triples giving the spin vectors of all the magnetic
            atoms, in the same order as the atoms array provided as input.
    
    Note: At the moment, this only works for collinear magnetic structures
        with a single propagation vector.
    """
    lat=struc.lattice
    rlat=lat.reciprocal()
    astar,bstar,cstar=rlat.cartesian((1,0,0)),rlat.cartesian((0,1,0)),rlat.cartesian((0,0,1))
    kcart=kvec[0]*astar+kvec[1]*bstar+kvec[2]*cstar
    mags=np.cos(2*3.141593*np.dot(atoms-spinOrigin,kcart))
    spins=svec*mags.reshape(-1,1)
    return spins

class mPDFcalculator:
    """Create an mPDFcalculator object to help calculate mPDF functions.
    
    This class is loosely modelled after the PDFcalculator class in diffpy.
    At minimum, it requires input lists of atomic positions and spins, and
    it calculates the mPDF from that. The atoms and spins can also be 
    generated automatically if a diffpy structure object is supplied. Various
    other options can be specified for the calculated mPDF.
    
    Args:
        struc (diffpy.Structure object): provides lattice parameters and unit
            cell of the desired structure
        atoms (numpy array): list of atomic coordinates of all the magnetic
            atoms in the structure; e.g. generated by generateAtomsXYZ()
        rmaxAtoms (float): maximum distance from the origin of atomic
            positions generated by the makeAtoms method.
        spins (numpy array): triplets giving the spin vectors of all the 
            atoms, in the same order as the atoms array provided as input.
        svec (numpy array): three-vector describing the spin located at the
            spin origin.
        kvec (numpy array): three-vector giving the propagation vector of the
            magnetic structure
        spinOrigin (numpy array): three-vector giving the origin for the phase
            determined by the propagation vector
        ffqgrid (numpy array): grid of momentum transfer values used for 
            calculating the magnetic form factor.
        ff (numpy array): magnetic form factor. Same shape as ffqgrid.
        magIdxs (python list): list of integers giving indices of magnetic
            atoms in the unit cell
        calcList (python list): list giving the indices of the atoms array
            specifying the atoms to be used as the origin when calculating
            the mPDF.
        maxextension (float): extension of the r-grid on which the mPDF is
            calculated to properly account for contribution of pairs just
            outside the boundary.
        gaussPeakWidth(float): std deviation (in Angstroms) of Gaussian peak
            to be convoluted with the calculated mPDF to simulate thermal
            motion.
        dampRate (float): generalized ("stretched") exponential damping rate
                of the mPDF.
        dampPower (float): power of the generalized exponential function.
        qmin (float): minimum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.
        qmax (float): maximum experimentally accessible q-value (to be used
            for simulating termination ripples). If <0, no termination effects
            are included.
        rmin (float): minimum value of r for which mPDF should be calculated.
        rmax (float): maximum value of r for which mPDF should be calculated.
        rstep (float): step size for r-grid of calculated mPDF.
        ordScale (float): scale factor for the ordered part of the
            unnormalized mPDF function D(r).
        paraScale (float): scale factor for the paramagnetic part of the
            unnormalized mPDF function D(r).
        rmintr (float): minimum value of r for the Fourier transform of the
            magnetic form factor required for unnormalized mPDF.
        rmaxtr (float): maximum value of r for the Fourier transform of the
            magnetic form factor required for unnormalized mPDF.
        drtr (float): step size for r-grid used for calculating Fourier
            transform of magnetic form mactor.

        """
    def __init__(self,struc=[],atoms=np.array([]),rmaxAtoms=30.0,spins=np.array([]),svec=np.array([0,0,1]),kvec=np.array([0,0,0]),spinOrigin=np.array([0,0,0]),ffqgrid=np.array([]),ff=np.array([]),magIdxs=[0],calcList=np.array([0]),maxextension=10.0,gaussPeakWidth=0.1,dampRate=0.0,dampPower=2.0,qmin=-1.0,qmax=-1.0,rmin=0.0,rmax=20.0,rstep=0.01,ordScale=1.0/np.sqrt(2*np.pi),paraScale=1.0,rmintr=-5.0,rmaxtr=5.0,drtr=0.01):
        self.struc=struc        
        self.atoms=atoms
        self.rmaxAtoms=rmaxAtoms
        self.spins=spins
        self.svec=svec
        self.kvec=kvec
        self.spinOrigin=spinOrigin
        self.ffqgrid=ffqgrid
        self.ff=ff
        self.magIdxs=magIdxs
        self.calcList=calcList
        self.maxextension=maxextension
        self.gaussPeakWidth=gaussPeakWidth
        self.dampRate=dampRate
        self.dampPower=dampPower
        self.qmin=qmin
        self.qmax=qmax
        self.rmin=rmin
        self.rmax=rmax
        self.rstep=rstep
        self.ordScale=ordScale
        self.paraScale=paraScale
        self.rmintr=rmintr
        self.rmaxtr=rmaxtr
        self.drtr=drtr  

    def makeAtoms(self):
        """Generate the Cartesian coordinates of the atoms in the structure."""
        self.atoms=generateAtomsXYZ(self.struc,self.rmaxAtoms,self.magIdxs)

    def makeSpins(self):
        """Generate the Cartesian coordinates of the spin vectors in the
               structure. Must have a propagation vector, spin origin, and
               starting spin vector.
        """
        self.spins=generateSpinsXYZ(self.struc,self.atoms,self.spinOrigin,self.kvec,self.svec)

    def calc(self,normalized=True,both=False):
        """Calculate the magnetic PDF.

        Args:
            normalized (boolean): indicates whether or not the normalized mPDF
                should be returned.
            both (boolean): indicates whether or not both normalized and
                unnormalized mPDF quantities should be returned.

        Returns: numpy array giving the r grid of the calculation, as well as
            one or both the mPDF quantities.
        """
        rcalc,frcalc=calculatemPDF(self.atoms,self.spins,self.calcList,self.rstep,self.rmin,self.rmax,self.gaussPeakWidth,self.qmin,self.qmax,self.dampRate,self.dampPower,self.maxextension)
        if normalized and not both: 
            return rcalc,frcalc
        elif not normalized and not both:
            Drcalc=calculateDr(rcalc,frcalc,self.ffqgrid,self.ff,self.paraScale,self.ordScale,self.rmintr,self.rmaxtr,self.drtr,self.qmin,self.qmax)
            return rcalc,Drcalc
        else:
            Drcalc=calculateDr(rcalc,frcalc,self.ffqgrid,self.ff,self.paraScale,self.ordScale,self.rmintr,self.rmaxtr,self.drtr,self.qmin,self.qmax)            
            return rcalc,frcalc,Drcalc

    def rgrid(self):
        """Return the current r grid of the mPDF calculator."""
        return np.arange(self.rmin,self.rmax+self.rstep,self.rstep)

    def copy(self):
        """Return a deep copy of the mPDFcalculator object."""
        temp=[self]        
        return copy.deepcopy(temp)[0]



"""
A bunch of old stuff that may be worth pursuing at some point.
"""

# def test():
    # strufile = 'cif/ni_sc.cif'
    # from mstructure import MStruAdapter
    # stru = MStruAdapter(stru = strufile, name='mstru', periodic = True, rmax = 30)
    # stru.extend2Rmax(50)
    # xyz = stru.xyz_cartn
    # sxyz = stru.sxyz
    # uclist = stru.uclist
    # r, gr = calculateMPDF(xyz, sxyz, uclist, 0.01, 30, psigma=0.1)
    
    # plt.figure(1)
    # plt.plot(r,gr)
    # plt.show()
    # return

def calculateIQ(xyz, sxyz, uclist, qgrid, rstep, rmax, f):
    #qgrid = np.arange(qmin, qmax, qstep)
    S=np.linalg.norm(sxyz[0])
    r = np.arange(0, rmax, rstep)
    rbin =  np.concatenate([r-rstep/2, [r[-1]+rstep/2]])
    
    s1 = np.zeros(len(r))
    s2 = np.zeros(len(r))
    
    for i in range(len(uclist)):
        print 'Working on: '+str(i+1)+'/'+str(len(uclist))
        uu = uclist[i]
        
        ri = xyz0 = xyz[uu]
        rj = xyz
        si = sxyz0 = sxyz[uu]
        sj = sxyz
        
        dxyz = rj-ri
        d2xyz = np.sum((dxyz)**2, axis=1).reshape(dxyz.shape[0], 1)
        d1xyz = np.sqrt(d2xyz)
        d1xyzr = d1xyz.ravel()
        
        xh = dxyz / d1xyz
        xh[uu] = 0
        yh = si - xh * np.sum(si*xh, axis=1).reshape(dxyz.shape[0], 1)
        yh_dis = np.sum(yh**2, axis = 1)
        yh_ind = np.nonzero(np.abs(yh_dis)<1e-10)
        yh[yh_ind] = [0,0,0]
        
        aij = np.sum(si * yh, axis=1) * np.sum(sj * yh, axis=1) / yh_dis
        aij[yh_ind] = 0
        bij = 2 * np.sum(si * xh, axis=1) * np.sum(sj * xh, axis=1) - aij
        bij[uu] = 0
        
        s1 += np.histogram(d1xyzr, bins=rbin, weights=aij)[0]
        s2 += np.histogram(d1xyzr, bins=rbin, weights=bij)[0]
        
    rv = np.zeros_like(qgrid)
    #index non-zero s1 and s2
    inds1 = np.nonzero(s1)[0]
    inds2 = np.nonzero(s2)[0]
    for i in inds1:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s1[i] * np.sin(qxr)/(qxr)
    for i in inds2:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s2[i] * (np.sin(qxr)/qxr**3-np.cos(qxr)/qxr**2)
    rv=rv*(f**2)
    rv+=len(uclist)*2.*S*(S+1)*(f**2)/3.
    return [qgrid, rv]

def calculateIQPBC(xyz, sxyz, uclist, qgrid, rstep, f, latparams):
    S=np.linalg.norm(sxyz[0])
    x,y,z = np.transpose(xyz)
    nx,ny,nz=np.ceil((np.max(x)-np.min(x))/latparams[0]),np.ceil((np.max(y)-np.min(y))/latparams[1]),np.ceil((np.max(z)-np.min(z))/latparams[2]) ### assuming input list is cubic
    X,Y,Z=nx*latparams[0],ny*latparams[1],nz*latparams[2]
    boxsize=np.array([X,Y,Z])
    rmax=0.5*np.min(boxsize)
    print 'rmax='+str(rmax)
    r = np.arange(0, rmax, rstep)
    rbin =  np.concatenate([r-rstep/2, [r[-1]+rstep/2]])
    
    s1 = np.zeros(len(r))
    s2 = np.zeros(len(r))
    
    for i in range(len(uclist)):
        print 'Working on: '+str(i+1)+'/'+str(len(uclist))
        uu = uclist[i]
        
        ri = xyz0 = xyz[uu]
        rj = xyz
        si = sxyz0 = sxyz[uu]
        sj = sxyz
        
        dxyz = rj-ri
        dxyz = np.where(dxyz>0.5*boxsize,dxyz-boxsize,dxyz)
        dxyz = np.where(dxyz<-0.5*boxsize,dxyz+boxsize,dxyz)
        d2xyz = np.sum((dxyz)**2, axis=1).reshape(dxyz.shape[0], 1)
        d1xyz = np.sqrt(d2xyz)
        d1xyzr = d1xyz.ravel() ## need to maybe check that all distances are less than rmax?
        
        xh = dxyz / d1xyz
        xh[np.isnan(xh)] = 0
        yh = si - xh * np.sum(si*xh, axis=1).reshape(dxyz.shape[0], 1)
        yh_dis = np.sum(yh**2, axis = 1)
        yh_ind = np.nonzero(np.abs(yh_dis)<1e-10)
        yh[yh_ind] = np.array([0,0,0])
        
        aij = np.sum(si * yh, axis=1) * np.sum(sj * yh, axis=1) / yh_dis ## check and see why I am dividing by yh_dis
        aij[yh_ind] = 0
        bij = 2 * np.sum(si * xh, axis=1) * np.sum(sj * xh, axis=1) - aij
        bij[uu] = 0
        
        r_ind=np.nonzero(d1xyzr>rmax)
        aij[r_ind] = 0
        bij[r_ind] = 0
        
        s1 += np.histogram(d1xyzr, bins=rbin, weights=aij)[0]
        s2 += np.histogram(d1xyzr, bins=rbin, weights=bij)[0]
        
    rv = np.zeros_like(qgrid)
    #index non-zero s1 and s2
    inds1 = np.nonzero(s1)[0]
    inds2 = np.nonzero(s2)[0]
    for i in inds1:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s1[i] * np.sin(qxr)/(qxr)
    for i in inds2:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s2[i] * (np.sin(qxr)/qxr**3-np.cos(qxr)/qxr**2)
    rv=rv*(f**2)
    rv+=len(uclist)*2.*S*(S+1)*(f**2)/3.
    return [qgrid, rv]

    
def calculateIQPBCold(xyz, sxyz, uclist, qgrid, rstep, rmax, f, a):  ### a is lattice parameter of unit cell
    from cubicPBC import cubicPBC
    S=np.linalg.norm(sxyz[0])
    r = np.arange(0, rmax, rstep)
    rbin =  np.concatenate([r-rstep/2, [r[-1]+rstep/2]])
    
    s1 = np.zeros(len(r))
    s2 = np.zeros(len(r))
    
    xyz,sxyz=cubicPBC(xyz,sxyz,a)
    
    for i in range(len(uclist)):
        print 'Working on: '+str(i+1)+'/'+str(len(uclist))
        uu = uclist[i]
        
        ri = xyz0 = xyz[uu]
        rj = xyz
        si = sxyz0 = sxyz[uu]
        sj = sxyz
        
        dxyz = rj-ri
        d2xyz = np.sum((dxyz)**2, axis=1).reshape(dxyz.shape[0], 1)
        d1xyz = np.sqrt(d2xyz)
        d1xyzr = d1xyz.ravel()
        
        xh = dxyz / d1xyz
        xh[uu] = 0
        yh = si - xh * np.sum(si*xh, axis=1).reshape(dxyz.shape[0], 1)
        yh_dis = np.sum(yh**2, axis = 1)
        yh_ind = np.nonzero(np.abs(yh_dis)<1e-10)
        yh[yh_ind] = [0,0,0]
        
        aij = np.sum(si * yh, axis=1) * np.sum(sj * yh, axis=1) / yh_dis
        aij[yh_ind] = 0
        bij = 2 * np.sum(si * xh, axis=1) * np.sum(sj * xh, axis=1) - aij
        bij[uu] = 0
        
        s1 += np.histogram(d1xyzr, bins=rbin, weights=aij)[0]
        s2 += np.histogram(d1xyzr, bins=rbin, weights=bij)[0]
        
    rv = np.zeros_like(qgrid)
    #index non-zero s1 and s2
    inds1 = np.nonzero(s1)[0]
    inds2 = np.nonzero(s2)[0]
    for i in inds1:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s1[i] * np.sin(qxr)/(qxr)
    for i in inds2:
        qxr = qgrid*r[i]
        if r[i]>0:
            rv += s2[i] * (np.sin(qxr)/qxr**3-np.cos(qxr)/qxr**2)
    rv=rv*(f**2)
    rv+=len(uclist)*2.*S*(S+1)*(f**2)/3.
    return [qgrid, rv]


def test():
    print 'This is not a rigorous test.'
    return
    
if __name__=='__main__':
    test()
