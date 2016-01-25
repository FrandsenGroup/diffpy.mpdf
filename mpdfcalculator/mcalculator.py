#!/usr/bin/env python
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, convolve
import diffpy as dp
from diffpy.srreal.bondcalculator import BondCalculator
import copy

def jCalc(q,params=[0.2394,26.038,0.4727,12.1375,0.3065,3.0939,-0.01906],j2=False):
    """Calculate the magnetic form factor j0.

    This method for calculating the magnetic form factor is based on the
    approximate empirical forms based on the tables by Brown, consisting of
    the sum of 3 Gaussians and a constant. 
    
    Args:
        q (numpy array): 1-d grid of momentum transfer on which the form
            factor is to be computed
        params (python list): provides the 7 numerical coefficients. The
            default is an average form factor of 3d j0 approximations.  
        j2 (boolean): if True, calculate the j2 approximation for orbital
            angular momentum contributions   
    
    Returns:
        numpy array with same shape as q giving the magnetic form factor j0 or j2.
    """
    [A,a,B,b,C,c,D] = params
    if j2:
        return (A*np.exp(-a*(q/4/np.pi)**2)+B*np.exp(-b*(q/4/np.pi)**2)+C*np.exp(-c*(q/4/np.pi)**2)+D)*(q/4.0/np.pi)**2
    else:
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
	
def calculatemPDF(xyz, sxyz, gfactors=np.array([2.0]), calcList=np.array([0]), rstep=0.01, rmin=0.0, rmax=20.0, psigma=0.1,qmin=0,qmax=-1,dampRate=0.0,dampPower=2.0,maxextension=10.0):
    """Calculate the normalized mPDF.
    
    At minimum, this module requires input lists of atomic positions and spins.
    
    Args:
        xyz (numpy array): list of atomic coordinates of all the magnetic
            atoms in the structure.
        sxyz (numpy array): triplets giving the spin vectors of all the 
            atoms, in the same order as the xyz array provided as input.
        gfactors (numpy array): Lande g-factors of spins in same order as
            spin array.
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
    # check if g-factors have been provided
    if sxyz.shape[0]!=gfactors.shape[0]:
        gfactors=2.0*np.ones(sxyz.shape[0])

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
        gi = gfactors[uu]
        gj = gfactors
        
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
        
        s1 += np.histogram(d1xyzr, bins=rbin, weights=gi*gj*aij)[0]
        s2 += np.histogram(d1xyzr, bins=rbin, weights=gi*gj*w2)[0]
    
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
    fr /= len(calcList)*np.mean(gfactors)**2

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
    
def generateAtomsXYZ(struc,rmax=30.0,magIdxs=[[0]],square=False):
    """Generate array of atomic Cartesian coordinates from a given structure.

    Args:
        struc (diffpy.Structure object): provides lattice parameters and unit
            cell of the desired structure
        rmax (float): largest distance from central atom that should be
            included
        magIdxs (python list): list of integers giving indices of magnetic
            atoms in the unit cell
        square (boolean): if not True, atoms within a given radius from the
            origin will be returned; if True, then the full grid will be
            returned rather than just a spherical selection.

    Returns:
        numpy array of triples giving the Cartesian coordinates of all the
            magnetic atoms. Atom closest to the origin placed first in array.
    
    Note: If square=True, this may have problems for structures that have
        a several distorted unit cell (i.e. highly non-orthorhombic).
    """
    if not square:    
        magAtoms=struc[magIdxs]
        bc=BondCalculator(rmax=rmax+np.linalg.norm(struc.lattice.stdbase.sum(axis=1)))
        bc.setPairMask(0,'all',True,others=False)
        bc(magAtoms)
        atoms=np.vstack([struc.xyz_cartn[magIdxs[0]],bc.directions[bc.sites0==0]])        

    else:
        # generate the coordinates of each unit cell
        lat=struc.lattice
        unitcell=lat.stdbase
        cellwithatoms=struc.xyz_cartn[np.array(magIdxs)]
        radius=rmax+15.0

        dim1=np.round(rmax/np.linalg.norm(unitcell[0]))
        dim2=np.round(rmax/np.linalg.norm(unitcell[1]))
        dim3=np.round(rmax/np.linalg.norm(unitcell[2]))
        latos=np.dot(np.mgrid[-dim1:dim1+1,-dim2:dim2+1,-dim3:dim3+1].transpose().ravel().reshape((2*dim1+1)*(2*dim2+1)*(2*dim3+1),3),unitcell)

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

def generateSpinsXYZ(struc,atoms=np.array([[]]),kvecs=np.array([[0,0,0]]),basisvecs=np.array([[0,0,1]])):
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
    i=1j
 
    spins=0*atoms
    for idx in range(len(kvecs)):
        kvec=kvecs[idx]
        kcart=kvec[0]*astar+kvec[1]*bstar+kvec[2]*cstar
        phasefac=np.exp(-2.0*3.14159265359*i*np.dot(atoms,kcart))
        cspins=basisvecs[idx]*phasefac[:,np.newaxis]
        spins+=np.real(cspins)

    if np.abs(np.imag(cspins)).max()>0.0001:
        print np.abs(np.imag(cspins)).max()
        print 'Warning: basis vectors resulted in complex spins.'
        print 'Imaginary parts have been discarded.'    

    return spins

def getFFparams(name,j2=False):
    """Get list of parameters for approximation of magnetic form factor

    Args:
        name (str): Name of magnetic ion in form 'Mn2' for Mn2+, etc.
        j2 (boolean): True of the j2 approximation should be calculated;
            otherwise, the j0 approximation is calculated.

    Returns:
        Python list of the 7 coefficients in the analytical approximation
            given at e.g. https://www.ill.eu/sites/ccsl/ffacts/ffachtml.html
    """
    if not j2:
        j0dict={'Am2': [0.4743, 21.7761, 1.58, 5.6902, -1.0779, 4.1451, 0.0218],
 'Am3': [0.4239, 19.5739, 1.4573, 5.8722, -0.9052, 3.9682, 0.0238],
 'Am4': [0.3737, 17.8625, 1.3521, 6.0426, -0.7514, 3.7199, 0.0258],
 'Am5': [0.2956, 17.3725, 1.4525, 6.0734, -0.7755, 3.6619, 0.0277],
 'Am6': [0.2302, 16.9533, 1.4864, 6.1159, -0.7457, 3.5426, 0.0294],
 'Am7': [0.3601, 12.7299, 1.964, 5.1203, -1.356, 3.7142, 0.0316],
 'Ce2': [0.2953, 17.6846, 0.2923, 6.7329, 0.4313, 5.3827, -0.0194],
 'Co0': [0.4139, 16.1616, 0.6013, 4.7805, -0.1518, 0.021, 0.1345],
 'Co1': [0.099, 33.1252, 0.3645, 15.1768, 0.547, 5.0081, -0.0109],
 'Co2': [0.4332, 14.3553, 0.5857, 4.6077, -0.0382, 0.1338, 0.0179],
 'Co3': [0.3902, 12.5078, 0.6324, 4.4574, -0.15, 0.0343, 0.1272],
 'Co4': [0.3515, 10.7785, 0.6778, 4.2343, -0.0389, 0.2409, 0.0098],
 'Cr0': [0.1135, 45.199, 0.3481, 19.4931, 0.5477, 7.3542, -0.0092],
 'Cr1': [-0.0977, 0.047, 0.4544, 26.0054, 0.5579, 7.4892, 0.0831],
 'Cr2': [1.2024, -0.0055, 0.4158, 20.5475, 0.6032, 6.956, -1.2218],
 'Cr3': [-0.3094, 0.0274, 0.368, 17.0355, 0.6559, 6.5236, 0.2856],
 'Cr4': [-0.232, 0.0433, 0.3101, 14.9518, 0.7182, 6.1726, 0.2042],
 'Cu0': [0.0909, 34.9838, 0.4088, 11.4432, 0.5128, 3.8248, -0.0124],
 'Cu1': [0.0749, 34.9656, 0.4147, 11.7642, 0.5238, 3.8497, -0.0127],
 'Cu2': [0.0232, 34.9686, 0.4023, 11.564, 0.5882, 3.8428, -0.0137],
 'Cu3': [0.0031, 34.9074, 0.3582, 10.9138, 0.6531, 3.8279, -0.0147],
 'Cu4': [-0.0132, 30.6817, 0.2801, 11.1626, 0.749, 3.8172, -0.0165],
 'Dy2': [0.1308, 18.3155, 0.3118, 7.6645, 0.5795, 3.1469, -0.0226],
 'Dy3': [0.1157, 15.0732, 0.327, 6.7991, 0.5821, 3.0202, -0.0249],
 'Er2': [0.1122, 18.1223, 0.3462, 6.9106, 0.5649, 2.7614, -0.0235],
 'Er3': [0.0586, 17.9802, 0.354, 7.0964, 0.6126, 2.7482, -0.0251],
 'Eu2': [0.0755, 25.296, 0.3001, 11.5993, 0.6438, 4.0252, -0.0196],
 'Eu3': [0.0204, 25.3078, 0.301, 11.4744, 0.7005, 3.942, -0.022],
 'Fe0': [0.0706, 35.0085, 0.3589, 15.3583, 0.5819, 5.5606, -0.0114],
 'Fe1': [0.1251, 34.9633, 0.3629, 15.5144, 0.5223, 5.5914, -0.0105],
 'Fe2': [0.0263, 34.9597, 0.3668, 15.9435, 0.6188, 5.5935, -0.0119],
 'Fe3': [0.3972, 13.2442, 0.6295, 4.9034, -0.0314, 0.3496, 0.0044],
 'Fe4': [0.3782, 11.38, 0.6556, 4.592, -0.0346, 0.4833, 0.0005],
 'Gd2': [0.0636, 25.3823, 0.3033, 11.2125, 0.6528, 3.7877, -0.0199],
 'Gd3': [0.0186, 25.3867, 0.2895, 11.1421, 0.7135, 3.752, -0.0217],
 'Ho2': [0.0995, 18.1761, 0.3305, 7.8556, 0.5921, 2.9799, -0.023],
 'Ho3': [0.0566, 18.3176, 0.3365, 7.688, 0.6317, 2.9427, -0.0248],
 'Mn0': [0.2438, 24.9629, 0.1472, 15.6728, 0.6189, 6.5403, -0.0105],
 'Mn1': [-0.0138, 0.4213, 0.4231, 24.668, 0.5905, 6.6545, -0.001],
 'Mn2': [0.422, 17.684, 0.5948, 6.005, 0.0043, -0.609, -0.0219],
 'Mn3': [0.4198, 14.2829, 0.6054, 5.4689, 0.9241, -0.0088, -0.9498],
 'Mn4': [0.376, 12.5661, 0.6602, 5.1329, -0.0372, 0.563, 0.0011],
 'Mo0': [0.1806, 49.0568, 1.2306, 14.7859, -0.4268, 6.9866, 0.0171],
 'Mo1': [0.35, 48.0354, 1.0305, 15.0604, -0.3929, 7.479, 0.0139],
 'Nb0': [0.3946, 49.2297, 1.3197, 14.8216, -0.7269, 9.6156, 0.0129],
 'Nb1': [0.4572, 49.9182, 1.0274, 15.7256, -0.4962, 9.1573, 0.0118],
 'Nd2': [0.1645, 25.0453, 0.2522, 11.9782, 0.6012, 4.9461, -0.018],
 'Nd3': [0.054, 25.0293, 0.3101, 12.102, 0.6575, 4.7223, -0.0216],
 'Ni0': [-0.0172, 35.7392, 0.3174, 14.2689, 0.7136, 4.5661, -0.0143],
 'Ni1': [0.0705, 35.8561, 0.3984, 13.8042, 0.5427, 4.3965, -0.0118],
 'Ni2': [0.0163, 35.8826, 0.3916, 13.2233, 0.6052, 4.3388, -0.0133],
 'Ni3': [-0.0134, 35.8677, 0.2678, 12.3326, 0.7614, 4.2369, -0.0162],
 'Ni4': [-0.009, 35.8614, 0.2776, 11.7904, 0.7474, 4.2011, -0.0163],
 'Np3': [0.5157, 20.8654, 2.2784, 5.893, -1.8163, 4.8457, 0.0211],
 'Np4': [0.4206, 19.8046, 2.8004, 5.9783, -2.2436, 4.9848, 0.0228],
 'Np5': [0.3692, 18.19, 3.151, 5.85, -2.5446, 4.9164, 0.0248],
 'Np6': [0.2929, 17.5611, 3.4866, 5.7847, -2.8066, 4.8707, 0.0267],
 'Pd0': [0.2003, 29.3633, 1.1446, 9.5993, -0.3689, 4.0423, 0.0251],
 'Pd1': [0.5033, 24.5037, 1.9982, 6.9082, -1.524, 5.5133, 0.0213],
 'Pr3': [0.0504, 24.9989, 0.2572, 12.0377, 0.7142, 5.0039, -0.0219],
 'Pu3': [0.384, 16.6793, 3.1049, 5.421, -2.5148, 4.5512, 0.0263],
 'Pu4': [0.4934, 16.8355, 1.6394, 5.6384, -1.1581, 4.1399, 0.0248],
 'Pu5': [0.3888, 16.5592, 2.0362, 5.6567, -1.4515, 4.2552, 0.0267],
 'Pu6': [0.3172, 16.0507, 3.4654, 5.3507, -2.8102, 4.5133, 0.0281],
 'Rh0': [0.0976, 49.8825, 1.1601, 11.8307, -0.2789, 4.1266, 0.0234],
 'Rh1': [0.3342, 29.7564, 1.2209, 9.4384, -0.5755, 5.332, 0.021],
 'Ru0': [0.1069, 49.4238, 1.1912, 12.7417, -0.3176, 4.9125, 0.0213],
 'Ru1': [0.441, 33.3086, 1.4775, 9.5531, -0.9361, 6.722, 0.0176],
 'Sc0': [0.2512, 90.0296, 0.329, 39.4021, 0.4235, 14.3222, -0.0043],
 'Sc1': [0.4889, 51.1603, 0.5203, 14.0764, -0.0286, 0.1792, 0.0185],
 'Sc2': [0.5048, 31.4035, 0.5186, 10.9897, -0.0241, 1.1831, 0.0],
 'Sm2': [0.0909, 25.2032, 0.3037, 11.8562, 0.625, 4.2366, -0.02],
 'Sm3': [0.0288, 25.2068, 0.2973, 11.8311, 0.6954, 4.2117, -0.0213],
 'Tb2': [0.0547, 25.5086, 0.3171, 10.5911, 0.649, 3.5171, -0.0212],
 'Tb3': [0.0177, 25.5095, 0.2921, 10.5769, 0.7133, 3.5122, -0.0231],
 'Tc0': [0.1298, 49.6611, 1.1656, 14.1307, -0.3134, 5.5129, 0.0195],
 'Tc1': [0.2674, 48.9566, 0.9569, 15.1413, -0.2387, 5.4578, 0.016],
 'Ti0': [0.4657, 33.5898, 0.549, 9.8791, -0.0291, 0.3232, 0.0123],
 'Ti1': [0.5093, 36.7033, 0.5032, 10.3713, -0.0263, 0.3106, 0.0116],
 'Ti2': [0.5091, 24.9763, 0.5162, 8.7569, -0.0281, 0.916, 0.0015],
 'Ti3': [0.3571, 22.8413, 0.6688, 8.9306, -0.0354, 0.4833, 0.0099],
 'Tm2': [0.0983, 18.3236, 0.338, 6.9178, 0.5875, 2.6622, -0.0241],
 'Tm3': [0.0581, 15.0922, 0.2787, 7.8015, 0.6854, 2.7931, -0.0224],
 'U3': [0.5058, 23.2882, 1.3464, 7.0028, -0.8724, 4.8683, 0.0192],
 'U4': [0.3291, 23.5475, 1.0836, 8.454, -0.434, 4.1196, 0.0214],
 'U5': [0.365, 19.8038, 3.2199, 6.2818, -2.6077, 5.301, 0.0233],
 'V0': [0.4086, 28.8109, 0.6077, 8.5437, -0.0295, 0.2768, 0.0123],
 'V1': [0.4444, 32.6479, 0.5683, 9.0971, -0.2285, 0.0218, 0.215],
 'V2': [0.4085, 23.8526, 0.6091, 8.2456, -0.1676, 0.0415, 0.1496],
 'V3': [0.3598, 19.3364, 0.6632, 7.6172, -0.3064, 0.0296, 0.2835],
 'V4': [0.3106, 16.816, 0.7198, 7.0487, -0.0521, 0.302, 0.0221],
 'Y0': [0.5915, 67.6081, 1.5123, 17.9004, -1.113, 14.1359, 0.008],
 'Yb2': [0.0855, 18.5123, 0.2943, 7.3734, 0.6412, 2.6777, -0.0213],
 'Yb3': [0.0416, 16.0949, 0.2849, 7.8341, 0.6961, 2.6725, -0.0229],
 'Zr0': [0.4106, 59.9961, 1.0543, 18.6476, -0.4751, 10.54, 0.0106],
 'Zr1': [0.4532, 59.5948, 0.7834, 21.4357, -0.2451, 9.036, 0.0098]}
        try:
            return j0dict[name]
        except KeyError:
            print 'No magnetic form factor found for that element/ion.'
            return ['none']
    else:
        j2dict={'Am2': [0.4743, 21.7761, 1.58, 5.6902, -1.0779, 4.1451, 0.0218],
 'Am3': [0.4239, 19.5739, 1.4573, 5.8722, -0.9052, 3.9682, 0.0238],
 'Am4': [0.3737, 17.8625, 1.3521, 6.0426, -0.7514, 3.7199, 0.0258],
 'Am5': [0.2956, 17.3725, 1.4525, 6.0734, -0.7755, 3.6619, 0.0277],
 'Am6': [0.2302, 16.9533, 1.4864, 6.1159, -0.7457, 3.5426, 0.0294],
 'Am7': [0.3601, 12.7299, 1.964, 5.1203, -1.356, 3.7142, 0.0316],
 'Ce2': [0.2953, 17.6846, 0.2923, 6.7329, 0.4313, 5.3827, -0.0194],
 'Co0': [0.4139, 16.1616, 0.6013, 4.7805, -0.1518, 0.021, 0.1345],
 'Co1': [0.099, 33.1252, 0.3645, 15.1768, 0.547, 5.0081, -0.0109],
 'Co2': [0.4332, 14.3553, 0.5857, 4.6077, -0.0382, 0.1338, 0.0179],
 'Co3': [0.3902, 12.5078, 0.6324, 4.4574, -0.15, 0.0343, 0.1272],
 'Co4': [0.3515, 10.7785, 0.6778, 4.2343, -0.0389, 0.2409, 0.0098],
 'Cr0': [0.1135, 45.199, 0.3481, 19.4931, 0.5477, 7.3542, -0.0092],
 'Cr1': [-0.0977, 0.047, 0.4544, 26.0054, 0.5579, 7.4892, 0.0831],
 'Cr2': [1.2024, -0.0055, 0.4158, 20.5475, 0.6032, 6.956, -1.2218],
 'Cr3': [-0.3094, 0.0274, 0.368, 17.0355, 0.6559, 6.5236, 0.2856],
 'Cr4': [-0.232, 0.0433, 0.3101, 14.9518, 0.7182, 6.1726, 0.2042],
 'Cu0': [0.0909, 34.9838, 0.4088, 11.4432, 0.5128, 3.8248, -0.0124],
 'Cu1': [0.0749, 34.9656, 0.4147, 11.7642, 0.5238, 3.8497, -0.0127],
 'Cu2': [0.0232, 34.9686, 0.4023, 11.564, 0.5882, 3.8428, -0.0137],
 'Cu3': [0.0031, 34.9074, 0.3582, 10.9138, 0.6531, 3.8279, -0.0147],
 'Cu4': [-0.0132, 30.6817, 0.2801, 11.1626, 0.749, 3.8172, -0.0165],
 'Dy2': [0.1308, 18.3155, 0.3118, 7.6645, 0.5795, 3.1469, -0.0226],
 'Dy3': [0.1157, 15.0732, 0.327, 6.7991, 0.5821, 3.0202, -0.0249],
 'Er2': [0.1122, 18.1223, 0.3462, 6.9106, 0.5649, 2.7614, -0.0235],
 'Er3': [0.0586, 17.9802, 0.354, 7.0964, 0.6126, 2.7482, -0.0251],
 'Eu2': [0.0755, 25.296, 0.3001, 11.5993, 0.6438, 4.0252, -0.0196],
 'Eu3': [0.0204, 25.3078, 0.301, 11.4744, 0.7005, 3.942, -0.022],
 'Fe0': [0.0706, 35.0085, 0.3589, 15.3583, 0.5819, 5.5606, -0.0114],
 'Fe1': [0.1251, 34.9633, 0.3629, 15.5144, 0.5223, 5.5914, -0.0105],
 'Fe2': [0.0263, 34.9597, 0.3668, 15.9435, 0.6188, 5.5935, -0.0119],
 'Fe3': [0.3972, 13.2442, 0.6295, 4.9034, -0.0314, 0.3496, 0.0044],
 'Fe4': [0.3782, 11.38, 0.6556, 4.592, -0.0346, 0.4833, 0.0005],
 'Gd2': [0.0636, 25.3823, 0.3033, 11.2125, 0.6528, 3.7877, -0.0199],
 'Gd3': [0.0186, 25.3867, 0.2895, 11.1421, 0.7135, 3.752, -0.0217],
 'Ho2': [0.0995, 18.1761, 0.3305, 7.8556, 0.5921, 2.9799, -0.023],
 'Ho3': [0.0566, 18.3176, 0.3365, 7.688, 0.6317, 2.9427, -0.0248],
 'Mn0': [0.2438, 24.9629, 0.1472, 15.6728, 0.6189, 6.5403, -0.0105],
 'Mn1': [-0.0138, 0.4213, 0.4231, 24.668, 0.5905, 6.6545, -0.001],
 'Mn2': [0.422, 17.684, 0.5948, 6.005, 0.0043, -0.609, -0.0219],
 'Mn3': [0.4198, 14.2829, 0.6054, 5.4689, 0.9241, -0.0088, -0.9498],
 'Mn4': [0.376, 12.5661, 0.6602, 5.1329, -0.0372, 0.563, 0.0011],
 'Mo0': [0.1806, 49.0568, 1.2306, 14.7859, -0.4268, 6.9866, 0.0171],
 'Mo1': [0.35, 48.0354, 1.0305, 15.0604, -0.3929, 7.479, 0.0139],
 'Nb0': [0.3946, 49.2297, 1.3197, 14.8216, -0.7269, 9.6156, 0.0129],
 'Nb1': [0.4572, 49.9182, 1.0274, 15.7256, -0.4962, 9.1573, 0.0118],
 'Nd2': [0.1645, 25.0453, 0.2522, 11.9782, 0.6012, 4.9461, -0.018],
 'Nd3': [0.054, 25.0293, 0.3101, 12.102, 0.6575, 4.7223, -0.0216],
 'Ni0': [-0.0172, 35.7392, 0.3174, 14.2689, 0.7136, 4.5661, -0.0143],
 'Ni1': [0.0705, 35.8561, 0.3984, 13.8042, 0.5427, 4.3965, -0.0118],
 'Ni2': [0.0163, 35.8826, 0.3916, 13.2233, 0.6052, 4.3388, -0.0133],
 'Ni3': [-0.0134, 35.8677, 0.2678, 12.3326, 0.7614, 4.2369, -0.0162],
 'Ni4': [-0.009, 35.8614, 0.2776, 11.7904, 0.7474, 4.2011, -0.0163],
 'Np3': [0.5157, 20.8654, 2.2784, 5.893, -1.8163, 4.8457, 0.0211],
 'Np4': [0.4206, 19.8046, 2.8004, 5.9783, -2.2436, 4.9848, 0.0228],
 'Np5': [0.3692, 18.19, 3.151, 5.85, -2.5446, 4.9164, 0.0248],
 'Np6': [0.2929, 17.5611, 3.4866, 5.7847, -2.8066, 4.8707, 0.0267],
 'Pd0': [0.2003, 29.3633, 1.1446, 9.5993, -0.3689, 4.0423, 0.0251],
 'Pd1': [0.5033, 24.5037, 1.9982, 6.9082, -1.524, 5.5133, 0.0213],
 'Pr3': [0.0504, 24.9989, 0.2572, 12.0377, 0.7142, 5.0039, -0.0219],
 'Pu3': [0.384, 16.6793, 3.1049, 5.421, -2.5148, 4.5512, 0.0263],
 'Pu4': [0.4934, 16.8355, 1.6394, 5.6384, -1.1581, 4.1399, 0.0248],
 'Pu5': [0.3888, 16.5592, 2.0362, 5.6567, -1.4515, 4.2552, 0.0267],
 'Pu6': [0.3172, 16.0507, 3.4654, 5.3507, -2.8102, 4.5133, 0.0281],
 'Rh0': [0.0976, 49.8825, 1.1601, 11.8307, -0.2789, 4.1266, 0.0234],
 'Rh1': [0.3342, 29.7564, 1.2209, 9.4384, -0.5755, 5.332, 0.021],
 'Ru0': [0.1069, 49.4238, 1.1912, 12.7417, -0.3176, 4.9125, 0.0213],
 'Ru1': [0.441, 33.3086, 1.4775, 9.5531, -0.9361, 6.722, 0.0176],
 'Sc0': [0.2512, 90.0296, 0.329, 39.4021, 0.4235, 14.3222, -0.0043],
 'Sc1': [0.4889, 51.1603, 0.5203, 14.0764, -0.0286, 0.1792, 0.0185],
 'Sc2': [0.5048, 31.4035, 0.5186, 10.9897, -0.0241, 1.1831, 0.0],
 'Sm2': [0.0909, 25.2032, 0.3037, 11.8562, 0.625, 4.2366, -0.02],
 'Sm3': [0.0288, 25.2068, 0.2973, 11.8311, 0.6954, 4.2117, -0.0213],
 'Tb2': [0.0547, 25.5086, 0.3171, 10.5911, 0.649, 3.5171, -0.0212],
 'Tb3': [0.0177, 25.5095, 0.2921, 10.5769, 0.7133, 3.5122, -0.0231],
 'Tc0': [0.1298, 49.6611, 1.1656, 14.1307, -0.3134, 5.5129, 0.0195],
 'Tc1': [0.2674, 48.9566, 0.9569, 15.1413, -0.2387, 5.4578, 0.016],
 'Ti0': [0.4657, 33.5898, 0.549, 9.8791, -0.0291, 0.3232, 0.0123],
 'Ti1': [0.5093, 36.7033, 0.5032, 10.3713, -0.0263, 0.3106, 0.0116],
 'Ti2': [0.5091, 24.9763, 0.5162, 8.7569, -0.0281, 0.916, 0.0015],
 'Ti3': [0.3571, 22.8413, 0.6688, 8.9306, -0.0354, 0.4833, 0.0099],
 'Tm2': [0.0983, 18.3236, 0.338, 6.9178, 0.5875, 2.6622, -0.0241],
 'Tm3': [0.0581, 15.0922, 0.2787, 7.8015, 0.6854, 2.7931, -0.0224],
 'U3': [0.5058, 23.2882, 1.3464, 7.0028, -0.8724, 4.8683, 0.0192],
 'U4': [0.3291, 23.5475, 1.0836, 8.454, -0.434, 4.1196, 0.0214],
 'U5': [0.365, 19.8038, 3.2199, 6.2818, -2.6077, 5.301, 0.0233],
 'V0': [0.4086, 28.8109, 0.6077, 8.5437, -0.0295, 0.2768, 0.0123],
 'V1': [0.4444, 32.6479, 0.5683, 9.0971, -0.2285, 0.0218, 0.215],
 'V2': [0.4085, 23.8526, 0.6091, 8.2456, -0.1676, 0.0415, 0.1496],
 'V3': [0.3598, 19.3364, 0.6632, 7.6172, -0.3064, 0.0296, 0.2835],
 'V4': [0.3106, 16.816, 0.7198, 7.0487, -0.0521, 0.302, 0.0221],
 'Y0': [0.5915, 67.6081, 1.5123, 17.9004, -1.113, 14.1359, 0.008],
 'Yb2': [0.0855, 18.5123, 0.2943, 7.3734, 0.6412, 2.6777, -0.0213],
 'Yb3': [0.0416, 16.0949, 0.2849, 7.8341, 0.6961, 2.6725, -0.0229],
 'Zr0': [0.4106, 59.9961, 1.0543, 18.6476, -0.4751, 10.54, 0.0106],
 'Zr1': [0.4532, 59.5948, 0.7834, 21.4357, -0.2451, 9.036, 0.0098]}
        try:
            return j2dict[name]
        except KeyError:
            print 'No magnetic form factor found for that element/ion.'
            return ['none']

class magSpecies:
    """Store information for a single species of magnetic atom.
    """
    def __init__(self,struc=None,label='',magIdxs=[0],atoms=None,spins=None,rmaxAtoms=30.0,basisvecs=None,kvecs=None,gS=2.0,gL=0.0,ffparamkey=None,ffqgrid=np.arange(0,10.0,0.01),ff=None):
        self.label=label
        self.magIdxs=magIdxs or [0]
        self.rmaxAtoms=30.0
        self.gS=gS
        self.gL=gL
        self.ffparamkey=ffparamkey
        self.ffqgrid=ffqgrid
        if struc==None:
            self.struc=[]
        else:
            self.struc=struc        
        if atoms==None:
            self.atoms=np.array([])
        else:
            self.atoms=atoms
        if spins==None:
            self.spins=np.array([])
        else:
            self.spins=spins
        if basisvecs==None:
            self.basisvecs=np.array([[0,0,1]])
        else:
            self.basisvecs=basisvecs
        if kvecs==None:
            self.kvecs=np.array([[0,0,0]])
        else:
            self.kvecs=kvecs
        if ff==None:
            self.ff=np.array([])
        else:
            self.ff=ff

    def makeAtoms(self):
        """Generate the Cartesian coordinates of the atoms for this species.
        """
        self.atoms=generateAtomsXYZ(self.struc,self.rmaxAtoms,self.magIdxs)

    def makeSpins(self):
        """Generate the Cartesian coordinates of the spin vectors in the
               structure. Must have a propagation vector, spin origin, and
               starting spin vector.
        """
        self.spins=generateSpinsXYZ(self.struc,self.atoms,self.kvecs,self.basisvecs)
    
    def makeFF(self):    
        g=self.gS+self.gL
        if getFFparams(self.ffparamkey) != ['none']:
            self.ff=self.gS/g * jCalc(self.ffqgrid,getFFparams(self.ffparamkey))+self.gL/g * jCalc(self.ffqgrid,getFFparams(self.ffparamkey),j2=True)
        else:
            print 'Using generic magnetic form factor.'
            self.ff=jCalc(self.ffqgrid)

    def copy(self):
        """Return a deep copy of the mPDFcalculator object."""
        temp=[self]        
        return copy.deepcopy(temp)[0]

class magStructure:
    """Extend the diffpy.Structure class to include magnetic attributes.
    """

    def __init__(self,struc=None,species=None,atoms=None,spins=None,gfactors=None,rmaxAtoms=30.0,ffqgrid=np.arange(0,10,0.01),ff=None):
        self.rmaxAtoms=rmaxAtoms
        self.ffqgrid=ffqgrid

        if struc==None:
            self.struc=[]
        else:
            self.struc=struc        
        if atoms==None:
            self.atoms=np.array([])
        else:
            self.atoms=atoms
        if spins==None:
            self.spins=np.array([])
        else:
            self.spins=spins
        if gfactors==None:
            self.gfactors=np.array([2.0])
        else:
            self.gfactors=gfactors
        if species==None:
            self.species={}
        else:
            self.species=species
        if ff==None:
            self.ff=np.array([])
        else:
            self.ff=ff

    def makeSpecies(self,label='mag1',magIdxs=None,atoms=None,spins=None,basisvecs=None,kvecs=None,gS=2.0,gL=0.0,ffparamkey=None,ffqgrid=np.arange(0,10.0,0.01),ff=None):
        self.species[label]=magSpecies(self.struc,label,magIdxs,atoms,spins,self.rmaxAtoms,basisvecs,kvecs,gS,gL,ffparamkey,ffqgrid,ff)

    def loadSpecies(self,magSpec):
        """Add a copy of an already-existing magSpecies object
        """
        self.species[magSpec.label]=magSpec

    def removeSpecies(self,label):
        del self.species[label]

    def makeAtoms(self):
        """Generate the Cartesian coordinates of the atoms in the structure."""
        temp=np.array([[0,0,0]])
        for key in self.species:
            self.species[key].makeAtoms()
            temp=np.concatenate((temp,self.species[key].atoms))
        self.atoms=temp[1:]
    
    def makeSpins(self):
        """Generate the Cartesian coordinates of the spin vectors in the
               structure. Must have a propagation vector, spin origin, and
               starting spin vector.
        """
        temp=np.array([[0,0,0]])
        for key in self.species:
            self.species[key].makeSpins()
            temp=np.concatenate((temp,self.species[key].spins))
        self.spins=temp[1:]

    def makeGfactors(self):
        temp=np.array([2.0])
        for key in self.species:
            temp=np.concatenate((temp,(self.species[key].gS+self.species[key].gL)*np.ones(self.species[key].spins.shape[0])))
        self.gfactors=temp[1:]
    
    def makeFF(self):
        try:
            self.ffqgrid=self.species.values()[0].ffqgrid
            self.ff=np.zeros_like(self.ffqgrid)
            totatoms=0.0
            for key in self.species:
                totatoms+=self.species[key].atoms.shape[0]
            for key in self.species:
                frac=float(self.species[key].atoms.shape[0])/totatoms
                self.species[key].makeFF()
                self.ff+=frac*self.species[key].ff
        except:
            print 'Check that all mag species have same q-grid.'

    def makeAll(self):
        self.makeAtoms()
        self.makeSpins()
        self.makeGfactors()
        self.makeFF()

    def copy(self):
        """Return a deep copy of the mPDFcalculator object."""
        temp=[self]        
        return copy.deepcopy(temp)[0]

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
    def __init__(self,magstruc=None,calcList=[0],maxextension=10.0,gaussPeakWidth=0.1,dampRate=0.0,dampPower=2.0,qmin=-1.0,qmax=-1.0,rmin=0.0,rmax=20.0,rstep=0.01,ordScale=1.0/np.sqrt(2*np.pi),paraScale=1.0,rmintr=-5.0,rmaxtr=5.0,drtr=0.01):
        if magstruc==None:
            self.magstruc=[]
        else:
            self.magstruc=magstruc
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
        rcalc,frcalc=calculatemPDF(self.magstruc.atoms,self.magstruc.spins,self.magstruc.gfactors,self.calcList,self.rstep,self.rmin,self.rmax,self.gaussPeakWidth,self.qmin,self.qmax,self.dampRate,self.dampPower,self.maxextension)
        if normalized and not both: 
            return rcalc,frcalc
        elif not normalized and not both:
            Drcalc=calculateDr(rcalc,frcalc,self.magstruc.ffqgrid,self.magstruc.ff,self.paraScale,self.ordScale,self.rmintr,self.rmaxtr,self.drtr,self.qmin,self.qmax)
            return rcalc,Drcalc
        else:
            Drcalc=calculateDr(rcalc,frcalc,self.magstruc.ffqgrid,self.magstruc.ff,self.paraScale,self.ordScale,self.rmintr,self.rmaxtr,self.drtr,self.qmin,self.qmax)            
            return rcalc,frcalc,Drcalc

    def plot(self,normalized=True,both=False):
        """Plot the magnetic PDF.

        Args:
            normalized (boolean): indicates whether or not the normalized mPDF
                should be plotted.
            both (boolean): indicates whether or not both normalized and
                unnormalized mPDF quantities should be plotted.
        """
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_xlabel('r ($\AA$)')
        ax.set_xlim(xmin=self.rmin,xmax=self.rmax)        
        rcalc,frcalc=calculatemPDF(self.magstruc.atoms,self.magstruc.spins,self.magstruc.gfactors,np.array([2.0]),self.calcList,self.rstep,self.rmin,self.rmax,self.gaussPeakWidth,self.qmin,self.qmax,self.dampRate,self.dampPower,self.maxextension)
        if normalized and not both: 
            ax.plot(rcalc,frcalc)
            ax.set_ylabel('f ($\AA^{-2}$)')
        elif not normalized and not both:
            Drcalc=calculateDr(rcalc,frcalc,self.magstruc.ffqgrid,self.magstruc.ff,self.paraScale,self.ordScale,self.rmintr,self.rmaxtr,self.drtr,self.qmin,self.qmax)
            ax.plot(rcalc,Drcalc)            
            ax.set_ylabel('D ($\AA^{-2}$)')
        else:
            Drcalc=calculateDr(rcalc,frcalc,self.magstruc.ffqgrid,self.magstruc.ff,self.paraScale,self.ordScale,self.rmintr,self.rmaxtr,self.drtr,self.qmin,self.qmax)            
            ax.plot(rcalc,frcalc,'b-',label='f(r)')
            ax.plot(rcalc,Drcalc,'r-',label='D(r)')
            ax.set_ylabel('f, D ($\AA^{-2}$)')
            plt.legend(loc='best')
        plt.show()

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

def cubicPBC(atomlist,spinlist,a):
    x,y,z = np.transpose(atomlist)
    nUnitCells=np.ceil((np.max(x)-np.min(x))/a)
    A=nUnitCells*a
    translationVecs=[np.array([A,0,0]),np.array([0,A,0]),np.array([0,0,A]), \
    np.array([A,A,0]),np.array([A,0,A]),np.array([0,A,A]),np.array([A,-A,0]),np.array([A,0,-A]),np.array([0,A,-A]), \
    np.array([A,A,A]),np.array([A,A,-A]),np.array([A,-A,A]),np.array([-A,A,A])]
    atomlistBig=atomlist
    spinlistBig=spinlist
    for vec in translationVecs:
        atomlistBig,spinlistBig=np.concatenate((atomlistBig,atomlist-vec)),np.concatenate((spinlistBig,spinlist))
        atomlistBig,spinlistBig=np.concatenate((atomlistBig,atomlist+vec)),np.concatenate((spinlistBig,spinlist))
    return atomlistBig,spinlistBig    

def inFold(vecs,xmin,xmax,ymin,ymax,zmin,zmax):
    X,Y,Z=xmax-xmin,ymax-ymin,zmax-zmin
    x,y,z=np.transpose(vecs)
    xbool,ybool,zbool=0*x,0*y,0*z
    xbool[x<xmin]=1.0
    xbool[x>=xmax]=-1.0
    ybool[y<ymin]=1.0
    ybool[y>=ymax]=-1.0
    zbool[z<zmin]=1.0
    zbool[z>=zmax]=-1.0
    return vecs+np.transpose((xbool*X,ybool*Y,zbool*Z))

## try a version of inFold where it is based on unitcell fractional
## coordinates instead of Cartesian coordinatesaa

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
    return qgrid, rv

def calculateIQinFold(xyz, sxyz, uclist, qgrid,f,xmin,xmax,ymin,ymax,zmin,zmax,rstep=0.02,rmin=1.0, rmax=15.0):
    #qgrid = np.arange(qmin, qmax, qstep)
    S=np.linalg.norm(sxyz[0])
    r = np.arange(rmin, rmax, rstep)
    rbin =  np.concatenate([r-rstep/2, [r[-1]+rstep/2]])
    
    s1 = np.zeros(len(r))
    s2 = np.zeros(len(r))

    xyzOrig=xyz.copy()
    
    for i in range(len(uclist)):
        print 'Working on: '+str(i+1)+'/'+str(len(uclist))
        uu = uclist[i]
        
        xyz=inFold(xyz-xyz[uu],xmin,xmax,ymin,ymax,zmin,zmax)
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
        
        xyz=xyzOrig.copy()

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
    return qgrid, rv

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
    return qgrid, rv

    
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

