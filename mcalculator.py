#!/usr/bin/env python
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, convolve
import diffpy as dp

'''
xyz: 2d array (3*n), position of spins
sxyz: 2d array (3*n), direction of spins
uclist: 1d array, m, a list of spin to calculate the mPDF,
    example: [10, 20, 30] -> calculate the mPDF of no.10, no.20, no.30 spin in the list
qgrid: 1d array, Q grid of IQ
rstep, rmax: float, define the r grid of PDF 
'''
def j0calc(q,params):
    '''
    Module to calculate the magnetic form factor j0 based on the tabulated analytical approximations.
    
    Inputs: array q giving the q grid, and list params, containing the coefficients for the analytical approximation to j0 as contained in tables by Brown.
    
    Returns: array with same shape as q giving the magnetic form factor j0.
    '''
    [A,a,B,b,C,c,D] = params
    return A*np.exp(-a*(q/4/np.pi)**2)+B*np.exp(-b*(q/4/np.pi)**2)+C*np.exp(-c*(q/4/np.pi)**2)+D

def cv(x1,y1,x2,y2):
    '''
    Module to compute convolution of functions y1 and y2.
    
    Inputs: array y1, x1, y2, x2. Should have the same grid spacing to be safe.
    
    Returns: arrays ycv and xcv giving the convolution.
    '''
    dx=x1[1]-x1[0]
    ycv = dx*np.convolve(y1,y2,'full')
    xcv=np.linspace(x1[0]+x2[0],x1[-1]+x2[-1],len(ycv))

    return xcv,ycv
    
def costransform(q,fq,rmin=0.0,rmax=50.0,rstep=0.1): # does not require even q-grid
    '''
    Module to compute cosine Fourier transform of f(q). Uses direct integration rather than FFT and does not require an even q-grid.
    
    Inputs: array q (>=0 only), array f(q) to be transformed, optional arguments giving rmin, rmax, and rstep of output r-grid.
    
    Returns: arrays r and fr, where fr is the cosine Fourier transform of fq.
    '''
    lostep = int(np.ceil((rmin - 1e-8) / rstep))
    histep = int(np.floor((rmax + 1e-8) / rstep)) + 1
    r = np.arange(lostep,histep)*rstep
    qrmat=np.outer(r,q)
    integrand=fq*np.cos(qrmat)
    fr=np.sqrt(2.0/np.pi)*np.trapz(integrand,q)
    return r,fr

	
def getDiffData(fileNames,fmt='pdfgui',writedata=False):
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
	
def calculatemPDF(xyz, sxyz, calcList=np.array([0]), rstep=0.01, rmin=0, rmax=20, psigma=0.1,qmin=-1,qmax=-1,dampRate=0.0,dampPower=2.0,maxextension=10.0):
    
    # calculate s1, s2
    r = np.arange(rmin+rstep, rmax+maxextension, rstep)
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
    fr = s1 / r + r * (ss2[-1] - ss2)
    fr /= len(calcList)

    fr *= np.exp(-1.0*(dampRate*r)**dampPower)
    # Do the convolution with the termination function if qmin/qmax have been given
    if qmin >= 0 and qmax > qmin:
        th=(np.sin(qmax*r)-np.sin(qmin*r))/np.pi/r
        rcv,frcv=cv(r,fr,r,th)
    else:
        rcv,frcv=r,fr
    #fr[0] = 0
    return rcv[np.logical_and(rcv>=r[0]-0.5*rstep,rcv<=rmax+0.5*rstep)], frcv[np.logical_and(rcv>=r[0]-0.5*rstep,rcv<=rmax+0.5*rstep)]
    

def calculateDr(r,fr,q,ff,paraScale=1.0,orderedScale=1.0/np.sqrt(2*np.pi),rmintr=-5.0,rmaxtr=5.0,drtr=0.01):
    rsr,sr=costransform(q,ff,rmintr,rmaxtr,drtr)
    sr=np.sqrt(np.pi/2.0)*sr
    rSr,Sr=cv(rsr,sr,rsr,sr)
    para=-1.0*np.sqrt(2.0*np.pi)*np.gradient(Sr,rSr[1]-rSr[0]) ### paramagnetic term in d(r)
    rDr,Dr=cv(r,fr,rSr,Sr)
    Dr*=orderedScale
    Dr[:len(para)]+=para*paraScale
    dr=r[1]-r[0]
    return Dr[np.logical_and(rDr>=np.min(r)-0.5*dr,rDr<=np.max(r)+0.5*dr)]
    
def generateAtomsXYZ(struc,rmax=30.0,magIdxs=[0]):
    '''
    Module to spit out the xyz Cartesian coordinates of magnetic atoms in a structure.

    struc = diffpy.structure object
    rmax = float, largest distance from central spin that should be included
    magIdxs = list of integers giving indices of magnetic atoms in the structure

    Note: This will only work well for structures that can be expressed with a unit cell that is close to at least orthorhombic.
    '''
    lat=struc.lattice
    unitcell=lat.stdbase
    cellwithatoms=struc.xyz_cartn[np.array(magIdxs)]
    radius=1.5*rmax
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

def generateSpinsXYZ(struc,atoms=np.array([]),origin=np.array([0,0,0]),kvec=np.array([0,0,0]),svec=np.array([0,0,1])):
    '''
    Module to spit out the xyz Cartesian coordinates of the spins in the same order
        as the list of atoms (i.e. spin positions). Requires a propagation vector.

    atoms = N x 3 array of the Cartesian coordinates of the N spins.
    origin = Cartesian coordinates of position of the spin chosen to be at the origin
    Qvec = propagation vector in rlat units
    svec = Cartesian coordinates giving orientation of the spin chosen to be at the origin

    Note: only works for collinear structures at the moment.
    '''
    lat=struc.lattice
    rlat=lat.reciprocal()
    astar,bstar,cstar=rlat.cartesian((1,0,0)),rlat.cartesian((0,1,0)),rlat.cartesian((0,0,1))
    kcart=kvec[0]*astar+kvec[1]*bstar+kvec[2]*cstar
    mags=np.cos(2*3.141593*np.dot(atoms-origin,kcart))
    spins=svec*mags.reshape(-1,1)
    return spins

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


class mPDFcalculator:
    ''' mPDFcalculator class.

    '''
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
        self.atoms=generateAtomsXYZ(self.struc,self.rmaxAtoms,self.magIdxs)

    def makeSpins(self):
        self.spins=generateSpinsXYZ(self.struc,self.atoms,self.spinOrigin,self.kvec,self.svec)

    def calc(self,normalized=True,both=False):
        rcalc,frcalc=calculatemPDF(self.atoms,self.spins,self.calcList,self.rstep,self.rmin,self.rmax,self.gaussPeakWidth,self.qmin,self.qmax,self.dampRate,self.dampPower,self.maxextension)
        if normalized and not both: 
            return rcalc,frcalc
        elif not normalized and not both:
            Drcalc=calculateDr(rcalc,frcalc,self.ffqgrid,self.ff,self.paraScale,self.ordScale,self.rmintr,self.rmaxtr,self.drtr)
            return rcalc,Drcalc
        else:
            Drcalc=calculateDr(rcalc,frcalc,self.ffqgrid,self.ff,self.paraScale,self.ordScale,self.rmintr,self.rmaxtr,self.drtr)            
            return rcalc,frcalc,Drcalc

    def copy(self):
        return self

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
