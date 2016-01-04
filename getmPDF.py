import scipy
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.linalg import lstsq

def readin(infofile):
    '''
    This module reads in the information file containing the details for the
    mPDF transform: datafile, rinfo, qinfo, j0info. See 'infofile_template.txt' for an example.
    
    Input: text file containing the required information.
    
    Returns: list [datafile,[qmin,qmax],[rmin,rmax,rstep],j0info,[rplmin,rplmax],qplots]
    '''
    ### Read in the metadata from the plot file
    metadata=open(infofile,'r').readlines()
    
    ### Get rid of pesky newlines
    for i in range(len(metadata)):
        metadata[i] = metadata[i].strip()
    
    ### Extract name of datafile
    datafile=metadata[metadata.index('datafile')+1]
    
    ### Extract q-grid info
    [qmin,qmax]=map(float,metadata[metadata.index('qinfo')+1].split(','))
    
    ### Extract r-grid info
    [rmin,rmax,rstep]=map(float,metadata[metadata.index('rinfo')+1].split(','))
    
    ### Extract magnetic form factor parameters
    j0info=map(float,metadata[metadata.index('j0info')+1].split(','))
    
    ### Extract mPDF real-space plot info, if provided
    try:
        [rplmin,rplmax]=map(float,metadata[metadata.index('rplotinfo')+1].split(','))
    except:
        [rplmin,rplmax]=[0,30.0]

    ### Extract mPDF q-space plot info, if provided
    try:
        qplots=metadata[metadata.index('qplotinfo')+1]
        qplots=[s.strip() for s in qplots.split(',')]
    except:
        qplots=['iq','sq','fq','ff']
        
    return [datafile,[qmin,qmax],[rmin,rmax,rstep],j0info,[rplmin,rplmax],qplots]


def j0calc(q,params):
    '''
    Module to calculate the magnetic form factor j0 based on the tabulated analytical approximations.
    
    Inputs: array q giving the q grid, and list params, containing the coefficients for the analytical approximation to j0 as contained in tables by Brown.
    
    Returns: array with same shape as q giving the magnetic form factor j0.
    '''
    [A,a,B,b,C,c,D] = params
    return A*np.exp(-a*(q/4/np.pi)**2)+B*np.exp(-b*(q/4/np.pi)**2)+C*np.exp(-c*(q/4/np.pi)**2)+D

def scalefit(i,f):
    '''
    Module to properly normalize the intensity i(q) according to the form factor to produce s(q).
    
    Inputs: array i giving i(q) and array f giving the form factor j0.
    
    Returns: scaling factor scale (float) and the properly scaled s(q)
    '''
    yn = i/(f**2)
    fn = (f**2)/(f**2)
    scale = np.dot(yn,fn)/np.dot(yn,yn)
    return scale,scale*yn - fn + 1.0

def polybkdfit(q,sq,porder):
    '''
    Module to fit a polynomial background to s(q).
    
    Inputs: array q, array s(q), and desired order of the background polynomial to be fit.
    
    Returns: polynomial coefficients in array p.
    '''
    qscale=q[-1]
    qsc = q / qscale
    Mv0 = np.vander(qsc,porder+1)
    Mv1 = Mv0[:,:-1]
    yfq = q * (sq - 1.0)
    p,resids,rank,s=lstsq(Mv1,yfq)
    p /= np.vander([qscale],porder+1)[0,:-1]
    return p

def transform(q,fq,rmin=0.0,rmax=50.0,rstep=0.1): # does not require even q-grid
    '''
    Module to compute sine Fourier transform of f(q). Uses direct integration rather than FFT and does not require an even q-grid.
    
    Inputs: array q, array f(q) to be transformed, optional arguments giving rmin, rmax, and rstep of output r-grid.
    
    Returns: arrays r and fr, where fr is the sine Fourier transform of fq.
    '''
    lostep = int(np.ceil((rmin - 1e-8) / rstep))
    histep = int(np.floor((rmax + 1e-8) / rstep)) + 1
    r = np.arange(lostep,histep)*rstep
    qrmat=np.outer(r,q)
    integrand=fq*np.sin(qrmat)
    fr=(2/np.pi)*np.trapz(integrand,q)
    return r,fr
    
def transform_pdfgetx3(q,fq,rmin=0.0,rmax=50.0,rstep=0.1): # requires even q-grid
    '''
    Module to compute sine Fourier transform of f(q). Uses the FFT as in pdfgetx3. Requires an even q-grid. Take caution with the zero-padding to ensure clean FFT.
    
    Inputs: array q, array f(q) to be transformed, optional arguments giving rmin, rmax, and rstep of output r-grid.
    
    Returns: arrays xout and yout, where fr is the sine Fourier transform of fq.
    '''
    lostep = int(np.ceil((rmin - 1e-8) / rstep))
    histep = int(np.floor((rmax + 1e-8) / rstep)) + 1
    xout = np.arange(lostep,histep)*rstep
    qstep = q[1] - q[0]
    qmaxrstep = np.pi/rstep
    nin = len(q)
    nbase = max([nin,histep,qmaxrstep/qstep])
    nlog2 = int(np.ceil(np.log2(nbase)))
    nout = 2**nlog2
    qmaxdb = 2*nout*qstep
    yindb=np.concatenate((fq,np.zeros(2*nout - nin)))
    cyoutdb = np.fft.ifft(yindb)*2/np.pi*qmaxdb
    youtdb = np.imag(cyoutdb)
    xstepfine = 2*np.pi/qmaxdb
    xoutfine = np.arange(nout) * xstepfine
    youtfine = youtdb[:nout]
    yout = np.interp(xout, xoutfine, youtfine)
    return xout, yout

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

def main():
    '''
    Workflow module. The mPDF is obtained through the following steps:
    1. Relevant information is read in from the info file.
    2. q-data are read in.
    3. Normalize i(q) to obtain s(q) and f(q).
    4. Compute Fourier transform either by direct integration (safer) or FFT (still has some problems).
    5. Generate plots.
    6. Provide options for saving data.
    '''
    ### Read in the infofile
    infofile = sys.argv[1]
    [datafile,[qmin,qmax],[rmin,rmax,rstep],j0params,[rplmin,rplmax],qplots] = readin(infofile)
    
    ### read in the data, restrict to q-range of interest
    q,i=np.loadtxt(datafile,unpack=True)[0:2]
    lowq=q<qmax
    q=q[lowq]
    i=i[lowq]
    highq=q>qmin
    q=q[highq]
    i=i[highq]
    
    ### calculate
    f=j0calc(q,j0params)
    scale,sq = scalefit(i,f)
    print scale
    porder=8
    p = polybkdfit(q,sq,porder)
    sq = sq - np.polyval(p,q)
    fq = q*(sq-1.0)
    r,fr=transform(q,fq,rmin,rmax,rstep)
    r2,dr=transform(q,q*i,rmin,rmax,rstep)
    
    ### q-space plots
    fig=plt.figure()
    ax=fig.add_subplot(111)
    if 'iq' in qplots:
        ax.plot(q,scale*i,'b-',label='I(Q)')
    if 'sq' in qplots:
        ax.plot(q,sq,'k-',label='S(Q)')
    if 'fq' in qplots:
        ax.plot(q,fq,'g-',label='F(Q)')
    if 'ff' in qplots:
        ax.plot(q,f**2,'r-',label='F.F.')
    ax.legend(loc='best',fancybox=True,shadow=True)
    ax.set_xlim(xmin=qmin,xmax=qmax)
    ax.set_xlabel('Q ($\AA ^{-1}$)')
    ax.set_ylabel('Intensity (arb. units)')
    
    ### real-space plots
    fig2=plt.figure()    
    ax2=fig2.add_subplot(111)
    ax2.plot(r,fr/np.max(fr),'b-')
    ax2.plot(r2,dr/np.max(dr),'r-')
    #ax2.plot(r,fr/np.max(fr),'b-',r,dr/np.max(dr),'r-')
    #ax2.plot(r,fr/np.max(fr)/15.0,'b-',r,dr/np.max(dr),'r-')
    #ax2.set_ylim(ymin=-0.1,ymax=0.1)
    ax2.plot(r,np.zeros_like(r),'k--')
    ax2.set_xlim(xmin=rplmin,xmax=rplmax)
    ax2.set_ylabel('f ($\AA ^{-2}$)')
    ax2.set_xlabel('r ($\AA$)')
    plt.show()
    
    ### provide options to save data
    go = True
    while go:
        choice = raw_input('Save data? (fq, sq, ff, fr, dr, none): ')
        if choice == 'fq':
            savefile=raw_input('File name for F(q): ')
            np.savetxt(savefile,np.transpose([q,fq]))
            if raw_input('Save another file? (y/n): ') != 'y':
                go = False
        elif choice == 'sq':
            savefile=raw_input('File name for S(q): ')
            np.savetxt(savefile,np.transpose([q,sq]))
            if raw_input('Save another file? (y/n): ') != 'y':
                go = False
        elif choice == 'ff':
            savefile=raw_input('File name for f(q)**2: ')
            np.savetxt(savefile,np.transpose([q,f**2]))
            if raw_input('Save another file? (y/n): ') != 'y':
                go = False
        elif choice == 'fr':
            savefile=raw_input('File name for f(r): ')
            np.savetxt(savefile,np.transpose([r,fr]))
            if raw_input('Save another file? (y/n): ') != 'y':
                go = False
        elif choice == 'dr':
            savefile=raw_input('File name for d(r): ')
            np.savetxt(savefile,np.transpose([r,dr]))
            if raw_input('Save another file? (y/n): ') != 'y':
                go = False
        else:
            print 'No file will be saved. Closing up shop.'
            go = False
    
    return r,fr

if __name__ == "__main__":

    main()

# End of file

### original pdfgetx3 code
    
    # def transform(self):
        # from numpy.fft import ifft
        # lostep = int(numpy.ceil((self.rmin - 1e-8) / self.rstep))
        # histep = int(numpy.floor((self.rmax + 1e-8) / self.rstep)) + 1
        # self.xout = numpy.arange(lostep, histep) * self.rstep
        # self.qstep = self.xin[1] - self.xin[0]
        # self.qmaxrstep = numpy.pi / self.rstep
        # nin = len(self.xin)
        # nbase = max([nin, histep, self.qmaxrstep / self.qstep])
        # nlog2 = int(numpy.ceil(numpy.log2(nbase)))
        # nout = 2 ** nlog2
        # qmaxdb = 2 * nout * self.qstep
        # yindb = numpy.concatenate((self.yin, numpy.zeros(2 * nout - nin)))
        # cyoutdb = ifft(yindb) * 2 / numpy.pi * qmaxdb
        # youtdb = numpy.imag(cyoutdb)
        # xstepfine = 2 * numpy.pi / qmaxdb
        # xoutfine = numpy.arange(nout) * xstepfine
        # youtfine = youtdb[:nout]
        # self.yout = numpy.interp(self.xout, xoutfine, youtfine)
        # return
    
#saveyn=raw_input('Save file? (y/n)')
#if saveyn=='y':
#    np.savetxt('magonly300K.iq',np.transpose([Q,magfixed]))
