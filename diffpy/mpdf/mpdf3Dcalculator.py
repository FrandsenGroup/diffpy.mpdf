#!/usr/bin/env python
##############################################################################
#
# diffpy.mpdf         by Frandsen Group
#                     Benjamin A. Frandsen benfrandsen@byu.edu
#                     (c) 2022 Benjamin Allen Frandsen
#                      All rights reserved
#
# File coded by:    Parker Hamilton and Benjamin Frandsen
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""class to perform 3D-mPDF calculations"""

import copy
import numpy as np
from scipy.signal import convolve, correlate
from diffpy.mpdf.magutils import gauss, ups, vec_con, vec_ac

class MPDF3Dcalculator:
    """Create an MPDF3Dcalculator object to help calculate 3D-mPDF functions
    This class is loosely modelled after the PDFcalculator cless in diffpy.
    At minimum, tie requires a magnetic structure with atoms and spins and will
    calculate the 3D-mPDF from that.
    Args:
        magstruc (MagStructure object): gives the information about the magnetic
            structure. Must have arrays of atoms and spins
        gaussPeakWidth (float): The width of the gaussian function that represents atoms
        label (string): Optional label from the MPDF3Dcalculator
    """

    def __init__(self, magstruc=None, gaussPeakWidth=0.5, label=""):
        if magstruc is None:
            self.magstruc = []
        else:
            self.magstruc = magstruc

        self.gaussPeakWidth = gaussPeakWidth
        self.label = label
        self.Nx = None
        self.Ny = None
        self.Nz = None
        self.dr = None

    def __repr__(self):
        if self.label == None:
            return "3DMPDFcalculator() object"
        else:
            return self.label +  ": 3DMPDFcalculator() object"

    def calc(self, verbose=False, dr=None, originIdx=0):
        """Calculate the 3D magnetic PDF
        Args:
            verbose (boolean): indicates whether to output progress 
            dr (float): the grid spacing to use
            originIdx (int): index of spin to be used as origin
                if a correlation length is being applied
        """

        if dr is not None:
            self.dr = dr
        self._makeRgrid()

        s_arr = np.zeros((self.Nx,self.Ny,self.Nz,3))
        if verbose :
            print("Setting up point spins")

        # Use the scaled spins if the correlation length has been set;
        # otherwise, use the full magnitude spins
        spins = self.magstruc.generateScaledSpins(originIdx)       

        for i in range(len(self.magstruc.atoms)):
            idx = np.rint((self.magstruc.atoms[i] - self.rmin)/self.dr).astype(int) 
            s_arr[idx[0],idx[1],idx[2]] = spins[i]

        if verbose:
            print("Setting up filter grid")

        filter_x = np.arange(-3,3+self.dr,self.dr)
        X,Y,Z = np.meshgrid(filter_x,filter_x,filter_x,indexing='ij')
        filter_grid = np.moveaxis([X,Y,Z],0,-1)

        if verbose:
            print("Making filters")

        gaussian = gauss(filter_grid,s=self.gaussPeakWidth)
        upsilon = ups(filter_grid)

        if verbose:
            print("Convolving spin array")

        s_arr[:,:,:,0] = convolve(s_arr[:,:,:,0],gaussian,mode='same')*self.dr**3
        s_arr[:,:,:,1] = convolve(s_arr[:,:,:,1],gaussian,mode='same')*self.dr**3
        s_arr[:,:,:,2] = convolve(s_arr[:,:,:,2],gaussian,mode='same')*self.dr**3

        if verbose:
            print("Computing mpdf")

        mag_ups = vec_con(s_arr,upsilon,self.dr)
        if verbose:
            print("comp1")
        self.mpdf = vec_ac(s_arr,s_arr,self.dr,"full")
        #comp1 = vec_ac(s_arr,s_arr,self.dr,"full")
        if verbose:
            print("comp2")
        self.mpdf += -1/(np.pi**4)*correlate(mag_ups,mag_ups,mode="full")*self.dr**3
        #comp2 = correlate(mag_ups,mag_ups,mode="full")*self.dr**3
        if verbose:
            print("mpdf")
        #self.mpdf = comp1 - 1/(np.pi**4)*comp2
        return 

    def _makeRgrid(self,dr = None,buf=0):
        """Set up bounds and intervals of the spatial grid to use
        Args:
            dr (float): the grid spacing to use
            buf (float): the space to include on either side of the 
                spin distribution
        """
        if dr is not None:
            self.dr = dr
        if self.dr is None:
            self.dr = 0.2
        pos = np.array([a for a in self.magstruc.atoms])
        x_min = np.min(pos[:,0]) - buf
        x_max = np.max(pos[:,0]) + buf
        y_min = np.min(pos[:,1]) - buf
        y_max = np.max(pos[:,1]) + buf
        z_min = np.min(pos[:,2]) - buf
        z_max = np.max(pos[:,2]) + buf

        x = np.arange(x_min,x_max + self.dr, self.dr)
        y = np.arange(y_min,y_max + self.dr, self.dr)
        z = np.arange(z_min,z_max + self.dr, self.dr)
        N_x = len(x)
        N_y = len(y)
        N_z = len(z)
        x_max = np.max(x)
        y_max = np.max(y)
        z_max = np.max(z)
        X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
        
        rgrid = np.moveaxis([X,Y,Z],0,-1)
        
        self.Nx = N_x
        self.Ny = N_y
        self.Nz = N_z
        self.rmin = np.array([x_min,y_min,z_min])
        self.rmax = np.array([x_max,y_max,z_max])
        return rgrid

    def plot(self):
        """Plot the 3D-mPDF
        ToDo: implement plotting, use Jacobs visualilze
        """
        raise NotImplementedError()

    def runChecks(self):
        """Runs bounds and compatibility checks for internal variables. 
            This should be called during __init__
         
        ToDo: implement for troubleshooting
        """        
        raise NotImplementedError()

    def rgrid(self):
        """Returns the spatial grid the 3D-mPDF is output on
        Generates the spatial grid for the 3D-mPDF when needed by
        the user
        """

        if self.dr is None:
            self._makeRgrid()
        X = np.arange(-(self.Nx-1)*self.dr,stop = (self.Nx-1)*self.dr+self.dr/2,step = self.dr)
        X[self.Nx-1] = 0
        
        Y = np.arange(-(self.Ny-1)*self.dr,stop = (self.Ny-1)*self.dr+self.dr/2,step = self.dr)
        Y[self.Ny-1] = 0
        
        Z = np.arange(-(self.Nz-1)*self.dr,stop = (self.Nz-1)*self.dr+self.dr/2,step = self.dr)
        Z[self.Nz-1] = 0
        

        return X,Y,Z


    def copy(self):
        '''
        Return a deep copy of the object
        '''
        return copy.deepcopy(self)
