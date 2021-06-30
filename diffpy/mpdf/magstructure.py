#!/usr/bin/env python
##############################################################################
#
# diffpy.mpdf         by Billinge Group
#                     Simon J. L. Billinge sb2896@columbia.edu
#                     (c) 2016 trustees of Columbia University in the City of
#                           New York.
#                      All rights reserved
#
# File coded by:    Benjamin Frandsen
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################


"""classes to create magnetic structures for mPDF calculations."""

import copy
import numpy as np
from diffpy.srreal.bondcalculator import BondCalculator
from diffpy.mpdf.magutils import generateAtomsXYZ, generateFromUnitCell, \
    generateSpinsXYZ, getFFparams, jCalc, spinsFromAtoms, atomsFromSpins, \
    findAtomIndices, visualizeSpins

class MagSpecies:
    """Store information for a single species of magnetic atom.


    This class takes a diffpy.Structure object and uses it to generate spins
    based on a set of propagation vectors and basis vectors. For more info
    about magnetic propagation vectors, visit e.g.
    http://andrewsteele.co.uk/physics/mmcalc/docs/magneticpropagationvector

    Args:
        struc (diffpy.Structure object): provides lattice parameters and unit
            cell of desired structure.
        label (string): label for this particular magnetic species. Should be
            different from the labels for any other magnetic species you make.
        strucIdxs (python list): list of integers giving indices of magnetic
            atoms in the unit cell
        atoms (numpy array): list of atomic coordinates of all the magnetic
            atoms in the structure; e.g. generated by generateAtomsXYZ()
        spins (numpy array): triplets giving the spin vectors of all the
            atoms, in the same order as the atoms array provided as input.
            In units of hbar.
        calcIdxs (python list): list giving the indices of the atoms array
            specifying the atoms to be used as the origin when calculating
            the mPDF. If given the string argument 'all', then every atom
            will be used (potentially causing very long calculation times).
            These indices are relative to the atoms array for this specific
            MagSpecies, not relative to the atoms array for the
            MagStructure as a whole.
        rmaxAtoms (float): maximum distance from the origin of atomic
            positions generated by the makeAtoms method.
        basisvecs (numpy array): nested three-vector(s) giving the basis
            vectors to generate the spins. e.g. np.array([[0, 0, 1]]). Any
            phase factor should be included directly with the basisvecs.
        kvecs (numpy array): nested three-vector(s) giving the propagation
            vectors for the magnetic structure in r.l.u.,
            e.g. np.array([[0.5, 0.5, 0.5]])
        S (float): Spin angular momentum quantum number in units of hbar.
        L (float): Orbital angular momentum quantum number in units of hbar.
        J (float): Total angular momentum quantum number in units of hbar.
        gS (float): spin component of the Lande g-factor (g = gS+gL)
        gL (float): orbital component of the Lande g-factor
        ffparamkey (string): gives the appropriate key for getFFparams()
        ffqgrid (numpy array): grid of momentum transfer values used for
            calculating the magnetic form factor.
        ff (numpy array): magnetic form factor.
        useDiffpyStruc (boolean): True if atoms/spins to be generated from
            a diffpy structure object; False if a user-provided unit cell is
            to be used. Note that there may be some problems with user-
            provided unit cells with lattice angles strongly deviated from
            90 degrees.
        latVecs (numpy array): Provides the unit cell lattice vectors as
            np.array([avec, bvec, cvec]). Only useful if useDiffpyStruc = False.
        atomBasis (numpy array): Provides positions of the magnetic atoms
            in fractional coordinates within the unit cell. Only useful if
            useDiffpyStruc = False. Example: np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        spinBasis (numpy array): Provides the orientations of the spins in
            the unit cell, in the same order as atomBasis. Only useful if
            useDiffpyStruc = False. Example: np.array([[0, 0, 1], [0, 0, -1]]
        spinOrigin (numpy array): Cartesian coordinates of the position that will
            be considered the origin when generating spin directions from basis
            vectors and propagation vectors. Default is np.array([0,0,0]).
        verbose (boolean): If True, will print messages relating to the structure.
            Useful for troubleshooting. Default is False.
    """
    def __init__(self, struc=None, label='', strucIdxs=[0], atoms=None, spins=None,
                 calcIdxs=[0], rmaxAtoms=30.0, basisvecs=None, kvecs=None, S=0.5,
                 L=0.0, J=None, gS=None, gL=None, ffparamkey=None,
                 ffqgrid=None, ff=None, useDiffpyStruc=True, latVecs=None,
                 atomBasis=None, spinBasis=None, spinOrigin=None, verbose=False):
        self.label = label
        self.rmaxAtoms = rmaxAtoms
        self.S = S
        self.L = L
        if J is None:
            J = S + L
            self.J = J
        else:
            self.J = J
        if gS is None:
            self.gS = 1.0 + 1.0*(S*(S+1)-L*(L+1))/(J*(J+1))
        else:
            self.gS = gS
        if gL is None:
            self.gL = 0.5 + 1.0*(L*(L+1)-S*(S+1))/(2*J*(J+1))
        else:
            self.gL = gL
        self.ffparamkey = ffparamkey
        self.useDiffpyStruc = useDiffpyStruc
        if strucIdxs is None:
            self.strucIdxs = [0]
        else:
            self.strucIdxs = strucIdxs
        if calcIdxs is None:
            self.calcIdxs = [0]
        else:
            self.calcIdxs = calcIdxs
        if struc is None:
            self.struc = []
        else:
            self.struc = struc
        if atoms is None:
            self.atoms = np.array([])
        else:
            self.atoms = atoms
        if spins is None:
            self.spins = np.array([])
        else:
            self.spins = spins
        if basisvecs is None:
            self.basisvecs = np.array([[0, 0, 1]])
        else:
            self.basisvecs = basisvecs
        if kvecs is None:
            self.kvecs = np.array([[0, 0, 0]])
        else:
            self.kvecs = kvecs
        if ff is None:
            self.ff = np.array([])
        else:
            self.ff = ff
        if ffqgrid is None:
            self.ffqgrid = np.arange(0, 10.0, 0.01)
        else:
            self.ffqgrid = ffqgrid
        if latVecs is None:
            self.latVecs = np.array([[4., 0, 0], [0, 4., 0], [0, 0, 4.]])
        else:
            self.ff = latVecs
        if atomBasis is None:
            self.atomBasis = np.array([[0, 0, 0]])
        else:
            self.atomBasis = atomBasis
        if spinBasis is None:
            self.spinBasis = np.array([[0, 0, 1]])
        else:
            self.spinBasis = spinBasis
        if spinOrigin is None:
            self.spinOrigin = np.array([[0, 0, 0]])
        else:
            self.spinOrigin = spinOrigin
        self.verbose = verbose

    def __repr__(self):
        if self.label == '':
            return 'MagSpecies() object'
        else:
            return self.label+': MagSpecies() object'

    def makeAtoms(self):
        """Generate the Cartesian coordinates of the atoms for this species.
        """
        if self.useDiffpyStruc:
            self.atoms = generateAtomsXYZ(self.struc, self.rmaxAtoms, self.strucIdxs)
        else:
            try:
                self.atoms, self.spins = generateFromUnitCell(self.latVecs,
                                                              self.atomBasis,
                                                              self.spinBasis,
                                                              self.rmaxAtoms)
            except:
                print('Please check latVecs, atomBasis, and spinBasis.')

    def makeSpins(self):
        """Generate the Cartesian coordinates of the spin vectors in the
               structure. Must provide propagation vector(s) and basis
               vector(s).
        """
        if self.useDiffpyStruc:
            self.spins = generateSpinsXYZ(self.struc, self.atoms, self.kvecs, self.basisvecs, 
                                          self.spinOrigin)
        else:
            print('Since you are not using a diffpy Structure object,')
            print('the spins are generated from the makeAtoms() method.')
            print('Please call that method if you have not already.')

    def makeFF(self):
        """Generate the magnetic form factor.
        """
        g = self.gS+self.gL
        if getFFparams(self.ffparamkey) != ['none']:
            self.ff = (self.gS/g * jCalc(self.ffqgrid, getFFparams(self.ffparamkey))+
                       self.gL/g * jCalc(self.ffqgrid, getFFparams(self.ffparamkey), j2=True))
        else:
            print('Using generic magnetic form factor.')
            self.ff = jCalc(self.ffqgrid)

    def spinsFromAtoms(self,positions,fractional=True,returnIdxs=False):
        """Return the spin vectors corresponding to specified atomic
           positions.

        This method calls the diffpy.mpdf.magutils.spinsFromAtoms() method.

        Args:
            magstruc: MagSpecies or MagStructure object containing atoms and spins
            positions (list or array): atomic positions for which the
                corresponding spins should be returned.
            fractional (boolean): set as True if the atomic positions are in
                fractional coordinates of the crystallographic lattice
                vectors.
            returnIdxs (boolean): if True, the indices of the spins will also be
                returned.
        Returns:
            Array consisting of the spins corresponding to the atomic positions.
        """
        return spinsFromAtoms(self,positions,fractional,returnIdxs)

    def atomsFromSpins(self,spinvecs,fractional=True,returnIdxs=False):
        """Return the atomic positions corresponding to specified spins.

        This method calls the diffpy.mpdf.magutils.atomsFromSpins() method.

        Args:
            magstruc: MagSpecies or MagStructure object containing atoms and spins
            spinvecs (list or array): spin vectors for which the
                corresponding atoms should be returned.
            fractional (boolean): set as True if the atomic positions are to be
                returned as fractional coordinates of the crystallographic lattice
                vectors.
            returnIdxs (boolean): if True, the indices of the atoms will also be
                returned.

        Returns:
            List of arrays of atoms corresponding to the spins.
        """
        return atomsFromSpins(self,spinvecs,fractional,returnIdxs)

    def findAtomIndices(self,atomList):
        """Return list of indices corresponding to input list of atomic coordinates.

        This method calls the diffpy.mpdf.findAtomIndices() method. 

        Args:
            atomList (numpy array of atomic coordinates)

        Returns:
            List of indices corresponding to the atomList.
        """
        return findAtomIndices(self,atomList)

    def runChecks(self):
        """Run some simple checks and raise a warning if a problem is found.
        """
        if self.verbose:        
            print(('Running checks for '+self.label+' MagSpecies object...\n'))

        flagCount = 0
        flag = False

        if self.useDiffpyStruc:
            # check that basisvecs and kvecs have same shape
            if self.kvecs.shape != self.basisvecs.shape:
                flag = True
            if flag:
                flagCount += 1
                if self.verbose:
                    print('kvecs and basisvecs must have the same dimensions.')

        else:
            # check for improperlatVecs array
            if self.latVecs.shape != (3, 3):
                flag = True
            if flag:
                flagCount += 1
                if self.verbose:
                    print('latVecs array does not have the correct dimensions.')
                    print('It must be a 3 x 3 nested array.')
                    print('Example: np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])')
            flag = False

            # check for mismatched number of atoms and spins in basis
            if self.atomBasis.shape != self.spinBasis.shape:
                flag = True
            if flag:
                flagCount += 1
                if self.verbose:
                    print('atomBasis and spinBasis must have the same dimensions.')

        # summarize results
        if flagCount == 0:
            if self.verbose:
                print('All MagSpecies() checks passed. No obvious problems found.\n')

    def copy(self):
        """Return a deep copy of the MagSpecies object.
        """
        return copy.deepcopy(self)

class MagStructure:
    """Build on the diffpy.Structure class to include magnetic attributes.

    This class takes a diffpy.Structure object and packages additional info
    relating to magnetic structure, which can then be fed to an MPDFcalculator
    object.

    Args:
        struc (diffpy.Structure object): provides lattice parameters and unit
            cell of desired structure.
        species (python dictionary): dictionary of magnetic species in the
            structure. The values are MagSpecies objects.
        atoms (numpy array): list of atomic coordinates of all the magnetic
            atoms in the structure; e.g. generated by generateAtomsXYZ()
        spins (numpy array): triplets giving the spin vectors of all the
            atoms, in the same order as the atoms array provided as input.
        gfactors (numpy array): Lande g-factors of the magnetic moments
        rmaxAtoms (float): maximum distance from the origin of atomic
            positions generated by the makeAtoms method.
        ffqgrid (numpy array): grid of momentum transfer values used for
            calculating the magnetic form factor.
        ff (numpy array): magnetic form factor. Should be same shape as
            ffqgrid.
        label (string): Optional descriptive string for the MagStructure.
        K1 (float): a constant used for calculating Dr; should be averaged
            over all magnetic species. Important if physical information is
            to be extracted from mPDF scale factors, e.g. moment size.
        K2 (float): another constant used for calculating Dr.
        fractions (python dictionary): Dictionary providing the fraction of
            spins in the magnetic structure corresponding to each species.
        verbose (boolean): If True, will print messages relating to the structure.
            Useful for troubleshooting. Default is False.
        calcIdxs (python list): list giving the indices of the atoms array
            specifying the atoms to be used as the origin when calculating
            the mPDF. If given the string argument 'all', then every atom
            will be used (potentially causing very long calculation times).
        corrLength (scalar): magnetic correlation length such that the
            magnitude of the correlation between two spins separated by a
            distance d is given by exp(-d/corrLength). If set to zero, the
            correlation length is assumed to be infinite.
        rho0 (float): number of magnetic moments per cubic Angstrom in the
            magnetic structure; default value is 0.
        netMag (float): net magnetization in Bohr magnetons per magnetic moment
            in the sample; default is 0. Only nonzero for ferro/ferrimagnets or
            canted antiferromagnets.

   """

    def __init__(self, struc=None, species=None, atoms=None, spins=None,
                 gfactors=None, rmaxAtoms=30.0, ffqgrid=None, ff=None,
                 label='', K1=None, K2=None, fractions=None, Uiso=0.01,
                 calcIdxs=None, corrLength=0.0, verbose=False,
                 netMag=0, rho0=0):

        self.rmaxAtoms = rmaxAtoms
        self.label = label

        if struc is None:
            self.struc = []
        else:
            self.struc = struc
        if atoms is None:
            self.atoms = np.array([])
        else:
            self.atoms = atoms
        if spins is None:
            self.spins = np.array([])
        else:
            self.spins = spins
        if gfactors is None:
            self.gfactors = np.array([2.0])
        else:
            self.gfactors = gfactors
        if species is None:
            self.species = {}
        else:
            self.species = species
        if ffqgrid is None:
            self.ffqgrid = np.arange(0, 10.0, 0.01)
        else:
            self.ffqgrid = ffqgrid
        if ff is None:
            self.ff = jCalc(self.ffqgrid)
        else:
            self.ff = ff
        if K1 is None:
            self.K1 = 0.66667*(1.913*2.81794/2.0)**2*2.0**2*0.5*(0.5+1)
        else:
            self.K1 = K1
        if K2 is None:
            self.K2 = self.K1
        else:
            self.K2 = K2
        if fractions is None:
            self.fractions = {}
        else:
            self.fractions = fractions
        if calcIdxs is None:
            self.calcIdxs = [0]
        else:
            self.calcIdxs = calcIdxs
        self.Uiso = Uiso
        self.corrLength = corrLength        
        self.verbose = verbose
        self.rho0 = rho0
        self.netMag = netMag

    def __repr__(self):
        if self.label == '':
            return 'MagStructure() object'
        else:
            return self.label+': MagStructure() object'

    def makeSpecies(self, label, strucIdxs=None, atoms=None, spins=None,
                    basisvecs=None, kvecs=None, S=0.5, L=0.0, J=None, gS=None,
                    gL=None, ffparamkey=None,ffqgrid=None, ff=None):
        """Create a MagSpecies object and add it to the species dictionary.

        Args:
            label (string): label for this particular magnetic species. Should be
                different from the labels for any other magnetic species you make.
            strucIdxs (python list): list of integers giving indices of magnetic
                atoms in the unit cell
            atoms (numpy array): list of atomic coordinates of all the magnetic
                atoms in the structure; e.g. generated by generateAtomsXYZ()
            spins (numpy array): triplets giving the spin vectors of all the
                atoms, in the same order as the atoms array provided as input.
            basisvecs (numpy array): nested three-vector(s) giving the basis
                vectors to generate the spins. e.g. np.array([[0, 0, 1]]). Any
                phase factor should be included directly with the basisvecs.
            kvecs (numpy array): nested three-vector(s) giving the propagation
                vectors for the magnetic structure in r.l.u.,
                e.g. np.array([[0.5, 0.5, 0.5]])
            gS (float): spin component of the Lande g-factor (g = gS+gL)
            gL (float): orbital component of the Lande g-factor
            ffparamkey (string): gives the appropriate key for getFFparams()
            ffqgrid (numpy array): grid of momentum transfer values used for
                calculating the magnetic form factor.
            ff (numpy array): magnetic form factor.

        """
        # check that the label is not a duplicate with any other mag species.
        duplicate = False
        for name in list(self.species.keys()):
            if name == label:
                duplicate = True
        if not duplicate:
            if ffqgrid is None:
                ffqgrid = np.arange(0, 10.0, 0.01)
            self.species[label] = MagSpecies(self.struc, label, strucIdxs, atoms, spins,
                                             self.rmaxAtoms, basisvecs, kvecs, S, L,
                                             J, gS, gL, ffparamkey, ffqgrid, ff,
                                             self.verbose)
            # update the list of fractions
            totatoms = 0.0
            for key in self.species:
                totatoms += self.species[key].atoms.shape[0]
            for key in self.species:
                if totatoms == 0.0:
                    totatoms = 1.0 # prevent divide by zero problems
                frac = float(self.species[key].atoms.shape[0])/totatoms
                self.fractions[key] = frac
            self.runChecks()
        else:
            print('This label has already been assigned to another species in')
            print('the structure. Please choose a new label.')

    def getCoordsFromSpecies(self):
        """Read in atomic positions and spins from magnetic species.

        This differs from makeSpins() and makeAtoms() because it simply loads
        the atoms and spins from the species without re-generating them from 
        the structure.
        """
        tempA = np.array([[0, 0, 0]])
        tempS = np.array([[0, 0, 0]])
        for key in self.species:
            na = self.species[key].atoms.shape[0]
            ns = self.species[key].atoms.shape[0]
            if (na > 0) and (na == ns):            
                tempA = np.concatenate((tempA, self.species[key].atoms))
                tempS = np.concatenate((tempS, self.species[key].spins))
            else:
                if self.verbose:
                    print(('Coordinates of atoms and spins for ' + key))
                    print('have not been loaded because they have not yet been')
                    print('generated and/or do not match in shape.')
        if tempA.shape != (1, 3):        
            self.atoms = tempA[1:]
            self.spins = tempS[1:]
        elif len(self.species) == 0:
            self.atoms = np.array([])
            self.spins = np.array([])

    def loadSpecies(self, magSpec):
        """Load in an already-existing MagSpecies object

        Args:
            magSpec (MagSpecies object): The magnetic species to be imported
                into the structure.
        """
        # check that the label is not a duplicate with any other mag species.
        duplicate = False
        for name in list(self.species.keys()):
            if name == magSpec.label:
                duplicate = True
        if not duplicate:
            self.species[magSpec.label] = magSpec
            self.struc = magSpec.struc
            self.getCoordsFromSpecies()
            # update the list of fractions
            totatoms = 0.0
            for key in self.species:
                totatoms += self.species[key].atoms.shape[0]
            for key in list(self.species.keys()):
                if totatoms == 0.0:
                    totatoms = 1.0 # prevent divide by zero problems
                frac = float(self.species[key].atoms.shape[0])/totatoms
                self.fractions[key] = frac
            self.runChecks()
        else:
            print('The label for this species has already been assigned to')
            print('another species in the structure. Please choose a new label')
            print('for this species.')

    def removeSpecies(self, label, update=True):
        """Remove a magnetic species from the species dictionary.

        Args:
            label (string): key for the dictionary entry to be removed.
            update (boolean): if True, the MagStructure will update its atoms
                and spins with the removed species now excluded.
        """
        try:
            del self.species[label]
            if update:
                self.getCoordsFromSpecies()
                # update the list of fractions
                totatoms = 0.0
                for key in self.species:
                    totatoms += self.species[key].atoms.shape[0]
                for key in self.species:                
                    if totatoms == 0.0:
                        totatoms = 1.0 # prevent divide by zero problems
                    frac = float(self.species[key].atoms.shape[0])/totatoms
                    self.fractions[key] = frac
        except:
            print('Species cannot be deleted. Check that you are using the')
            print('correct species label.')

    def makeAtoms(self):
        """Generate the Cartesian coordinates of the atoms for this species.

        Args:
            fromUnitCell (boolean): True if atoms/spins to be generated from
                a unit cell provided by the user; False if the diffpy structure
                object is to be used.
            unitcell (numpy array): Provides the unit cell lattice vectors as
                np.array((avec, bvec, cvec)).
            atombasis (numpy array): Provides positions of the magnetic atoms
                in fractional coordinates within the unit cell.
            spin cell (numpy array): Provides the orientations of the spins in
                the unit cell, in the same order as atombasis
        """
        temp = np.array([[0, 0, 0]])
        for key in self.species:
            self.species[key].makeAtoms()
            temp = np.concatenate((temp, self.species[key].atoms))
        self.atoms = temp[1:]

    def makeSpins(self):
        """Generate the Cartesian coordinates of the spin vectors in the
               structure. Calls the makeSpins() method for each MagSpecies in
               the species dictionary and concatenates them together.
        """
        temp = np.array([[0, 0, 0]])
        for key in self.species:
            self.species[key].makeSpins()
            temp = np.concatenate((temp, self.species[key].spins))
        self.spins = temp[1:]

    def makeGfactors(self):
        """Generate an array of Lande g-factors in the same order as the spins
                in the MagStructure.
        """
        temp = np.array([2.0])
        for key in self.species:
            temp = np.concatenate((temp,
                                   (self.species[key].gS+self.species[key].gL)*np.ones(self.species[key].spins.shape[0])))
        self.gfactors = temp[1:]

    def makeFractions(self):
        """Generate the fractions dictionary.
        """
        try:
            totatoms = 0.0
            for key in self.species:
                totatoms += self.species[key].atoms.shape[0]
            for key in self.species:                
                if totatoms == 0.0:
                    totatoms = 1.0 # prevent divide by zero problems
                frac = float(self.species[key].atoms.shape[0])/totatoms
                self.fractions[key] = frac
        except:
            if len(self.species) == 0:
                self.fractions = {}
            else:
                print('Check MagStructure.fractions dictionary for problems.')

    def makeKfactors(self):
        """Set the factors K1 and K2 used for unnormalized mPDF. The fractions
           dictionary must be accurate before running this method.
        """
        K1, K2 = 0, 0        
        for key in self.species:
            gSa, gLa = self.species[key].gS, self.species[key].gL
            ga = gSa + gLa
            Ja = self.species[key].J
            K1 += self.fractions[key]*ga*np.sqrt(Ja*(Ja+1))
            K2 += self.fractions[key]*ga**2*Ja*(Ja+1)
        K1 = K1**2
        K1 *= (1.913*2.81794/2.0)**2*2.0/3.0
        K2 *= (1.913*2.81794/2.0)**2*2.0/3.0
        self.K1 = K1
        self.K2 = K2

    def makeFF(self):
        """Generate the properly weighted average magnetic form factor of all
                the magnetic species in the structure.
        """
        try:
            self.ffqgrid = list(self.species.values())[0].ffqgrid
            self.ff = np.zeros_like(self.ffqgrid)
            totatoms = 0.0
            for key in self.species:
                totatoms += self.species[key].atoms.shape[0]
            for key in self.species:
                frac = float(self.species[key].atoms.shape[0])/totatoms
                self.species[key].makeFF()
                self.ff += frac*self.species[key].ff
        except:
            if len(self.species) == 0:
                self.ff = jCalc(self.ffqgrid)
            else:
                print('Check that all mag species have same q-grid.')

    def makeAll(self):
        """Shortcut method to generate atoms, spins, g-factors, and form
                factor for the magnetic structure all in one go.
        """
        self.makeAtoms()
        self.makeSpins()
        self.makeGfactors()
        self.makeFractions()
        self.makeKfactors()
        self.makeFF()
        self.makeCalcIdxs()
        self.runChecks()

    def spinsFromAtoms(self,positions,fractional=True,returnIdxs=False):
        """Return the spin vectors corresponding to specified atomic
           positions.

        This method calls the diffpy.mpdf.spinsFromAtoms() method. 

        Args:
            magstruc: MagSpecies or MagStructure object containing atoms and spins
            positions (list or array): atomic positions for which the
                corresponding spins should be returned.
            fractional (boolean): set as True if the atomic positions are in
                fractional coordinates of the crystallographic lattice
                vectors.
            returnIdxs (boolean): if True, the indices of the spins will also be
                returned.
        Returns:
            Array consisting of the spins corresponding to the atomic positions.
        """
        return spinsFromAtoms(self,positions,fractional,returnIdxs)

    def atomsFromSpins(self,spinvecs,fractional=True,returnIdxs=False):
        """Return the atomic positions corresponding to specified spins.

        This method calls the diffpy.mpdf.atomsFromSpins() method. 

        Args:
            magstruc: MagSpecies or MagStructure object containing atoms and spins
            spinvecs (list or array): spin vectors for which the
                corresponding atoms should be returned.
            fractional (boolean): set as True if the atomic positions are to be
                returned as fractional coordinates of the crystallographic lattice
                vectors.
            returnIdxs (boolean): if True, the indices of the atoms will also be
                returned.

        Returns:
            List of arrays of atoms corresponding to the spins.
        """
        return atomsFromSpins(self,spinvecs,fractional,returnIdxs)

    def visualize(self,atoms,spins,showcrystalaxes=False,
                  axesorigin=np.array([0,0,0])):
        """Generate a crude 3-d plot to visualize the selected spins.

        Args:
            atoms (numpy array): array of atomic positions of spins to be
                visualized.
            spins (numpy array): array of spin vectors in same order as atoms.
            showcrystalaxes (boolean): if True, will display the crystal axes
                determined from the first magnetic species in the MagStructure
            axesorigin (array): position at which the crystal axes should be
                displayed
        """
        import matplotlib.pyplot as plt        
        from mpl_toolkits.mplot3d import axes3d

        fig = visualizeSpins(atoms,spins)
        if showcrystalaxes:
            ax3d = fig.axes[0]
            try:
                mspec=list(self.species.items())[0][1]
                if mspec.useDiffpyStruc:
                    lat=mspec.struc.lattice
                    a, b, c = lat.stdbase
                else:
                    a, b, c = mspec.latVecs
                xo, yo, zo = axesorigin
                ax3d.quiver(xo, yo, zo, a[0], a[1], a[2], pivot='tail', color='r')
                ax3d.quiver(xo, yo, zo, b[0], b[1], b[2], pivot='tail', color='g')
                ax3d.quiver(xo, yo, zo, c[0], c[1], c[2], pivot='tail', color='b')
            except:
                print('Please make sure your magnetic structure contains a')
                print('magnetic species with MagSpecies.struc set to a diffpy')
                print('structure or MagSpecies.latVecs provided and')
                print('MagSpecies.useDiffpyStruc set to False.')
        plt.show()

    def findAtomIndices(self,atomList):
        """Return list of indices corresponding to input list of atomic coordinates.

        This method calls the diffpy.mpdf.findAtomIndices() method. 

        Args:
            atomList (numpy array of atomic coordinates)

        Returns:
            List of indices corresponding to the atomList.
        """
        return findAtomIndices(self,atomList)

    def runChecks(self):
        """Run some simple checks and raise a warning if a problem is found.
        """
        # do the MagSpecies checks
        for key in self.species:
            self.species[key].runChecks()

        if self.verbose:        
            print(('Running checks for '+self.label+' MagStructure object...\n'))

        flag = False
        flagCount = 0

        # check for duplication among magnetic species
        if len(self.species) > 0:        
            if list(self.species.values())[0].useDiffpyStruc:
                idxs = []
                for key in self.species:
                    idxs.append(self.species[key].strucIdxs)
                idxs = [item for sublist in idxs for item in sublist] # flatten the list
                for idx in idxs:
                    if idxs.count(idx) > 1:
                        flag = True
                if flag:
                    flagCount += 1
                    if self.verbose:
                        print('Warning: Magnetic species may have overlapping atoms.')
                        print('Check the strucIdxs lists for your magnetic species.')
                    flag = False

        # check that the fractions are consistent
        totatoms = 0.0
        for key in self.species:
            totatoms += self.species[key].atoms.shape[0]
        for key in self.species:
            if totatoms == 0.0:
                totatoms = 1.0 # prevent divide by zero problems
            frac = float(self.species[key].atoms.shape[0])/totatoms
            if (frac > 0) and (np.abs(frac - self.fractions[key])/frac > 0.1):
                flag = True
        if flag:
            flagCount += 1
            if self.verbose:
                print('Species fractions do not correspond to actual number of')
                print('spins of each species in the structure.')
        flag = False

        ### check if calcIdxs may not be representative of all MagSpecies.
        if len(self.calcIdxs) < len(self.species):
            flag = True
        if flag:
            flagCount += 1
            print('Warning: your calcIdxs may not be representative of all')
            print('the magnetic species. calcIdxs should have the index of')
            print('at least one spin from each species. Use')
            print('magStruc.getSpeciesIdxs() to see starting indices for')
            print('each species.\n')
        flag = False

        ### check if calcIdxs has indices that exceed the spin array
        if self.atoms.shape[0]>0:
            if (np.array(self.calcIdxs).max()+1) > self.atoms.shape[0]:
                flag = True
        if flag:
            flagCount += 1
            print('calcIdxs contains indices that are too large for the')
            print('arrays of atoms and spins contained in the MagStructure.')
        flag = False

        # summarize results
        if flagCount == 0:
            if self.verbose:
                print('All MagStructure checks passed. No obvious problems found.')

    def getSpeciesIdxs(self):
        """Return a dictionary with the starting index in the atoms and spins
           arrays corresponding to each magnetic species.
        """
        idxDict = {}
        startIdx = 0
        for key in self.species:
            idxDict[key] = startIdx
            startIdx += self.species[key].atoms.shape[0]
        return idxDict

    def makeCalcIdxs(self):
        """Generate the indices of the atoms to be used for the calculation.
        """
        idxDict = self.getSpeciesIdxs()
        calcIdxs = []
        for key in self.species:
            calcIdxs.append(np.array(self.species[key].calcIdxs) +
                            idxDict[key])
        calcIdxs = [ci for sublist in calcIdxs for ci in sublist]
        self.calcIdxs = calcIdxs

    def calcAtomicDensity(self, volume=0, numSpins=0):
        """Determine the number density of magnetic moments.
        Sets the calculated number density equal to self.rho0.

        Args:
            volume (scalar): Volume of the MagStructure. If equal to the
                default value of 0, then the volume will be calculated
                assuming a sphere of radius rmaxAtoms.
            numSpins (integer): number of magnetic moments in the volume
                being considered. If equal to the default value of 0, then
                the numSpins will be set to the length of the spins array.
        """
        if volume==0:
            radius = self.rmaxAtoms + \
                     np.linalg.norm(self.struc.lattice.stdbase.sum(axis=1))
            volume = 1.33333*np.pi*radius**3
        if numSpins==0:
            numSpins = len(self.spins)
        self.rho0 = numSpins / volume

    def copy(self):
        """Return a deep copy of the MagStructure object."""
        return copy.deepcopy(self)


