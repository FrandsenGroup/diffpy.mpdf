#!/usr/bin/env python
##############################################################################
#
# diffpy.mpdf         by Frandsen Group
#                     Benjamin A. Frandsen benfrandsen@byu.edu
#                     (c) 2022 Benjamin Allen Frandsen
#                      All rights reserved
#
# File coded by:    Eric Stubben and Benjamin Frandsen
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""functions for creating magnetic structures from MCIF files."""

import re

from cmath import exp

from diffpy.mpdf import *
from diffpy.structure.atom import Atom
from diffpy.structure.lattice import Lattice
from diffpy.structure.structure import Structure

from math import floor, ceil

import numpy as np
from numpy import pi, sin, cos, sqrt, dot
from numpy.linalg import det, inv, norm

from diffpy.mpdf.simpleparser import SimpleParser


def create_from_mcif(mcif, ffparamkey=None, rmaxAtoms=20, S=0.5, L=0.0,
                     J=None, gS=None, gL=None, g=None, j2type=None, occ=None):
    """
    Creates a MagStructure object from an MCIF file.
    
    Note: The only acceptable incommensurate structures at the moment are 1-k
    structures with no     Fourier harmonics (i.e. only coefficients for
    n = 1) and no atoms with nonzero average magnetic moment.

    Args:
        mcif (string): path to MCIF file for desired magnetic structure.
        ffparamkey (string): optional; gives the appropriate key for getFFparams()
            to generate the correct magnetic form factor.
        rmaxAtoms (float): radius to which magnetic atoms should be generated.
            Default is 20 Angstroms.
        S (float): Spin angular momentum quantum number in units of hbar.
        L (float): Orbital angular momentum quantum number in units of hbar.
        J (float): Total angular momentum quantum number in units of hbar.
        gS (float): spin component of the Lande g-factor (g = gS+gL).
            Calculated automatically from S, L, and J if not specified.
        gL (float): orbital component of the Lande g-factor. Calculated
            automatically from S, L, and J if not specified.
        g (float): Lande g-factor. Calculated automatically as gS+gL if not
            specified.
        j2type (string): Specifies the way the j2 integral is included in the
            magnetic form factor. Must be either 'RE' for rare earth or 'TM' for
            transition metal. If 'RE', the coefficient on the j2 integral is
            gL/g; if 'TM', the coefficient is (g-2)/g. Default is 'RE'. Note
            that for the default values of S, L, and J, we will have gS=2,
            gL=0, and g=2, and there will be no difference in the form factor
            calculation for 'RE' or 'TM'.

    Returns:
        MagStructure object corresponding to the magnetic structure
            encoded in the MCIF file. Note that the position and spin arrays
            are not automatically populated when using this function, so
            MagStructure.makeAll() will likely need to be called afterward.
            Also note that variables such as ffparamkey, rmaxAtoms, S, L, J,
            etc are applied uniformly to all magnetic atoms in the unit cell,
            so this method cannot create MagSpecies objects for which these
            attributes are different.

    """
    # creates an empty MagStructure
    mstruc = MagStructure()    

    #Flags whether the structure is incommensurate
    incomm = False

    #Extracts the needed information from the mcif file
    tags = {}
    parser = SimpleParser(tags)
    tags = parser.ReadFile(mcif)

    #These are the parameters of the basic or unit magnetic cell
    #'a', 'b', and 'c' are in angstroms, while 'alpha', 'beta', and 'gamma' are in degrees
    a = tags['_cell_length_a'][0]
    b = tags['_cell_length_b'][0]
    c = tags['_cell_length_c'][0]
    alpha = tags['_cell_angle_alpha'][0]
    beta = tags['_cell_angle_beta'][0]
    gamma = tags['_cell_angle_gamma'][0]
    transform = tags['_parent_space_group.child_transform_Pp_abc'][0]

    #symbols = tags['_atom_site_type_symbol']
    symbols = tags['_atom_site_label']
    symbols0 = 1*symbols # starting list of atom symbols
    try:
        occs = tags['_atom_site_occupancy']
    except KeyError:
        occs = [1.0] * len(symbols)
    occs0 = 1*occs # starting list of occupancies
    scaled_pos = np.array([tags['_atom_site_fract_x'],
                           tags['_atom_site_fract_y'],
                           tags['_atom_site_fract_z']]).T
    scaled_pos = np.mod(scaled_pos,1.)

    mag_mom = np.zeros_like(scaled_pos)
    mag_idx = []

    if '_cell_wave_vector_x' in tags:
        #This is the branch for incommensurate structures
        incomm = True
        k = np.matrix.flatten(np.array([tags['_cell_wave_vector_x'],
                                        tags['_cell_wave_vector_y'],
                                        tags['_cell_wave_vector_z']]).T)

        symm_cent = tags['_space_group_symop_magn_ssg_centering.algebraic']
        symm_ops = tags['_space_group_symop_magn_ssg_operation.algebraic']

        #For incommensurate structures, mag_mom is the average magnetic moment of an atom. Currently,
        #this variable is unused for incommensurate structures
        four_cos = np.zeros_like(scaled_pos)
        four_sin = np.zeros_like(scaled_pos)
        for i, label in enumerate(tags['_atom_site_label']):
            if label in tags['_atom_site_moment.label']:
                mag_idx.append(i)
                mi = tags['_atom_site_moment.label'].index(label)
                mag_mom[i] = [tags['_atom_site_moment.crystalaxis_x'][mi],
                              tags['_atom_site_moment.crystalaxis_y'][mi],
                              tags['_atom_site_moment.crystalaxis_z'][mi]]

                four_cos[i] = [tags['_atom_site_moment_Fourier_param.cos'][3*mi],
                               tags['_atom_site_moment_Fourier_param.cos'][3*mi + 1],
                               tags['_atom_site_moment_Fourier_param.cos'][3*mi + 2]]

                four_sin[i] = [tags['_atom_site_moment_Fourier_param.sin'][3*mi],
                               tags['_atom_site_moment_Fourier_param.sin'][3*mi + 1],
                               tags['_atom_site_moment_Fourier_param.sin'][3*mi + 2]]
    else:
        #This is the branch for commensurate structures
        k = np.array([0,0,0])

        symm_cent = tags['_space_group_symop_magn_centering.xyz']
        symm_ops = tags['_space_group_symop_magn_operation.xyz']

        for i, label in enumerate(tags['_atom_site_label']):
            if label in tags['_atom_site_moment.label']:
                mag_idx.append(i)
                mi = tags['_atom_site_moment.label'].index(label)
                mag_mom[i] = [tags['_atom_site_moment.crystalaxis_x'][mi],
                              tags['_atom_site_moment.crystalaxis_y'][mi],
                              tags['_atom_site_moment.crystalaxis_z'][mi]]

    #Takes the magnetic information to reduced lattice coordinates
    LL = np.diag([1./a,1./b,1./c])
    mag_mom = dot(mag_mom,LL)
    if incomm:
        four_cos = dot(four_cos,LL)
        four_sin = dot(four_sin,LL)

    #Performs the symmetry operations to populate the rest of the unit cell
    all_pos = np.copy(scaled_pos)
    all_mom = np.copy(mag_mom)
    if incomm:
        all_cos = np.copy(four_cos)
        all_sin = np.copy(four_sin)

    tol = 1e-3
    for j, pos in enumerate(scaled_pos):
        pos_subset = np.array([pos])
        eq_site_counter = 1
        for cent in symm_cent:
            #Applies a centering operation
            if incomm:
                Rc, hc, r1c, tc, t4c, trc = parse_magn_ssg_operation(cent)
            else:
                Rc, tc, trc = parse_magn_operation(cent)
            cent_pos = np.mod(dot(Rc,pos) + tc,1.)

            for op in symm_ops:
                #Applies a symmetry operation
                if incomm:
                    R, h, r1, t, t4, tr = parse_magn_ssg_operation(op)
                else:
                    R, t, tr = parse_magn_operation(op)
                eq_site = dot(R,cent_pos) + t

                #Handles the cases where eq_site has coordinate values extremely close to 0 or 1
                for l, coord in enumerate(eq_site):
                    if abs(1 - coord) < tol:
                        eq_site[l] = 1.
                    if abs(coord) < tol:
                        eq_site[l] = 0.
                eq_site = np.mod(eq_site,1.)
                eq_site = np.round(eq_site, decimals=6)

                #Checks to make sure site hasn't already been occupied
                for l, pp in enumerate(pos_subset):
                    if np.allclose(eq_site,pp,atol=tol): # and the atom labels are the same (fix this)
                        break
                else:
                    #Adds the new site found with symmetry
                    if j in mag_idx:
                        mag_idx.append(len(symbols))
                    symbols.append(symbols0[j]+'.'+str(eq_site_counter))
                    occs.append(occs0[j])
                    eq_site_counter += 1
                    pos_subset = np.append(pos_subset,[eq_site],axis=0)
                    all_pos = np.append(all_pos,[eq_site],axis=0)

                    #Transforms the moment information
                    if incomm:
                        deltc = 2*pi*(t4c + dot(hc,pos))
                        delt = 2*pi*(t4 + dot(h,cent_pos))

                        cent_cos = trc*det(Rc)*(cos(deltc)*dot(Rc,four_cos[j]) - r1c*sin(deltc)*dot(Rc,four_sin[j]))
                        cent_sin = trc*det(Rc)*(sin(deltc)*dot(Rc,four_cos[j]) + r1c*cos(deltc)*dot(Rc,four_sin[j]))

                        new_cos = tr*det(R)*(cos(delt)*dot(R,cent_cos) - r1*sin(delt)*dot(R,cent_sin))
                        new_sin = tr*det(R)*(sin(delt)*dot(R,cent_cos) + r1*cos(delt)*dot(R,cent_sin))

                        all_cos = np.append(all_cos,[new_cos],axis=0)
                        all_sin = np.append(all_sin,[new_sin],axis=0)

                    new_mom = tr*trc*det(R)*det(Rc)*dot(R,dot(Rc,mag_mom[j]))   
                    all_mom = np.append(all_mom,[new_mom],axis=0)

    #Determines the basis vectors of the structure (same as the magnetic moments if commensurate)
    basis_vecs = np.empty(np.shape(all_pos),dtype=complex)
    if incomm:
        basis_vecs = (0.5*np.exp(-2*pi*1j*dot(all_pos,k)))[:,np.newaxis]*(all_cos + 1j*all_sin)
        #for i, pos in enumerate(all_pos):
            #basis_vecs[i] = 0.5*exp(-2*pi*1j*dot(k,pos))*(all_cos[i] + 1j*all_sin[i])
    else:
        basis_vecs = all_mom.astype(complex)
        #for i, pos in enumerate(all_pos):
            #basis_vecs[i] = all_mom[i].astype(complex)

    #Converts the basis vectors to Cartesian coordinates
    lat = Lattice(a,b,c,alpha,beta,gamma)
    basis_vecs = dot(basis_vecs,lat.base)
    all_mom = dot(all_mom,lat.base)

    #Creates a diffpy Structure object
    atoms = []
    for i, pos in enumerate(all_pos):
        atoms.append(Atom(atype=symbols[i],xyz=pos, occupancy=occs[i]))
    astruc = Structure(atoms=atoms)
    astruc.lattice = lat
    for atom in astruc: # remove extra numbers from atom names
        name = atom.element
        newname = re.findall(r'[a-zA-Z]+', name)[0]
        atom.element = newname


    #Creates a separate MagSpecies object for each magnetic atom in the unit cell
    for idx in mag_idx:
        new_mspec = MagSpecies(ffparamkey=ffparamkey, rmaxAtoms=rmaxAtoms,
                               S=S, L=L, J=J, gS=gS, gL=gL, g=g,
                               j2type=j2type)

        new_mspec.struc = astruc
        new_mspec.label = symbols[idx]

        new_mspec.strucIdxs = [idx]
        new_mspec.origin = all_pos[idx]
        new_mspec.occ = occs[idx]

        new_mspec.kvecs = np.array([k])
        new_mspec.basisvecs = np.array([basis_vecs[idx]])
        if incomm:
            new_mspec.kvecs = np.append(new_mspec.kvecs,[-k],axis=0)
            new_mspec.basisvecs = np.append(new_mspec.basisvecs,[np.conjugate(basis_vecs[idx])],axis=0)
        new_mspec.avgmom = all_mom[idx] if incomm else np.array([0, 0, 0])

        mstruc.loadSpecies(new_mspec)

    mstruc.transform = transform
    print('MagStructure creation from mcif file successful.')
    return mstruc


   
def create_atomic_cell(mstruc,transform):
    """
    This method accepts a MagStructure object that has already been loaded with an mcif file
    and a string containing the parent-child basis transformation information. The output will be a new
    diffpy.Structure object loaded with the atomic structure information related to the magnetic structure
    of the original sample.
    """
    #Processes the string with the parent-child transformation info
    R, t = parse_parent_child_transform(transform)
    R_p = inv(R)
    t_p = dot(t,R_p)

    #Transforms the magnetic cell basis to an atomic cell basis per the inverse of the given transform
    mag_cell = mstruc.struc.lattice.base
    atom_cell = dot(R_p,mag_cell)

    #Determines the dimension of the supercell we need to completely contain the atomic cell
    dim  = [[floor(min(R_p[:,0]) - t_p[0]) - 1,ceil(max(R_p[:,0]) - t_p[0]) + 1],
            [floor(min(R_p[:,1]) - t_p[1]) - 1,ceil(max(R_p[:,1]) - t_p[1]) + 1],
            [floor(min(R_p[:,2]) - t_p[2]) - 1,ceil(max(R_p[:,2]) - t_p[2]) + 1]]

    sup_pos, sup_occup, sup_symb = supersize_me(mstruc.struc,dim)

    #Determines atomic unit cell from the newly created magnetic supercell
    scaled_pos = []
    occup = []
    symb = []

    tol = 1e-3
    for l, pos in enumerate(sup_pos):
        #Converts the position to fractional atomic lattice coordinates
        new_pos = dot(pos,R) + t

        #Tidies up coordinates close enough to 0 or 1        
        for i, coord in enumerate(new_pos):
            if abs(1 - coord) < tol:
                new_pos[i] = 1.
            if abs(coord) < tol:
                new_pos[i] = 0.

        #Checks if the atoms sits inside the atomic unit cell
        if (0<=new_pos[0]<1) and (0<=new_pos[1]<1) and (0<=new_pos[2]<1):
            add_pos = True            

            #Also checks for duplicates
            dupes, dupe_idxs = find_dupes(scaled_pos,new_pos,tol)
            if len(dupes) >= 1:
                tot_occup = sup_occup[l]
                for idx in dupe_idxs:
                    tot_occup += occup[idx]
                    if element(symb[idx]) == element(sup_symb[l]):
                        add_pos = False
                if (tot_occup - 1) > tol:
                    add_pos = False
            #for m, pp in enumerate(scaled_pos):
            #    if np.allclose(new_pos,pp,atol=tol):
            #        break
            #else:
            if add_pos:
                scaled_pos.append(new_pos)
                occup.append(sup_occup[l])
                symb.append(sup_symb[l])

    #Creates a new diffpy.Structure object representing the atomic unit cell
    atoms = []
    for l, pos in enumerate(scaled_pos):
        atoms.append(Atom(atype=symb[l],xyz=pos,occupancy=occup[l]))
    new_struc = Structure(atoms=atoms)
    new_struc.lattice = Lattice(base=atom_cell)

    return new_struc


def create_magnetic_cell(astruc,transform):
    """
    Takes a diffpy.Structure object representing the atomic unit cell and a string containing information 
    about the parent to child basis vector transformation. Returns a new diffpy.Structure object
    representing the magnetic unit cell. 

    Note: At the moment, the resulting magnetic structure won't actually have any magnetic moment 
    information; it will just be a list of atoms and their positions in the unit cell. Work still needs to
    be done to figure out how to connect atoms in the magnetic cell with the correct moment information.
    """
    #Processes the string with the parent-child transformation info
    R, t = parse_parent_child_transform(transform)
    R_p = inv(R)
    t_p = dot(t,R_p)
    
    #Transforms the atomic cell basis to a magnetic cell basis using the given transform
    atom_cell = astruc.lattice.base
    mag_cell = dot(R,atom_cell)
    
    #Determines the dimensions of the supercell we want to work with to get the magnetic cell
    dim = [[floor(min(R[:,0]) + t[0]) - 1,ceil(max(R[:,0]) + t[0]) + 1],
           [floor(min(R[:,1]) + t[1]) - 1,ceil(max(R[:,1]) + t[1]) + 1],
           [floor(min(R[:,2]) + t[2]) - 1,ceil(max(R[:,2]) + t[2]) + 1]]
    
    sup_pos, sup_occup, sup_symb = supersize_me(astruc,dim)
    
    #Picks out the magnetic unit cell from the newly created atomic supercell
    scaled_pos = []
    occup = []
    symb = []
    
    tol = 1e-3
    for l, pos in enumerate(sup_pos):
        #Converts the position to fractional magnetic lattice coordinates
        new_pos = dot(pos,R_p) - t_p

        #Tidies up coordinates close enough to 0 or 1        
        for i, coord in enumerate(new_pos):
            if abs(1 - coord) < tol:
                new_pos[i] = 1.
            if abs(coord) < tol:
                new_pos[i] = 0.

        #Checks if the atom sits inside the magnetic unit cell
        if (0<=new_pos[0]<1) and (0<=new_pos[1]<1) and (0<=new_pos[2]<1):
            add_pos = True

            #Also checks for duplicates (which may be allowed based on occupancy)
            dupes, dupe_idxs = find_dupes(scaled_pos,new_pos,tol)
            if len(dupes) >= 1:
                tot_occup = sup_occup[l]
                for idx in dupe_idxs:
                    tot_occup += occup[idx]
                    if element(symb[idx]) == element(sup_symb[l]):
                        add_pos = False
                if (tot_occup - 1) > tol:
                    add_pos = False
            #for m, pp in enumerate(scaled_pos):
            #    if np.allclose(new_pos,pp,atol=tol):
            #        break
            #else:
            if add_pos:
                scaled_pos.append(new_pos)
                occup.append(sup_occup[l])
                symb.append(sup_symb[l])
            
    #Creates a new diffpy.Structure object representing the magnetic unit cell (without the moment information)
    atoms = []
    for l, pos in enumerate(scaled_pos):
        atoms.append(Atom(atype=symb[l],xyz=pos,occupancy=occup[l]))
    new_struc = Structure(atoms=atoms)
    new_struc.lattice = Lattice(base=mag_cell)
    
    return new_struc


def parse_magn_ssg_operation(op):
    """
    Parses a string containing a superspace group symmetry operation in standard
    notation. Returns the rotation and translation part of the associated space
    group operation (R and t), the average structure reciprocal lattice vector (h), 
    superspace phase information (r1 and t4), and the time reversal sign (p).
    """
    r1,r2,r3,r4,p=op.split(',')
    M = np.zeros([4,4])
    t = np.zeros([3])
    h = np.zeros([3])
    
    #Compiles translation information
    t[0] = convert_to_float(r1[-4:]) if '/' in r1 else 0.
    t[1] = convert_to_float(r2[-4:]) if '/' in r2 else 0.
    t[2] = convert_to_float(r3[-4:]) if '/' in r3 else 0.
    t4 = convert_to_float(r4[-4:]) if '/' in r4 else 0.
    
    #Compiles rotation matrix
    for i, line in enumerate([r1,r2,r3,r4]):
        x1=x2=x3=x4=0.
        m = re.search(r'-?\d?x1',line)
        if m:
            xs=m.group()
            x1 = 1. if ('x1' in xs and not '-' in xs) else -1.
            xs = xs.replace('x1','')
            m = re.search(r'\d',xs)
            if m:
                x1 *= float(m.group())

        m = re.search(r'-?\d?x2',line)
        if m:
            ys=m.group()
            x2 = 1. if ('x2' in ys and not '-' in ys) else -1.
            ys = ys.replace('x2','')
            m = re.search(r'\d',ys)
            if m:
                x2 *= float(m.group())
                
        m = re.search(r'-?\d?x3',line)
        if m:
            zs=m.group()
            x3 = 1. if ('x3' in zs and not '-' in zs) else -1.
            zs = zs.replace('x3','')
            m = re.search(r'\d',zs)
            if m:
                x3 *= float(m.group())
                
        m = re.search(r'-?\d?x4',line)
        if m:
            ts = m.group()
            x4 = 1. if ('x4' in ts and not '-' in ts) else -1.
            ts = ts.replace('x4','')
            m = re.search(r'\d',ts)
            if m:
                x4 *= float(m.group())
            
        M[i,0], M[i,1], M[i,2], M[i,3] = [x1,x2,x3,x4]
    
    R = M[:3,:3]
    h = M[3,:3]
    r1 = M[3,3]
        
    return (R,h,r1,t,t4,float(p))
    

#Essentially a copy of muesr's parse_magn_operation_xyz_string function
def parse_magn_operation(op):
    """
    Parses a string containg a space group symmetry operation in standard notation.
    Returns the rotation and translation parts of the operation (R and t) and the 
    time reversal sign (p).
    """
    r1,r2,r3,p=op.split(',')
    R = np.zeros([3,3]) # rotations
    t = np.zeros([3])   # translations
    
    # compile traslation matrix
    t[0] = convert_to_float(r1[-4:]) if '/' in r1 else 0.
    t[1] = convert_to_float(r2[-4:]) if '/' in r2 else 0.
    t[2] = convert_to_float(r3[-4:]) if '/' in r3 else 0.
    
    # compile rotation matrix
    for i, line in enumerate([r1,r2,r3]):
        x=y=z=0.
        m = re.search(r'-?\d?x',line)
        if m:
            xs=m.group()
            x = 1. if ('x' in xs and not '-' in xs) else -1.
            m = re.search(r'\d',xs)
            if m:
                x *= float(m.group())

        m = re.search(r'-?\d?y',line)
        if m:
            ys=m.group()
            y = 1. if ('y' in ys and not '-' in ys) else -1.
            m = re.search(r'\d',ys)
            if m:
                y *= float(m.group())
                
        m = re.search(r'-?\d?z',line)
        if m:
            zs=m.group()
            z = 1. if ('z' in zs and not '-' in zs) else -1.
            m = re.search(r'\d',zs)
            if m:
                z *= float(m.group())                            
            
        R[i,0], R[i,1], R[i,2] = [x,y,z]
        
    return (R,t,float(p))


#This is based on muesr's parse_magn_operation_xyz_string method    
def parse_parent_child_transform(transform):
    """
    Takes in a string containing information about the transformation from the parent
    (paramagnetic) basis to the child (magnetic) basis and returns the "rotation" (not
    necessarily a pure rotation) matrix and translation vector associated with the 
    transformation.
    """
    #Splits up the string into related pieces of information
    rotation, translation = transform.split(';')
    r1, r2, r3 = rotation.split(',')
    t1, t2, t3 = translation.split(',')

    #Initializes the rotation matrix and translation vector
    R = np.zeros((3,3))
    t = np.zeros(3)
    
    #Compiles the translation vector
    t[0] = convert_to_float(t1) if '/' in t1 else t1
    t[1] = convert_to_float(t2) if '/' in t2 else t2
    t[2] = convert_to_float(t3) if '/' in t3 else t3
    
    #Compiles the rotation matrix
    for i, line in enumerate([r1,r2,r3]):
        x = y = z = 0
        
        m = re.search(r'-?\d?/?\d?a',line)
        if m:
            xs = m.group()
            x = 1. if ('a' in xs and not '-' in xs) else -1.
            m = re.search(r'\d/?\d?',xs)
            if m:
                if '/' in m.group():
                    x *= convert_to_float(m.group())
                else:
                    x *= float(m.group())
        
        m = re.search(r'-?\d?/?\d?b',line)
        if m:
            ys = m.group()
            y = 1. if ('b' in ys and not '-' in ys) else -1.
            m = re.search(r'\d/?\d?',ys)
            if m:
                if '/' in m.group():
                    y *= convert_to_float(m.group())
                else:
                    y *= float(m.group())
                    
        m = re.search(r'-?\d?/?\d?c',line)
        if m:
            zs = m.group()
            z = 1. if ('c' in zs and not '-' in zs) else -1.
            m = re.search(r'\d/?\d?',zs)
            if m:
                if '/' in m.group():
                    z *= convert_to_float(m.group())
                else:
                    z *= float(m.group())
        
        R[i,0] = x
        R[i,1] = y
        R[i,2] = z
        
    return (R,t)


#ATTENTION: This method should now work with diffpy, but testing still needs to be done
def supersize_me(struc,dim):
    """
    Accepts a diffpy.Structure object and a list of pairs specifying the beginning and ending
    cell along each crystal axis and returns two arrays containing the fractional coordinates of the 
    atoms in the supercell relative to the original unit cell and their chemical symbols, respectively.
    """
    #Extracts the necessary information from the MagStructure object
    old_pos = struc.xyz
    old_occup = struc.occupancy
    old_symb = []
    for i in range(len(struc.element)):
        old_symb.append(struc.element[i])
    
    #Appends the desired additional cells to the original unit cell
    pos_list = []
    occup_list = []
    symb = []
    
    for k in range(dim[2][0],dim[2][1]):
        for j in range(dim[1][0],dim[1][1]):
            for i in range(dim[0][0],dim[0][1]):
                for l, pos in enumerate(old_pos):
                    pos_list.append(pos + np.array([i,j,k]))
                    occup_list.append(old_occup[l])
                    symb.append(old_symb[l])
    
    pos = np.array(pos_list)
    occup = np.array(occup_list)
    
    return (pos,occup,symb)


def order_by_pos(original,unordered):
    """
    This method accepts two arrays of the same size and outputs a third array also of that size.
    The third array is created by permuting the rows of the 'unordered' array in such a way
    that the point given by the ith row of the output array is closer (in terms of Euclidean distance)
    to the point given by the ith row vector of the 'original' array than to the point given by any
    other row of the 'original' array. The intended application of this method is to recover the
    ordering of an array of atomic positions in a magnetic unit cell after downsizing to an atomic unit
    cell and then upsizing once more.
    
    In this method, we're assuming that each point in the 'unordered' array is closest to exactly one 
    point in the original array. The motivation for this is that the rescaling of the atomic unit cell
    during PDF and mPDF fits should only cause small changes in atomic positions within the unit cell, 
    so no atom should stray far enough from their original position to end up closer to a different
    atom.
    """
    #Checks if the input arrays have the same shape
    if np.shape(original)!=np.shape(unordered):
        print('Error: Input arrays must be of the same size.')
        return

    ordered = np.zeros_like(unordered)
    unordered_copy = np.copy(unordered)

    for i, pos_orig in enumerate(original):
        #For a given point in the 'original' array, finds the closest point in the 'unordered' array
        closest_pos = 0
        closest_dist = 1e6

        for j, pos_unord in enumerate(unordered_copy):
            curr_dist = norm(pos_orig - pos_unord)
            if curr_dist < closest_dist:
                closest_dist = curr_dist
                closest_pos = j

        #Adds the closest point in the 'unordered' array to the 'ordered' array in the correct position
        ordered[i] = unordered_copy[closest_pos]

        #Deletes the position from the 'unordered' array to speed up computation
        unordered_copy = np.delete(unordered_copy,closest_pos,0)

    return ordered


def find_dupes(array,value,tol=1e-3):
    """
    Given an m x n array, an n-vector, and a tolerance, determines which rows of the array match 
    the n-vector within the desired (absolute) tolerance. Returns an array consisting of the 
    matching rows and a list containing the respective indices of the rows as they were in the 
    original array.
    """
    dupes = np.array([])
    dupe_idxs = []
    
    for i, row in enumerate(array):
        if np.allclose(row,value,atol=tol):
            dupes = np.append(dupes,row,axis=0)
            dupe_idxs.append(i)
    
    return dupes, dupe_idxs


def element(site_label):
    """
    Finds and returns the chemical symbol of an element based on its site label. 
    """
    chem_symbol = site_label[0]
    
    if len(site_label) > 1:
        if site_label[1].isalpha():
            chem_symbol += site_label[1]
    
    return chem_symbol


def convert_to_float(frac_str):
    """
    This function takes in a string containing a fraction and returns a float.
    """
    num, denom = frac_str.split('/')
    return (float(num) / float(denom))
    

#This method can probably be removed in favor of diffpy's own cell vector generating method
def cellpar_to_cell(cell_par):
    """
    Takes in a list of cell parameters (three side lengths and three angles are expected) and
    returns an array with the cell lattice vectors as rows. The lattice vectors are output in 
    standard Cartesian coordinates with the a- and b-vectors in the xy-plane and the a-vector
    parallel to the x-axis.
    """
    #Extract the cell parameters from the list
    a, b, c, alpha, beta, gamma = cell_par
    
    #Initializes a-vector parallel to x-axis
    va = a * np.array([1,0.,0.])

    #Constructs b-vector in the xy-plane
    vb = b * np.array([cos(gamma),sin(gamma),0.])

    #Determines resulting c-vector subject to parameter constraints
    cx = cos(beta)
    cy = (cos(alpha) - cos(beta)*cos(gamma))/sin(gamma)
    cz = sqrt(1 - cx**2 - cy**2)
    vc = c * np.array([cx,cy,cz])

    #Makes and returns the cell array
    cell = np.vstack((va,vb,vc))
    return cell
