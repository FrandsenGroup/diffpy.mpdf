## diffpy.mpdf

Framework for comprehensive magnetic PDF analysis.

This package aims to provide a convenient method for computing the magnetic PDF (mPDF) from magnetic structures, performing fits to neutron total scattering data, and generating the experimental mPDF signal from magnetic scattering data. The mPDF is calculated by an MPDFcalculator object, which extracts the spin positions and spin vectors from a MagStructure object that the MPDFcalculator takes as input. The MagStructure object in turn can contain multiple MagSpecies objects, which generate magnetic configurations based on a diffpy.Structure object and a set of propagation vectors and basis vectors either provided by the user or read in directly from an MCIF file. Alternatively, the user can manually define a magnetic unit cell that will be used to generate the magnetic structure, or the magnetic structure can be defined simply as lists of spin positions and spin vectors provided by the user. The MPDFtransformer class is used to generate mPDF data from magnetic scattering data. Both one-dimensional (powder) and three-dimensional (single crystal) mPDF patterns can be calculated.

Please cite: Frandsen _et al._, "diffpy.mpdf: open-source software for magnetic pair distribution function analysis", _J. Appl. Cryst._ (2022) __55__, 1377-1382. https://doi.org/10.1107/S1600576722007257


## Requirements

This package requires Python 3.11 or greater and the following software:

numpy, matplotlib, scipy, diffpy.structure, diffpy.srreal

Recommended software:

Full diffpy.cmi suite, jupyter notebook

Current supported platforms are Linux, Unix, macOS, and Windows.

## Recommended Installation Procedure

#### Step 1: Install diffpy.cmi
Follow the instructions at https://github.com/diffpy/diffpy.cmi. 

#### Step 2: Install diffpy.mpdf with pip
Making sure you are in the correct python environment, run the command:

    >>> pip install diffpy.mpdf

Alternatively, you can install from source by installing diffpy.structure and diffpy.srreal following the instructions on their respective github pages, then cloning or downloading the github repository https://github.com/FrandsenGroup/diffpy.mpdf on your local machine, navigating to the downloaded repository, and running the following command (making sure that you are in the correct python environment):

    >>> pip install .

## Documentation and Helpful Examples
Complete documentation is available  at [https://frandsengroup.github.io/diffpy.mpdf/index.html](https://frandsengroup.github.io/diffpy.mpdf/index.html).

Several examples to help you get started with mPDF analysis are available as jupyter notebooks at [https://github.com/FrandsenGroup/mPDF-tutorial](https://github.com/FrandsenGroup/mPDF-tutorial).

You may also check out [https://addie.ornl.gov/simulating_mpdf](https://addie.ornl.gov/simulating_mpdf) for a web-based tool to calculate mPDF patterns from magnetic CIF (mCIF) files. 

## Contributors

Benjamin Frandsen, Parker Hamilton, Jacob Christensen, Eric Stubben, Victor Velasco, Pavol Juhas, Xiaohao Yang, and Simon Billinge.

## License

3-Clause BSD License
