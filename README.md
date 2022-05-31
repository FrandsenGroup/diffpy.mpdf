## diffpy.mpdf

Framework for computing and fitting magnetic PDFs.

This package aims to provide a convenient method for computing the magnetic PDF (mPDF) from magnetic structures and performing fits to neutron total scattering data. The mPDF is calculated by an mPDFcalculator object, which extracts the spin positions and spin vectors from a magStructure object that the mPDFcalculator takes as input. The magStructure object in turn can contain multiple magSpecies objects, which generate magnetic configurations based on a diffpy.Structure object and a set of propagation vectors and basis vectors provided by the user. Alternatively, the user can manually define a magnetic unit cell that will be used to generate the magnetic structure, or the magnetic structure can be defined simply as lists of spin positions and spin vectors provided by the user.


## Requirements

This package requires Python 3.5 or greater and the following software:

numpy, matplotlib, scipy, diffpy.Structure, diffpy.srreal

Recommended software:

Full diffpy-cmi suite.

Current supported platforms are Linux (64- and 32-bit) and MacOS (64-bit).

## Installation

The recommended way to install this package is first to install diffpy-cmi through conda, then install diffpy.mpdf from source.

#### Step 1 (recommended): Create and activate a conda environment for diffpy + diffpy.mpdf.
    >>> conda create --name diffpy python=3
    >>> conda activate diffpy
Note that you can name the environment anything you choose by passing it a different name after the --name flag in the first command.

#### Step 2: Install diffpy-cmi through conda.
    >>> conda install -c diffpy diffpy-cmi
Make sure you are installing this in the environment you created in the previous step.

#### Step 3: Install diffpy.mpdf
Clone or download this repository on your local machine. Navigate to the downloaded repository and run the following command (making sure that you are in the environment you created in Step 1).

    >>> python setup.py install

## Documentation and Helpful Examples
Complete documentation is available in the doc folder of this repository.

Several examples to help you get started with mPDF analysis are available as jupyter notebooks at https://github.com/FrandsenGroup/mPDF-tutorial.

## Contributors

Benjamin Frandsen, Parker Hamilton, Jacob Christensen, Eric Stubben, Victor Velasco, Pavol Juhas, Xiaohao Yang, and Simon Billinge.

## License

3-Clause BSD License
