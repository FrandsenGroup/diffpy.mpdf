## diffpy.mpdf

Framework for computing and fitting magnetic PDFs.

This package aims to provide a convenient method for computing the magnetic PDF (mPDF) from magnetic structures and performing fits to neutron total scattering data. The mPDF is calculated by an mPDFcalculator object, which extracts the spin positions and spin vectors from a magStructure object that the mPDFcalculator takes as input. The magStructure object in turn can contain multiple magSpecies objects, which generate magnetic configurations based on a diffpy.Structure object and a set of propagation vectors and basis vectors provided by the user. Alternatively, the user can manually define a magnetic unit cell that will be used to generate the magnetic structure, or the magnetic structure can be defined simply as lists of spin positions and spin vectors provided by the user.


## Requirements

This package requires Python 2.6 or 2.7 and the following software:

numpy, matplotlib, scipy, diffpy.Structure, diffpy.srreal

Recommended software:

Full diffpy-cmi suite.

Current supported platforms are Linux (64- and 32-bit) and MacOS (64-bit).

## Installation

The recommended way to install this package is through conda. For help installing conda, please visit https://www.anaconda.com/download.  Once conda is installed, you can follow the simple steps below to install diffpy.mpdf. You will also install the full diffpy-cmi suite along the way.

#### Step 1: Add the appropriate conda channels to your conda configuration.
    >>> conda config --add channels diffpy
    >>> conda config --add channels benfrandsen
These commands tell conda to look through these channels on the Anaconda cloud server to find the appropriate software.

#### Step 2 (recommended): Create and activate a conda environment for diffpy + diffpy.mpdf.
    >>> conda create --name diffpy python=2.7
    >>> source activate diffpy
Note that you can name the environment anything you choose by passing it a different name after the --name flag in the first command.

#### Step 3: Install diffpy.mpdf
    >>> conda install diffpy.mpdf
Note that this will also install the full diffpy-cmi suite if it has not already been installed in this environment.

#### Alternative option: Install from the python package index
If you prefer to use the python package index and have pip installed, please install the prerequisite software and then use the command:

    >>> pip install diffpy.mpdf

#### Alternative option: Download and install from source
If you prefer to install from source, you can also zip and download this entire github repository (see the Download Zip button in the upper right corner), then unzip the file and execute the following command from inside the unzipped directory:

    >>> python setup.py install
Note that this will only work if all the dependencies are already properly installed.

## Documentation
Please see http://pythonhosted.org/diffpy.mpdf/ for the documentation.

## Contributors

Benjamin Frandsen, Pavol Juhas, Xiaohao Yang, and Simon Billinge.

## License

Modified BSD License.
