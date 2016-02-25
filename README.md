## diffpy.magpdf

Framework for computing and fitting magnetic PDFs.

This package aims to provide a convenient method for computing the magnetic PDF (mPDF) from magnetic structures and performing fits to neutron total scattering data. The mPDF is calculated by an mPDFcalculator object, which extracts the spin positions and spin vectors from a magStructure object that the mPDFcalculator takes as input. The magStructure object in turn can contain multiple magSpecies objects, which generate magnetic configurations based on a diffpy.Structure object and a set of propagation vectors and basis vectors provided by the user. Alternatively, the user can manually define a magnetic unit cell that will be used to generate the magnetic structure, or the magnetic structure can be defined simply as lists of spin positions and spin vectors provided by the user.


## Requirements

This package requires Python 2.6 or 2.7 and the following software:

numpy
matplotlib
diffpy.Structure
diffpy.srreal

Recommended software:

diffpy.srfit

numpy and matplotlib can be installed through standard package managers such as pip or conda. The diffpy packages can be installed together through conda via the command ''conda install -c diffpy diffpy-cmi'', or by following the instructions at http://www.diffpy.org.

## Installation

The easiest way to install this package is to clone this github repository:

git clone https://github.com/benfrandsen/mPDFmodules.git

Then navigate into the newly created mPDFmodules directory and execute the command ''python setup.py install''. You can now call mPDF functions from any standard python script via ''from diffpy.magpdf import *''.


## Contributors

Simon Billinge
Benjamin Frandsen
Pavol Juhas
Xiaohao Yang

## License

GNU General Public License.
