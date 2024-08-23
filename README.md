## diffpy.mpdf

Framework for comprehensive magnetic PDF analysis.

This package aims to provide a convenient method for computing the magnetic PDF (mPDF) from magnetic structures, performing fits to neutron total scattering data, and generating the experimental mPDF signal from magnetic scattering data. The mPDF is calculated by an MPDFcalculator object, which extracts the spin positions and spin vectors from a MagStructure object that the MPDFcalculator takes as input. The MagStructure object in turn can contain multiple MagSpecies objects, which generate magnetic configurations based on a diffpy.Structure object and a set of propagation vectors and basis vectors either provided by the user or read in directly from an MCIF file. Alternatively, the user can manually define a magnetic unit cell that will be used to generate the magnetic structure, or the magnetic structure can be defined simply as lists of spin positions and spin vectors provided by the user. The MPDFtransformer class is used to generate mPDF data from magnetic scattering data.

Please cite: Frandsen _et al._, "diffpy.mpdf: open-source software for magnetic pair distribution function analysis", _J. Appl. Cryst._ (2022) __55__, 1377-1382. https://doi.org/10.1107/S1600576722007257


## Requirements

This package requires Python 3.7 and the following software:

numpy, matplotlib, scipy, diffpy.Structure, diffpy.srreal

Recommended software:

Full diffpy-cmi suite.

Current supported platforms are Linux (64- and 32-bit) and MacOS (64-bit). With some effort, it may also be possible to run the program on Windows using the Linux Subsystem available for Windows 10. Perhaps a better option if you have a Windows PC is to install a virtual machine with a Linux distribution. Here's a useful guide on how to do that for Ubuntu using VirtualBox: https://ubuntu.com/tutorials/how-to-run-ubuntu-desktop-on-a-virtual-machine-using-virtualbox#1-overview . 

## Installation

The recommended way to install this package is first to install diffpy-cmi through conda using the Anaconda python distribution, then install diffpy.mpdf using pip. See https://www.anaconda.com/distribution for instructions about installing the Anaconda python distribution. See also https://www.diffpy.org/products/diffpycmi/index.html for the diffpy-cmi installation instructions (reproduced here for convenience).

#### Step 1 (recommended): Create and activate a conda environment for diffpy + diffpy.mpdf.
    >>> conda create --name diffpy python=3.7
    >>> conda activate diffpy
Note that you can name the environment anything you choose by passing it a different name after the --name flag in the first command.

#### Step 2: Install diffpy-cmi through conda.
    >>> conda install -c diffpy diffpy-cmi
Make sure you are installing this in the environment you created in the previous step.

#### Step 3: Install diffpy.mpdf with pip
Making sure you have activated your diffpy environment, run the command:

    >>> pip install diffpy.mpdf

Alternatively, you can install from source by cloning or downloading the github repository https://github.com/FrandsenGroup/diffpy.mpdf on your local machine, navigating to the downloaded repository, and running the following command (making sure that you are in the environment you created in Step 1):

    >>> python setup.py install

## Documentation and Helpful Examples
Complete documentation is available  at [https://frandsengroup.github.io/diffpy.mpdf/index.html](https://frandsengroup.github.io/diffpy.mpdf/index.html).

Several examples to help you get started with mPDF analysis are available as jupyter notebooks at [https://github.com/FrandsenGroup/mPDF-tutorial](https://github.com/FrandsenGroup/mPDF-tutorial).

You may also check out [https://addie.ornl.gov/simulating_mpdf](https://addie.ornl.gov/simulating_mpdf) for a web-based tool to calculate mPDF patterns from magnetic CIF (mCIF) files. 

## Contributors

Benjamin Frandsen, Parker Hamilton, Jacob Christensen, Eric Stubben, Victor Velasco, Pavol Juhas, Xiaohao Yang, and Simon Billinge.

## License

3-Clause BSD License
