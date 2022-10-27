
pele : Python Energy Landscape Explorer
+++++++++++++++++++++++++++++++++++++++

Tools for global optimization, attractor finding and energy landscape exploration.

Source code: https://github.com/pele-python/pele

Documentation: http://pele-python.github.io/pele/



.. figure:: lj38_gmin_dgraph.png

  Images: The global minimum energy structure of a 38 atom Lennard-Jones cluster.  On
  the right is a disconnectivity graph showing a visualization of the energy
  landscape.  The competing low energy basins are shown in color.

pele started as a python partial-rewriting of GMIN, OPTIM, and PATHSAMPLE: fortran
programs written by David Wales of Cambridge University and collaborators
(http://www-wales.ch.cam.ac.uk/software.html).

The current version is being developed by the Martiniani group at New York University.

Description
===========
pele has tools for energy minimization, global optimization, saddle point
(transition state) search, data analysis, visualization and much more.  Some of
the algorithms implemented are:

#. Basinhopping global optimization
#. Potentials (Lennard-Jones, Morse, Hertzian, etc.) 
#. LBFGS minimization (plus other minimizers)
#. Attractor identification (Mixed Descent, CVODE)
#. Single ended saddle point search:
   - Hybrid Eigenvector Following
   - Dimer method
#. Double ended saddle point search
   - Nudged Elastic Band (NEB)
   - Doubly Nudged Elastic Band (DNEB)

#. Disconnectivity Graph visualization

#. Structure alignment algorithms

#. Thermodynamics (e.g. heat capacity) via the Harmonic Superposition Approximation

#. Transition rates analysis

INSTALLATION
============

Required packages
-----------------

for compilation
^^^^^^^^^^^^^^^

#. fortran compiler
#. c compiler (gcc preferably)
#. c++ compiler (g++ preferably)
#. CMake (version 3.5 or higher)

On Ubuntu this can be installed with :code:`sudo apt-get install gfortran gcc g++ cmake`
on older versions of Ubuntu you may need to provide a version number, and set the version used

commands::
     sudo apt install -y gcc-10 g++-10 gfortran-10 cmake # any gcc>5
     sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
     sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
     sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-10 100


C/C++ packages:
^^^^^^^^^^^^^^^^^
#. Eigen (http://eigen.tuxfamily.org)
#. SUNDIALS (https://computing.llnl.gov/projects/sundials)

SUNDIALS and Eigen are automatically downloaded with :code:`git submodule update --init --recursive` (which will also download GoogleTest for C++ tests)
an install script is provided for sundials, Eigen can be installed by running the command :code:`cp -r eigen/Eigen install/include/`

python packages:
^^^^^^^^^^^^^^^^
pele requires python 3.8 or higher, and the following packages

1. numpy: 
     We use numpy everywhere for doing numerical work.  It also installs f2py which
     is used to compile fortran code into modules callable by python.

#. scipy:
     For some of the optimizers and various scientific tools

#. networkx: 
     For graph functionality. https://networkx.lanl.gov

#. cython: 
     For calling C++ code from python for speed

#. matplotlib:
     For making plots (e.g. disconnectivity graphs)

#. SQLAlchemy 0.7: 
     For managing database of stationary points.  http://www.sqlalchemy.org/

#. munkres: 
     For permutational alignment

#. pyro4: 
     For parallel jobs

#. scikits.sparse: optional 
     For use of sparse Cholesky decomposition methods when calculating rates

#. pymol: optional
     For viewing molecular structures


We recommend installing all the above packages in a conda environment.

If you want to use the gui you will additionally need:

1. qt4 and qt4 python bindings

#. opengl python bindings
  

The Ubuntu packages (apt-get) for these are: python-qt4, python-opengl, and python-qt4-gl

In fedora Fedora (yum) you will want the packages: PyQt4, and PyOpenGl


Installing using Conda
----------------------------------
We recommend you set up a new conda environment using :code:`conda create -n myenv python=3.9`

commands::

  $ conda activate myenv
  $ conda install numpy scipy networkx matplotlib cython
  $ conda install -c conda-forge sqlalchemy munkres pyro4 scikit-sparse
  $ conda install -c conda-forge -c schrodinger pymol-bundle
  $ pip install future # used for upgrading to python 3


Compilation
-----------

Compilation is required as many of the computationally intensive parts (especially potentials)
are written in fortran and c++.  Theoretically you should be able to use any fortran compiler,
but we mostly use gfortran and GCC, so it's the least likely to have problems.  This
package uses the standard python setup utility (distutils).  The current installation procedure
is:

  $ python setup_with_cmake.py build_ext -i --fcompiler=gfortran

make sure to add the install directory to your
PYTHONPATH environment variable.

Tests
=====

The C++ tests use GoogleTest. To run the tests, after running :code:`git submodule update --init --recursive` to get the GoogleTest submodule if you haven't already, run::

  $ cd cpp_tests/source
  $ cmake -DCMAKE_BUILD_TYPE=Debug .
  $ make -j8
  $ ./test_main

The python tests have originally been written using nose. But we have transitioned to using pytests. 
To run the tests, run::

  $ pytest pele/

from the base directory.
