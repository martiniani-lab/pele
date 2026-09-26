pele : Python Energy Landscape Explorer
+++++++++++++++++++++++++++++++++++++++

.. image:: https://github.com/martiniani-lab/pele/actions/workflows/test.yml/badge.svg?branch=master
   :target: https://github.com/martiniani-lab/pele/actions/workflows/test.yml
   :alt: Build Status

.. image:: https://codecov.io/gh/martiniani-lab/pele/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/martiniani-lab/pele
   :alt: Coverage Status

Tools for global optimization, attractor finding and energy landscape exploration.

Source code: https://github.com/martiniani-lab/pele



.. figure:: lj38_gmin_dgraph.png

  Images: The global minimum energy structure of a 38 atom Lennard-Jones cluster.  On
  the right is a disconnectivity graph showing a visualization of the energy
  landscape.  The competing low energy basins are shown in color.

pele started as a python partial-rewriting of GMIN, OPTIM, and PATHSAMPLE: fortran
programs written by David Wales of Cambridge University and collaborators
(http://www-wales.ch.cam.ac.uk/software.html). The version started here https://github.com/pele-python/pele (documentation: http://pele-python.github.io/pele/)

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

Installation
============
We recommend creating a conda environment to work with the package

::

  $ conda create -n pele -c conda-forge python compilers sundials eigen blas-devel
  $ conda activate pele
  $ pip install git+https://github.com/martiniani-lab/pele

Python 3.11 or newer is required; 3.11 to 3.14 are tested on Linux and macOS.

If the machine already has gcc, g++ and gfortran (e.g. :code:`sudo apt install gcc g++ gfortran`),
leave out :code:`compilers` for a much smaller environment. 

Optional: :code:`scikit-sparse` (sparse Cholesky for rate calculations) and
:code:`pymol-open-source` (viewing structures). The GUI (:code:`pele.gui`) still uses
PyQt4, which is not available for current Python versions.

Development
-----------

From a clone, in the same environment::

  $ pip install .                  # install, or
  $ python setup.py build_ext -i   # build in place; then put the clone on PYTHONPATH

Build options are environment variables (or flags to :code:`setup.py`, e.g. :code:`-j 8`):
:code:`PELE_BUILD_TYPE=Debug`, :code:`PELE_WITH_CVODE=0` (no CVODE / attractor
identification; some tests will fail), :code:`PELE_JOBS=8`, and :code:`PELE_NATIVE=0`
(no :code:`-march=native`, for binaries that run on other machines).

SUNDIALS must be built in double precision (the build checks this). Instead of conda's
SUNDIALS and Eigen you can build them from the submodules; :code:`extern/install` is then
preferred over the environment::

  $ git submodule update --init --recursive
  $ cd extern && ./sun_inst.sh Release && cp -r eigen/Eigen install/include/ && cd ..

A :code:`CPATH`/:code:`PYTHONPATH` pointing at a pele clone takes precedence over the
installed package. If a build fails, remove cached files before trying again::

  $ rm -rf build cythonize.dat CMakeLists.txt

Tests
=====

The project uses GitHub Actions for continuous integration (CI) testing on both Linux and macOS.
The badges at the top of this README show the current build status and code coverage.

The C++ tests use GoogleTest. To run the tests, after running `git submodule update --init --recursive` to get the GoogleTest submodule if you haven't already, run::

  $ cd cpp_tests/source
  $ cmake -DCMAKE_BUILD_TYPE=Debug .
  $ make -j8
  $ ./test_main

On MacOs, use the same commands but make sure that cmake finds
the correct GNU compilers and the OpenBLAS library::

  $ cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_PREFIX_PATH=$(brew --prefix openblas) .

To run the Python tests on an installed pele::

  $ pip install pytest
  $ OMP_NUM_THREADS=1 pytest --pyargs pele

or :code:`pytest pele/` from a clone with an in-place build. For coverage reporting (as in CI),
add :code:`--cov=pele --cov-report=term-missing`.
