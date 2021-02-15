# install script for installing sundials. This should help make sure that the right flags are used for sundials to work
# copy the script into the sundials folder and make sure that the location is set right
# to work with pele it helps to make sure both petsc and sundials use 64 bit indices
mkdir build
cd build
# sundials compile options options shell script
# remember to do this for 64 bit integers
# the -lrt flags are because there was an undefined referenet
# to some clock functions
cmake -DCMAKE_INSTALL_PREFIX=/home/praharsh/.local \
      -DENABLE_PETSC=ON \
      -DENABLE_MPI=ON \
      -DMPI_CXX_COMPILER=/home/praharsh/.local/bin/mpic++ \
      -DMPI_C_COMPILER=/home/praharsh/.local/bin/mpicc \
      -DPETSC_DIR=~/.local \
      -DCMAKE_C_FLAGS='-lrt' \
      -DCMAKE_CXX_FLAG='-lrt' \
      ..
make install
