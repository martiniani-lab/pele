# Script to install sundials for pele
# first argument is the build type (debug or release)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
c_d=$SCRIPT_DIR/install

cd sundials
mkdir build
cd build


# note the newlines are important

cmake -DCMAKE_INSTALL_PREFIX=$c_d \
      -DCMAKE_BUILD_TYPE=$1 \
      -DBUILD_ARKODE=OFF \
      -DBUILD_CVODES=OFF \
      -DBUILD_IDA=OFF \
      -DBUILD_IDAS=OFF \
      -DBUILD_KINSOL=OFF \
      -DENABLE_OPENMP=OFF  \
      -DBUILD_STATIC_LIBS=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_CXX_STANDARD=17 \
      -DSUNDIALS_INDEX_SIZE=32 \
      ..
make install
cd ..
rm -rf build
