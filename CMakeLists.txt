# this file is meant to be parsed by python before being a proper CMakeLists.txt file
# in particular, the following strings will be set to their proper values by python when
# making CMakeLists.txt
set(numpy_include /home/praharsh/anaconda3/envs/changebranch3/lib/python3.8/site-packages/numpy/core/include)
set(PYTHON_INCLUDE_DIRS /home/praharsh/anaconda3/envs/changebranch3/include/python3.8 /home/praharsh/anaconda3/envs/changebranch3/include/python3.8)
set(PYTHON_LDFLAGS "-L/home/praharsh/anaconda3/envs/changebranch3/lib/python3.8/config-3.8-x86_64-linux-gnu -lpthread -ldl -lutil -lrt -lm -lpython3.8 -Xlinker -export-dynamic")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#find_package(PythonLibs)
#message("PYTHON_LIBRARIES ${PYTHON_LIBRARIES}")
#message("PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS}")

# osx makes the suffix for shared object libraries .dylib
IF(APPLE)
  SET(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
ENDIF(APPLE)

message("python include dirs: ${PYTHON_INCLUDE_DIRS}")
message("python ldflags: ${PYTHON_LDFLAGS}")
message("numpy include dirs: ${numpy_include}")
include_directories(${numpy_include})
include_directories(${PYTHON_INCLUDE_DIRS})
# note: We set CMAKE_SHARED_LINKER_FLAGS because I couldn't get it to work using 
# target_link_librarires().  CMake kept interpreting the flag CoreFoundation
# as a library and appending -l to it and the compiler would complain
unset(CMAKE_SHARED_LINKER_FLAGS CACHE)
set(CMAKE_SHARED_LINKER_FLAGS ${PYTHON_LDFLAGS} CACHE type string)

cmake_minimum_required(VERSION 2.8)


enable_language(CXX)
SET(CMAKE_CXX_FLAGS "-std=c++1z -Wall -Wextra -pedantic -O3 -fPIC -march=native -flto -fopenmp")

#cmake_policy(SET CMP0015 NEW)

# set the pele include directory
set(pele_include ${CMAKE_CURRENT_SOURCE_DIR}/source)
message("pele include directory: ${pele_include}")

# build the pele library
# NOTE: GLOBBING MIGHT BE A BAD IDEA WHEN ADDING FILES https://stackoverflow.com/questions/1027247/is-it-better-to-specify-source-files-with-glob-or-each-file-individually-in-cmake
# This is a likely source of undefined symbol errors
include_directories(${pele_include})
FILE(GLOB pele_sources ${pele_include}/*.c*)
add_library(pele_lib SHARED ${pele_sources})

function(make_cython_lib cython_cxx_source)
  get_filename_component(library_name ${cython_cxx_source} NAME)
  string(REGEX REPLACE ".cxx$" "" library_name ${library_name})
  add_library(${library_name} SHARED ${cython_cxx_source})
  target_link_libraries(${library_name} pele_lib)
  set_target_properties(${library_name} PROPERTIES PREFIX "")
  message("making library ${library_name} from source ${cython_cxx_source}")
endfunction(make_cython_lib)


make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_lj_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_morse_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_frozen_dof.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_hs_wca_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_wca_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_harmonic_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/atlj.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_frenkel.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_inversepower_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_inversepower_stillinger_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_inversepower_stillinger_cut_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_sumgaussianpot_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_pele.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_pspin_spherical_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/optimize/_pele_opt.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/optimize/_gradient_descent_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/optimize/_lbfgs_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/optimize/_modified_fire_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/potentials/_pythonpotential.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/angleaxis/_cpp_aa.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/utils/_cpp_utils.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/utils/_pressure_tensor.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/rates/_ngt_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/distance/_get_distance_cpp.cxx)
make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/pele/distance/_put_in_box_cpp.cxx)
