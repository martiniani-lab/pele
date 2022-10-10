set(CMAKE_CXX_COMPILER "g++-12")
set(CMAKE_C_COMPILER "gcc-12")
cmake_minimum_required(VERSION 2.8)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(pele_tests)

# # if not specified by user, the standard build type is release
# if(NOT CMAKE_BUILD_TYPE)
# set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release Coverage." FORCE)
# endif(NOT CMAKE_BUILD_TYPE)


enable_language(CXX)
#TODO: Add a build type option specifically for memory leak testing
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3 -O0")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb3 -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG")
elseif(CMAKE_BUILD_TYPE STREQUAL Coverage)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fprofile-arcs -ftest-coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG -fprofile-arcs -ftest-coverage")
endif()
message(CMAKE_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES})

# find conda path for python header
set(PYTHON_INCLUDE_DIR $ENV{CONDA_PREFIX}/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
message("PYTHON INCLUDE DIR" ${PYTHON_INCLUDE_DIR})
# set OpenMP
find_package(OpenMP)
message("${OpenMP_C_FLAGS}")
message("${OpenMP_CXX_FLAGS}")

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# include LAPACK AND BLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(HOOMD REQUIRED)

# set(SUNDIALS_EXTRA_LIBS  -lm /home/praharsh/anaconda3/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib/librt.so CACHE STRING "Additional libraries")
set(SUNDIALS_INCLUDE_DIR
  /home/praharsh/research/bv-libraries/sunnnew/sundials/SN_EXT/include
  CACHE PATH "Location of SUNDIALS header files")
set(SUNDIALS_LIBRARY_DIR
  /home/praharsh/research/bv-libraries/sunnnew/sundials/SN_EXT/lib
  CACHE PATH "Location of SUNDIALS libraries")

# KLU include directory and libraries
set(KLU_INCLUDE_DIR
  /home/praharsh/research/bv-libraries/SuiteSparse/include
  CACHE PATH "Location of KLU header files")

set(KLU_LIBRARIES
  /home/praharsh/research/bv-libraries/SuiteSparse/lib/libklu.so;/home/praharsh/research/bv-libraries/SuiteSparse/lib/libamd.so;/home/praharsh/research/bv-libraries/SuiteSparse/lib/libcolamd.so;/home/praharsh/research/bv-libraries/SuiteSparse/lib/libbtf.so;/home/praharsh/research/bv-libraries/SuiteSparse/lib/libsuitesparseconfig.so
  CACHE STRING "KLU libraries")

# Find KLU SUNLinsol library
find_library(SUNDIALS_SUNLINSOLKLU_LIB
  sundials_sunlinsolklu ${SUNDIALS_LIBRARY_DIR}
  DOC "SUNDIALS KLU linear solver library")

# Find the SUNDIALS libraries
find_library(SUNDIALS_SOLVER_LIB
  sundials_cvode ${SUNDIALS_LIBRARY_DIR}
  DOC "CVODE library")

find_library(SUNDIALS_NVEC_LIB
  sundials_nvecserial ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_SERIAL library")

find_library(SUNDIALS_MANYVEC_LIB
  sundials_nvecmanyvector ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_MANYVECTOR library")

set(SUNDIALS_LIBRARIES
  -L${SUNDIALS_LIBRARY_DIR}
  ${SUNDIALS_SOLVER_LIB}
  ${SUNDIALS_NVEC_LIB}
  ${SUNDIALS_MANYVEC_LIB}
  ${SUNDIALS_EXTRA_LIBS}
  ${SUNDIALS_SUNLINSOLKLU_LIB})

message("SUNDIALS KLU LIB ${SUNDIALS_SUNLINSOLKLU_LIB}")

# Add and compile the gtest library
set(gtest_src ../gtest)
set(gtest_binary_dir googletest)
set(gtest_include ${gtest_src}/include/)
include_directories(${gtest_include})
message("gtest include directory ${gtest_include}")
add_subdirectory(../gtest ${gtest_binary_dir})
set(gtest_libs gtest gtest_main)
message("gtest_libs ${gtest_libs}")

# HOOMD includes
get_property(HOOMD_INCLUDE_DIRECTORY TARGET HOOMD::_hoomd PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
get_property(PYBIND11_INCLUDE_DIRS TARGET pybind11::headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

# get_property(PYBIND11_LIBRARIES TARGET pybind11::headers PROPERTY INTERFACE_LINK_LIBRARIES)
# # HOOMD link flags
# get_property(HOOMD_LIBRARIES TARGET HOOMD::_hoomd PROPERTY INTERFACE_LINK_LIBRARIES)
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${HOOMD_INCLUDE_DIRECTORY})
include_directories(${PYBIND11_INCLUDE_DIRS})

# set the pele include directory
set(pele_include ../../source)
include_directories(${pele_include})
include_directories(${SUNDIALS_INCLUDE_DIR})
include_directories(${KLU_INCLUDE_DIR})

message("pele include directory ${pele_include}")

# build the pele library
FILE(GLOB pele_sources ${pele_include}/*.c*)
add_library(pele_lib ${pele_sources})

# get all the source files
FILE(GLOB sources *.c*)

add_executable(test_main ${sources})

# pybind11_add_module(test_main ${sources})
target_link_libraries(test_main pele_lib gtest gtest_main pthread)
target_link_libraries(test_main ${SUNDIALS_LIBRARIES})
target_link_libraries(test_main ${BLAS_LIBRARIES})
target_link_libraries(test_main ${LAPACK_LIBRARIES})
target_link_libraries(test_main ${KLU_LIBRARIES})

# link to python libraries as well to make HOOMD references work
# target_link_libraries(test_main PUBLIC HOOMD::_hoomd)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
  include(CodeCoverage)
  setup_target_for_coverage(${PROJECT_NAME}_coverage test_main coverage)
endif()