set(CMAKE_CXX_COMPILER "g++-11")
set(CMAKE_C_COMPILER "gcc-11")
cmake_minimum_required(VERSION 2.8)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(pele_tests)

# if not specified by user, the standard build type is release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release Coverage." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

enable_language(CXX)
ADD_DEFINITIONS(-std=c++20 -O0 -ggdb3 -fsanitize=address)


#find conda path for python header
set(PYTHON_INCLUDE_DIR $ENV{CONDA_PREFIX}/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
message("PYTHON INCLUDE DIR" ${PYTHON_INCLUDE_DIR})



#set OpenMP
find_package(OpenMP)
message("${OpenMP_C_FLAGS}")
message("${OpenMP_CXX_FLAGS}")
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# include LAPACK AND BLAS

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(HOOMD REQUIRED)




# set(SUNDIALS_EXTRA_LIBS  -lm /home/praharsh/anaconda3/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib/librt.so CACHE STRING "Additional libraries")
set(SUNDIALS_INCLUDE_DIR
  /home/praharsh/Dropbox/research/bv-libraries/sunnnew/sundials/SN_EXT/include
  CACHE PATH "Location of SUNDIALS header files")
set(SUNDIALS_LIBRARY_DIR
  /home/praharsh/Dropbox/research/bv-libraries/sunnnew/sundials/SN_EXT/lib
  CACHE PATH "Location of SUNDIALS libraries")


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
  ${SUNDIALS_EXTRA_LIBS})
#####################################################################################

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
get_property(PYBIND11_INCLUDE_DIRS TARGET pybind11::pybind11_headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${HOOMD_INCLUDE_DIRECTORY})
include_directories(${PYBIND11_INCLUDE_DIRS})


# set the pele include directory
set(pele_include ../../source)
include_directories(${pele_include})
include_directories(${SUNDIALS_INCLUDE_DIR})
message("pele include directory ${pele_include}")

# build the pele library
FILE(GLOB pele_sources ${pele_include}/*.c*)
add_library(pele_lib ${pele_sources})

# get all the source files
FILE(GLOB sources *.c*)

add_executable(test_main ${sources})
target_link_libraries(test_main pele_lib gtest gtest_main pthread)
target_link_libraries(test_main ${SUNDIALS_LIBRARIES})
target_link_libraries(test_main ${BLAS_LIBRARIES})
target_link_libraries(test_main ${LAPACK_LIBRARIES})


add_subdirectory(benchmarks)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    include(CodeCoverage)
    setup_target_for_coverage(${PROJECT_NAME}_coverage test_main coverage)

    #target_link_libraries(test_main gcov)
    #target_link_libraries(pele_lib gcov)
endif()