cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_COMPILER /opt/intel/compilers_and_libraries_2020.1.217/linux/mpi/intel64/bin/mpigxx)
set(CMAKE_C_COMPILER /opt/intel/compilers_and_libraries_2020.1.217/linux/mpi/intel64/bin/mpigcc)
project(pele_tests)

# if not specified by user, the standard build type is release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release Coverage." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

enable_language(CXX)
ADD_DEFINITIONS(-std=c++1z -g)
ADD_DEFINITIONS(-Wall)


#set OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
get_filename_component(TEST_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)


# get petsc/any other dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ../../cmake)
find_package(PETSc REQUIRED)
#cmake_policy(SET CMP0015 NEW)



# set(SUNDIALS_EXTRA_LIBS  -lm /home/praharsh/anaconda3/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib/librt.so CACHE STRING "Additional libraries")
set(SUNDIALS_INCLUDE_DIR
  /home/praharsh/.local/include
  CACHE PATH "Location of SUNDIALS header files")
set(SUNDIALS_LIBRARY_DIR
  /home/praharsh/.local/lib
  CACHE PATH "Location of SUNDIALS libraries")

set(MPI_INCLUDE_DIR
  /opt/intel/compilers_and_libraries_2020.1.217/linux/mpi/intel64/include
  CACHE PATH "Location of MPI header files")

set(MPI_LIBRARY_DIR
  /opt/intel/compilers_and_libraries_2020.1.217/linux/mpi/intel64/bin
  CACHE PATH "Location of MPI LIBRARIES")
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
set(MPI_LIBRARIES
  -L${MPI_LIBRARY_DIR})
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


# set the pele include directory
set(pele_include ../../source)
include_directories(${pele_include})
include_directories(${SUNDIALS_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_DIR})
message("pele include directory ${pele_include}")

# build the pele library
FILE(GLOB pele_sources ${pele_include}/*.c*)
add_library(pele_lib ${pele_sources})

# get all the source files
FILE(GLOB sources *.c*)

add_executable(test_main ${sources})
target_link_libraries(test_main pele_lib gtest gtest_main pthread)
target_link_libraries(test_main ${SUNDIALS_LIBRARIES})
target_link_libraries(test_main ${MPI_LIBRARIES})
target_link_libraries(test_main PkgConfig::PETSC)



add_subdirectory(benchmarks)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    include(CodeCoverage)
    setup_target_for_coverage(${PROJECT_NAME}_coverage test_main coverage)
    target_link_libraries(test_main gcov)
    target_link_libraries(pele_lib gcov)
  endif()
