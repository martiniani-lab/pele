cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_COMPILER /home/praharsh/.local/bin/mpicxx)
set(CMAKE_C_COMPILER /home/praharsh/.local/bin/mpicc)
project(pele_tests)

# if not specified by user, the standard build type is release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release Coverage." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

enable_language(CXX)
ADD_DEFINITIONS(-std=c++1z -g)
# ADD_DEFINITIONS(-Wall)

# optimize debugging experience
# debug builds with max information



# Flags useful for debugging
# 1. -ggdb3: for memory analysis with gdb
# 2. -O0 disable all optimizations
# 3. -fsanitize=address checking memory leaks quickly and uninitialized memory quickly


# for a full list of options  for memory checking and profiling 
# check https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html

# force home directory include for me for LSP
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/home/praharsh/.local/include -O0 -fsanitize=address -ggdb3 -fno-omit-frame-pointer -fno-inline")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/home/praharsh/.local/include -O0 -ggdb3 -fsanitize=address -fno-omit-frame-pointer -fno-inline")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)




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





# set(SUNDIALS_EXTRA_LIBS  -lm /home/praharsh/anaconda3/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib/librt.so CACHE STRING )
# Specify the path to SUNDIALS header files
set(SUNDIALS_INCLUDE_DIR
  /home/praharsh/Dropbox/research/bv-libraries/sundials/SN_EXT/include
  CACHE PATH "Location of SUNDIALS header files")

# Specify the path to SUNDIALS libraries
set(SUNDIALS_LIBRARY_DIR
  /home/praharsh/Dropbox/research/bv-libraries/sundials/SN_EXT/lib
  CACHE PATH "Location of SUNDIALS libraries")








set(PETSC_INCLUDE_DIR
  /home/praharsh/petsc/arch-linux-c-debug/include
  CACHE PATH "Location of PETSC headers"  
  )

set(PETSC_INCLUDE_DIR_2
  /home/praharsh/.local/include
  CACHE PATH "Location of PETSC headers"  
  )

set(MPI_INCLUDE_DIR
  /home/praharsh/.local/include
  CACHE PATH "Location of MPI header files")

set(MPI_LIBRARY_DIR
  /home/praharsh/.local/include
  CACHE PATH "Location of MPI LIBRARIES")


# Find the SUNDIALS libraries
find_library(SUNDIALS_SOLVER_LIB
  sundials_cvode ${SUNDIALS_LIBRARY_DIR}
  DOC "CVODE library")

find_library(SUNDIALS_NVEC_LIB
  sundials_nvecserial ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_SERIAL library")

find_library(SUNDIALS_NVEC_PETSC_LIB
  sundials_nvecpetsc ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_PETSC library")

find_library(SUNDIALS_MANYVEC_LIB
  sundials_nvecmanyvector ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_MANYVECTOR library")

FIND_LIBRARY(SUNDIALS_NLS_LIB
  sundials_sunnonlinsolpetscsnes ${SUNDIALS_LIBRARY_DIR}
  DOC "SUNNonlinearSolver SNES library")

set(SUNDIALS_LIBRARIES
  -L${SUNDIALS_LIBRARY_DIR}
  ${SUNDIALS_SOLVER_LIB}
  ${SUNDIALS_NVEC_LIB}
  ${SUNDIALS_MANYVEC_LIB}
  ${SUNDIALS_EXTRA_LIBS}
  ${SUNDIALS_NLS_LIB}
  ${SUNDIALS_NVEC_PETSC_LIB})

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
include_directories(${PETSC_INCLUDE_DIR_2})
include_directories(${SUNDIALS_INCLUDE_DIR})
include_directories(${pele_include})
include_directories(${PETSC_INCLUDE_DIR})
include_directories(SYSTEM ${MPI_INCLUDE_DIR})
message("pele include directory ${pele_include}")
message("sundials include directory ${SUNDIALS_INCLUDE_DIR}")

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
