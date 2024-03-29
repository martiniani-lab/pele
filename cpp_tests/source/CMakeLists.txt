
cmake_minimum_required(VERSION 3.5)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(pele_tests)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CXX)

# TODO: Add a build type option specifically for memory leak testing
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

# include LAPACK AND BLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# set OpenMP
find_package(OpenMP)
message("${OpenMP_C_FLAGS}")
message("${OpenMP_CXX_FLAGS}")

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3 -O0")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb3 -O0")
  if(NOT APPLE)
    # Apple does not support libsanitizer
    # See https://github.com/orgs/Homebrew/discussions/3384#discussioncomment-6263471
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -static-libasan")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -static-libasan")
  endif(NOT APPLE)
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

set(EXT_INSTALL_PATH "${CMAKE_SOURCE_DIR}/../../extern/install")

set(LAPACKE_INCLUDE_DIR ${EXT_INSTALL_PATH}/include/Eigen/src/misc)

# set(SUNDIALS_EXTRA_LIBS  -lm /home/praharsh/anaconda3/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib/librt.so CACHE STRING "Additional libraries")
set(SUNDIALS_INCLUDE_DIR
  ${EXT_INSTALL_PATH}/include
  CACHE PATH "Location of SUNDIALS header files")
set(SUNDIALS_LIBRARY_DIR
  ${EXT_INSTALL_PATH}/lib
  CACHE PATH "Location of SUNDIALS libraries")

message("SUNDIALS_INCLUDE_DIR ${SUNDIALS_INCLUDE_DIR}")

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

# Add and compile the gtest library
set(gtest_src ../gtest)
set(gtest_binary_dir googletest)
set(gtest_include ${gtest_src}/include/)
include_directories(${gtest_include})
message("gtest include directory ${gtest_include}")
add_subdirectory(../gtest ${gtest_binary_dir})
set(gtest_libs gtest gtest_main)
message("gtest_libs ${gtest_libs}")

include_directories(${PYTHON_INCLUDE_DIR})

# set the pele include directory
set(pele_include ../../source)
include_directories(BEFORE SYSTEM ${LAPACKE_INCLUDE_DIR})
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
if(APPLE)
  # Apple changed the ld linker in Xcode 15 which breaks pele.
  # The classic linker can be chosen with -ld_classic flag.
  # However, on older Xcode versions, this flag throws an error.
  # Thus, we simply check here whether this flag exists.
  # See https://www.scivision.dev/xcode-ld_classic/
  include(CheckLinkerFlag)
  check_linker_flag(C "-ld_classic" CLASSIC_FLAG)
  if(CLASSIC_FLAG)
    target_link_options(test_main PUBLIC "LINKER:-ld_classic")
  endif(CLASSIC_FLAG)
endif(APPLE)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
  include(CodeCoverage)
  setup_target_for_coverage(${PROJECT_NAME}_coverage test_main coverage)
endif()