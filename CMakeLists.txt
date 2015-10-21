cmake_minimum_required (VERSION 2.8)

include(cmake/LookUp-GreatCMakeCookOff.cmake)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/tests)
string(ASCII 27 Esc)

option(double "Enable testing" off)

if(double)
  add_definitions(-DDOUBLE__PRECISION)
endif()


set(ColourReset "${Esc}[m")
set(Red  "${Esc}[1;31m")
message("${Red}export PYTHONPATH=$PYTHONPATH:${LIBRARY_OUTPUT_PATH} ${ColourReset}")

find_package(Numpy REQUIRED)
find_package(CoherentPython REQUIRED)
find_package(CUDA QUIET REQUIRED)

find_library(FOUND_CUBLAS cublas)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS}/numpy)

set(CUBLAS_TARGET_LINK ${FOUND_CUBLAS})

subdirs(gp_emulator/gpu)
