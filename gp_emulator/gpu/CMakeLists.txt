cmake_minimum_required (VERSION 2.8)



find_package(CUDA QUIET REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonInterp REQUIRED)


find_library(FOUND_CUBLAS cublas)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIR})


if (NOT NUMPY_INCLUDE_DIR)
    exec_program ("${PYTHON_EXECUTABLE}"
      ARGS "-c 'import numpy; print numpy.get_include()'"
      OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
      RETURN_VALUE NUMPY_NOT_FOUND)
    if (NUMPY_INCLUDE_DIR MATCHES "Traceback")
    # Did not successfully include numpy
      set(NUMPY_FOUND FALSE)
    else (NUMPY_INCLUDE_DIR MATCHES "Traceback")
    # successful
      set (NUMPY_FOUND TRUE)
      set (NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR} CACHE PATH "Numpy include path")
    endif (NUMPY_INCLUDE_DIR MATCHES "Traceback")
    if (NUMPY_FOUND)
      if (NOT NUMPY_FIND_QUIETLY)
        message (STATUS "Numpy headers found")
      endif (NOT NUMPY_FIND_QUIETLY)
    else (NUMPY_FOUND)
      if (NUMPY_FIND_REQUIRED)
        message (FATAL_ERROR "Numpy headers missing")
      endif (NUMPY_FIND_REQUIRED)
    endif (NUMPY_FOUND)
    mark_as_advanced (NUMPY_INCLUDE_DIR)
endif (NOT NUMPY_INCLUDE_DIR)

set(CUBLAS_TARGET_LINK ${FOUND_CUBLAS})

cuda_add_library(python_predict _gpu_predict.c predict.cu computeTranspose.c MODULE)
cuda_add_cublas_to_target(python_predict)
target_link_libraries(python_predict python)
