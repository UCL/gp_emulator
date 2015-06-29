cmake_minimum_required (VERSION 2.8)

find_package(CBLAS REQUIRED)
find_package(Numpy REQUIRED)
find_package(CoherentPython REQUIRED)
find_package(CUDA QUIET REQUIRED)


find_library(FOUND_BLAS blas)
find_library(FOUND_CUBLAS cublas)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS}/numpy)
include_directories(${BLAS_INCLUDE_DIR})

set(CUBLAS_TARGET_LINK ${FOUND_CUBLAS})

cuda_add_library(_gpu_predict  _gpu_predict.cpp 
                 predict.cu
                 pure_c.cpp
                 computeTranspose.cu 
                 kernel_vectorTimesMatrix.cu 
                 kernel_init_array.cu
                 kernel_cdist.cu 
                 kernel_matrixExp.cu
                 kernel_elementwiseMult.cu 
                 kernel_scalarMinusVec.cu
                 kernel_rowSum.cu 
                 kernel_crossMinus.cu MODULE)
set_target_properties(_gpu_predict  PROPERTIES PREFIX "")
cuda_add_cublas_to_target(_gpu_predict)
target_link_libraries(_gpu_predict ${PYTHON_LIBRARIES})
target_link_libraries(_gpu_predict ${BLAS_LIBRARIES})
subdirs(tests)
