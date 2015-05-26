cmake_minimum_required (VERSION 2.8)


find_package(Numpy REQUIRED)
find_package(CoherentPython REQUIRED)
find_package(CUDA QUIET REQUIRED)

find_library(FOUND_CUBLAS cublas)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS}/numpy)


set(CUBLAS_TARGET_LINK ${FOUND_CUBLAS})

cuda_add_library(_gpu_predict _gpu_predict.cu predict.cu computeTranspose.cu MODULE)
SET_TARGET_PROPERTIES(_gpu_predict  PROPERTIES PREFIX "")
cuda_add_cublas_to_target(_gpu_predict)

target_link_libraries(_gpu_predict python)

subdirs(tests)
