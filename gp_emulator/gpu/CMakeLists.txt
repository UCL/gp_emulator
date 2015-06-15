cmake_minimum_required (VERSION 2.8)


find_package(Numpy REQUIRED)
find_package(CoherentPython REQUIRED)
find_package(CUDA QUIET REQUIRED)

find_library(FOUND_CUBLAS cublas)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS}/numpy)


set(CUBLAS_TARGET_LINK ${FOUND_CUBLAS})

cuda_add_library(_gpu_predict  _gpu_predict.c 
                 predict.cu  
                 computeTranspose.cu 
                 kernel_vectorTimesMatrix.cu 
                 kernel_init_array.cu
                 kernel_cdist.cu MODULE)
set_target_properties(_gpu_predict  PROPERTIES PREFIX "")
cuda_add_cublas_to_target(_gpu_predict)
target_link_libraries(_gpu_predict ${PYTHON_LIBRARIES})
subdirs(tests)
