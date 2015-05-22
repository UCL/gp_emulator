cmake_minimum_required (VERSION 2.8)


include(LookUp-GreatCMakeCookOff.cmake)
find_package(Numpy REQUIRED)
find_package(CoherentPython REQUIRED)
find_package(CUDA QUIET REQUIRED)

find_library(FOUND_CUBLAS cublas)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS}/numpy)



set(CUBLAS_TARGET_LINK ${FOUND_CUBLAS})

cuda_add_library(python_predict _gpu_predict.c predict.cu computeTranspose.c MODULE)
cuda_add_cublas_to_target(python_predict)

#target_link_libraries(python_predict python)
