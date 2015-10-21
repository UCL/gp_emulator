cmake_minimum_required (VERSION 2.8)

cuda_add_library(_gpu_predict  _gpu_predict.cpp 
                 predict.cu
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
subdirs(tests)
