find_library(CUNIT_LIBRARY NAMES cunit libcunit cunitlib)
find_path(CUNIT_INCLUDE_DIR NAMES CUnit/CUnit.h)
#mark_as_advanced(CUNIT_INCLUDE_DIR)
#MARK_AS_ADVANCED(CUNIT_LIBRARY)



include_directories(${CUNIT_INCLUDE_DIR})

cuda_add_executable(gpu_predict_test unit_test.cu 
                    readTestData.cu     
                    compare_result.cu
                    testCdist.cu        
                    testCublasgemm.cu   
                    testMatrixExp.cu    
                    testPredict.cu      
                    testVecTimesMat.cu
                    testInitArray.cu
                    #                    ../predict.cu
                    ../predictcpp.cu
                    ../computeTranspose.cu 
                    ../kernel_vectorTimesMatrix.cu
                    ../kernel_init_array.cu
                    ../kernel_cdist.cu
                    ../kernel_matrixExp.cu
                    ../kernel_elementwiseMult.cu
                    ../kernel_scalarMinusVec.cu
                    ../kernel_rowSum.cu
                    ../kernel_crossMinus.cu)
cuda_add_cublas_to_target(gpu_predict_test)
target_link_libraries(gpu_predict_test ${CUNIT_LIBRARY})

