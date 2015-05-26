FIND_PATH(CUNIT_INCLUDE_DIR NAMES CUnit/CUnit.h)
FIND_LIBRARY(CUNIT_LIBRARY NAMES cunit libcunit cunitlib)

CUDA_ADD_EXECUTABLE(gpu_predict_test test1.cu readTestData.cu     testCdist.cu        testCublasgemm.cu   
                    testMatrixExp.cu    testPredict.cu      testVecTimesMat.cu
                    ../predict.cu ../computeTranspose.cu)
cuda_add_cublas_to_target(gpu_predict_test)
target_link_libraries(gpu_predict_test cunit)
