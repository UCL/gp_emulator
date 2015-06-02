FIND_LIBRARY(CUNIT_LIBRARY NAMES cunit libcunit cunitlib)
FIND_PATH(CUNIT_INCLUDE_DIR NAMES CUnit/CUnit.h)
MARK_AS_ADVANCED(CUNIT_INCLUDE_DIR)
 
FIND_LIBRARY(CUNIT_LIBRARY NAMES 
    cunit
    libcunit
    cunitlib
)
MARK_AS_ADVANCED(CUNIT_LIBRARY)
 
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CUnit DEFAULT_MSG CUNIT_LIBRARY CUNIT_INCLUDE_DIR)
 
IF(CUNIT_FOUND)
  SET(CUNIT_LIBRARIES ${CUNIT_LIBRARY})
  SET(CUNIT_INCLUDE_DIRS ${CUNIT_INCLUDE_DIR})
ENDIF(CUNIT_FOUND)


INCLUDE_DIRECTORIES(${CUNIT_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(gpu_predict_test test1.cu readTestData.cu     testCdist.cu        testCublasgemm.cu   
                    testMatrixExp.cu    testPredict.cu      testVecTimesMat.cu
                    ../predict.cu ../computeTranspose.cu)
cuda_add_cublas_to_target(gpu_predict_test)
target_link_libraries(gpu_predict_test ${CUNIT_LIBRARIES})
