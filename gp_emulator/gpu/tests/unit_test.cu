#include <stdio.h>
#include <string.h>
#include "gpu_predict_test.h"
#include "cuda.h"

real *expX, *expXsqrt, *in_train, *in_predict, *invQ, *invQt;
real *cdist_a, *cdist_expa;
real *cdist_test_var1, *cdist_test_var2, *cdist_test_var3;
real *error_test1;
real *result_t, *error_t, *deriv_t;
int Npredict_t, Ntrain_t, Ninputs_t;
int theta_size_t;

int init_suite1(void)
{
    invQ            =  readTestData( "invQ.bin", Ntrain_t * Ntrain_t);
    invQt           =  readTestData( "invQt.bin", Ntrain_t );
    
    expX            =  readTestData( "expX.bin", Ninputs_t+2);
    expXsqrt        =  readTestData( "expXsqrt.bin", Ninputs_t );
    in_train           =  readTestData( "in_train.bin", Ntrain_t * Ninputs_t );
    in_predict         =  readTestData( "in_predict.bin", Npredict_t * Ninputs_t );
    cdist_test_var1 =  readTestData( "cdist_test_var1.bin", Ntrain_t * Ninputs_t);
    cdist_test_var2 =  readTestData( "cdist_test_var2.bin", Npredict_t * Ninputs_t);
    cdist_a         =  readTestData( "cdist_a.bin", Ntrain_t * Npredict_t);
    cdist_expa      =  readTestData( "cdist_expa.bin", Ntrain_t * Npredict_t ); 
    
    error_test1     =  readTestData( "error_test1.bin", Ntrain_t * Npredict_t );
    result_t          =  readTestData( "result.bin", Ntrain_t * Npredict_t);
    error_t           =  readTestData( "error.bin", Npredict_t);
    deriv_t           =  readTestData( "deriv.bin", Ntrain_t * Npredict_t );
    
    return 0;
}

int clean_suite(void)
{
    free(expXsqrt);
    free(in_train);
    free(in_predict);
    free(cdist_a);
    free(cdist_test_var1);
    free(cdist_test_var2);// have all variable been fully cleaned??
    return 0;
}

void tests_VecTimesMat(void)
{
    testVecTimesMat(expXsqrt, in_train, cdist_test_var1, Ninputs_t, Ntrain_t, Ninputs_t );
    testVecTimesMat(expXsqrt, in_predict, cdist_test_var2, Ninputs_t, Npredict_t, Ninputs_t );
}

void tests_cdist(void)
{
    testCdist(cdist_test_var1,cdist_test_var2, cdist_a, Ntrain_t, Npredict_t, Ninputs_t);
}


void tests_matrixExp(void)
{
    testMatrixExp( cdist_a, cdist_expa, -0.5, expX[Ninputs_t], Ntrain_t * Npredict_t );
}

void tests_cublasgemm(void)
{
   testCublasgemm(invQ, cdist_expa, error_test1, Ntrain_t, Ntrain_t, Ntrain_t, Npredict_t);
}


int main(int argc, char *argv[])
{
    if( argc != 4)
    {
        printf("ERROR: number of arguments is wrong (Ntrain_t, Npredict_t, Ninputs_t)\n");
    }
    
    Ntrain_t = atoi(argv[1]);
    Npredict_t = atoi(argv[2]);
    Ninputs_t = atoi(argv[3]);
    theta_size_t = Ninputs_t + 2;
    printf("===============================\n");
    printf("Testing with problem size:\n( npredict = %d, ntrain = %d, ninputs = %d )\n", Npredict_t, Ntrain_t, Ninputs_t);
    printf("===============================");

    CU_pSuite pSuite = NULL;

   /* initialize the CUnit test registry */
   if (CUE_SUCCESS != CU_initialize_registry())
      return CU_get_error();

   /* add a suite to the registry */
   pSuite = CU_add_suite("Unit Tests", init_suite1, clean_suite);
   
   if (NULL == pSuite) {
      CU_cleanup_registry();
      return CU_get_error();
   }


  /* add the tests to the suite */
   if ((NULL == CU_add_test(pSuite, "test of gpu_vectorTimesMatrix", tests_VecTimesMat)) ||
       (NULL == CU_add_test(pSuite, "test of gpu_init_array", testInitArray))||
       (NULL == CU_add_test(pSuite, "test of gpu_cdist", tests_cdist))||
       (NULL == CU_add_test(pSuite, "test of gpu_MatrixExp", tests_matrixExp))||
       (NULL == CU_add_test(pSuite, "test of cublasgemm", tests_cublasgemm))||
       (NULL == CU_add_test(pSuite, "test of gpu_predict", testPredict))
       )
   {
      CU_cleanup_registry();
      return CU_get_error();
   }

   //Run all tests using the CUnit Basic interface 
   CU_basic_set_mode(CU_BRM_VERBOSE);
   CU_basic_run_tests();
   CU_cleanup_registry();
   printf("\n");
   return CU_get_error();
}


