#include <stdio.h>
#include <string.h>
#include "gpu_predict_test.h"
#include "cuda.h"

/*
static real *expXsqrt, *inputs, *testing;
static real *cdist_a;
static real *cdist_test_var1, *cdist_test_var2, *cdist_test_var3;
 

static  int M, N, D;

*/
static int init_suite1(void)
{
    invQ            =  readTestData( "invQ.bin", M, N, D, M * M);
    invQt           =  readTestData( "invQt.bin", M, N, D, M );
    
    expX            =  readTestData( "expX.bin", M, N, D, D+2);
    expXsqrt        =  readTestData( "expXsqrt.bin", M, N, D, D );
    inputs          =  readTestData( "inputs.bin", M, N, D, M * D );
    testing         =  readTestData( "testing.bin", M, N, D, N * D );
    cdist_test_var1 =  readTestData( "cdist_test_var1.bin", M, N, D, M * D);
    cdist_test_var2 =  readTestData( "cdist_test_var2.bin", M, N, D, N * D);
    cdist_a         =  readTestData( "cdist_a.bin", M, N, D, M * N);
    cdist_expa      =  readTestData( "cdist_expa.bin", M, N, D, M * N ); 
    
    var_test1       =  readTestData( "var_test1.bin", M, N, D, M * N );
    mu              =  readTestData( "mu.bin", M, N, D, M * N);
    var             =  readTestData( "var.bin", M, N, D, N);
    deriv           =  readTestData( "deriv.bin", M, N, D, M * N );
    
    return 0;
    
}






int clean_suite(void)
{
    free(expXsqrt);
    free(inputs);
    free(testing);
    free(cdist_a);
    free(cdist_test_var1);
    free(cdist_test_var2);
    return 0;
}

static void tests_VecTimesMat(void)
{
    dim3 nblocks_1(M,1);
    dim3 nthreads_1(1,D);
    testVecTimesMat(expXsqrt, inputs, cdist_test_var1, D, M, D, nblocks_1, nthreads_1 );
    dim3 nblocks_2(N/100,1);
    dim3 nthreads_2(100,D);
    testVecTimesMat(expXsqrt, testing, cdist_test_var2, D, N, D, nblocks_2, nthreads_2 );
}

void tests_cdist(void)
{
    dim3 nblocks(N/200, M/5, D);
    dim3 nthreads(200, 5, 1);
    testCdist(cdist_test_var1,cdist_test_var2, cdist_a, M, N, D, nblocks, nthreads);
}


void tests_matrixExp(void)
{
    testMatrixExp( cdist_a, cdist_expa, -0.5, expX[D], M * N );
}

void tests_cublasgemm(void)
{
   testCublasgemm(invQ, cdist_expa, var_test1, M, M, M, N);
}



void tests_predict(void)
{
   testPredict(expX, inputs, invQt, invQ, testing, mu, var, deriv, M, N, D);
}
























int main(int argc, char *argv[])
{
   
    if( argc != 4)
    {
        printf("ERROR: number of arguments is wrong (M, N, D)\n");
    }
    
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    D = atoi(argv[3]);

    printf("%d,%d,%d\n", M, N, D);
    CU_pSuite pSuite = NULL;

   /* initialize the CUnit test registry */
   if (CUE_SUCCESS != CU_initialize_registry())
      return CU_get_error();

   /* add a suite to the registry */
   pSuite = CU_add_suite("Testing data set (3e4, 250, 10)", init_suite1, clean_suite);
   
   if (NULL == pSuite) {
      CU_cleanup_registry();
      return CU_get_error();
   }


  /* add the tests to the suite */
   if ((NULL == CU_add_test(pSuite, "test of gpu_vectorTimesMatrix", tests_VecTimesMat)) ||
       (NULL == CU_add_test(pSuite, "test of gpu_cdist", tests_cdist))||
       (NULL == CU_add_test(pSuite, "test of gpu_MatrixExp", tests_matrixExp))||
       (NULL == CU_add_test(pSuite, "test of cublasgemm", tests_cublasgemm))||
       (NULL == CU_add_test(pSuite, "test of gpu_predict", tests_predict))
      )
   {
      CU_cleanup_registry();
      return CU_get_error();
   }

   //Run all tests using the CUnit Basic interface 
   CU_basic_set_mode(CU_BRM_VERBOSE);
   CU_basic_run_tests();
   CU_cleanup_registry();
   return CU_get_error();


 
  /* 
   
   pSuite = CU_add_suite("Testing data set (100, 250, 10)", init_suite2, clean_suite);
   
   if (NULL == pSuite) {
      CU_cleanup_registry();
      return CU_get_error();
   }

   // add the tests to the suite 
   if ((NULL == CU_add_test(pSuite, "test of gpu_vectorTimesMatrix", tests_VecTimesMat)) ||
       (NULL == CU_add_test(pSuite, "test of gpu_cdist", tests_cdist))||
       (NULL == CU_add_test(pSuite, "test of gpu_MatrixExp", tests_matrixExp))||
       (NULL == CU_add_test(pSuite, "test of cublasgemm", tests_cublasgemm))||
       (NULL == CU_add_test(pSuite, "test of gpu_predict", tests_predict))
     )
   {
      CU_cleanup_registry();
      return CU_get_error();
   }


   // Run all tests using the CUnit Basic interface 
   CU_basic_set_mode(CU_BRM_VERBOSE);
   CU_basic_run_tests();
   CU_cleanup_registry();
   return CU_get_error();
   */
 

}


