#include <stdio.h>
#include <string.h>
#include "CUnit/Basic.h"
#include "gpu_predict_test.h"





static real * expXsqrt, *inputs, *testing;
static real * cdist_a;
static real *cdist_test_var1, *cdist_test_var2, *cdist_test_var3;
 

static int M, N, D;
static int init_suite1(void)
{
    M = 250;
    N = 100;
    D = 10;
   
    expXsqrt        =  readTestData( "expXsqrt.bin", M, N, D, D );
    inputs          =  readTestData( "inputs.bin", M, N, D, M * D );
    testing         =  readTestData( "testing.bin", M, N, D, N * D );
    cdist_a         =  readTestData( "cdist_a.bin", M, N, D, M * N);
    cdist_test_var1 =  readTestData( "cdist_test_var1.bin", M, N, D, M * D);
    cdist_test_var2 =  readTestData( "cdist_test_var2.bin", M, N, D, N * D);
    return 0;
    
}


static int init_suite2(void)
{
    M = 250;
    N = 100;
    D = 10;
   
    expXsqrt        =  readTestData( "expXsqrt.bin", M, N, D, D );
    inputs          =  readTestData( "inputs.bin", M, N, D, M * D );
    testing         =  readTestData( "testing.bin", M, N, D, N * D );
    cdist_a         =  readTestData( "cdist_a.bin", M, N, D, M * N);
    cdist_test_var1 =  readTestData( "cdist_test_var1.bin", M, N, D, M * D);
    cdist_test_var2 =  readTestData( "cdist_test_var2.bin", M, N, D, N * D);
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

void testFPRINTF(void)
{
    printf("\ncdist_a[0] = %f\n", cdist_a[0]);
}

void testFREAD(void)
{
}

int main()
{
   CU_pSuite pSuite = NULL;

   /* initialize the CUnit test registry */
   if (CUE_SUCCESS != CU_initialize_registry())
      return CU_get_error();

   /* add a suite to the registry */
   pSuite = CU_add_suite("Suit_1", init_suite1, clean_suite);
   
   if (NULL == pSuite) {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* add the tests to the suite */
   if ((NULL == CU_add_test(pSuite, "test of fprintf()", testFPRINTF)) ||
       (NULL == CU_add_test(pSuite, "test of fread()", testFREAD)))
   {
      CU_cleanup_registry();
      return CU_get_error();
   }

  
 
   
   
   pSuite = CU_add_suite("Suit_2", init_suite2, clean_suite);
   
   if (NULL == pSuite) {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* add the tests to the suite */
   if ((NULL == CU_add_test(pSuite, "test of fprintf()", testFPRINTF)) ||
       (NULL == CU_add_test(pSuite, "test of fread()", testFREAD)))
   {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* Run all tests using the CUnit Basic interface */
   CU_basic_set_mode(CU_BRM_VERBOSE);
   CU_basic_run_tests();
   CU_cleanup_registry();
   return CU_get_error();

   /* Run all tests using the CUnit Basic interface */
   CU_basic_set_mode(CU_BRM_VERBOSE);
   CU_basic_run_tests();
   CU_cleanup_registry();
   return CU_get_error();
   
 

}


