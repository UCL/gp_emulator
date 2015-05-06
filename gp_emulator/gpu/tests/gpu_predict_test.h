#include <stdio.h>
#include <stdlib.h>
#include "../gpu_predict.h"
#include "CUnit/Basic.h"
#include "../gpu_predict.h"
#include <cuda.h>




static  real *expXsqrt, *inputs, *testing;
static real *cdist_a;
static  real *cdist_test_var1, *cdist_test_var2, *cdist_test_var3;


static  int M, N, D;



real *readTestData(char *file_name, int M, int N, int D, int size);


void testVecTimesMat(real *c_vec,  real *c_mat, const real *c_res,const int vec_len, const int mat_nrows, const int mat_ncols,  const dim3 nblock, const dim3 nthreads);
