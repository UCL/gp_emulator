#include <stdio.h>
#include <stdlib.h>
#include "../gpu_predict.h"
#include "CUnit/Basic.h"
#include "../gpu_predict.h"
#include <cuda.h>




static  real *expX, *expXsqrt, *inputs, *testing;
static  real *cdist_a, *cdist_expa;
static  real *cdist_test_var1, *cdist_test_var2, *cdist_test_var3;


static  int M, N, D;



real *readTestData(char *file_name, int M, int N, int D, int size);


void testVecTimesMat(const real *c_vec,const  real *c_mat, const real *c_res,const int vec_len, const int mat_nrows, const int mat_ncols,  const dim3 nblock, const dim3 nthreads);
void testCdist(const real *in1,const real *in2, const real *res, const int in1_nrows, const int in2_nrows, const int in_ncols,  const dim3 nblocks, const dim3 nthreads);
void testMatrixExp(const real *mat, const real *res, const real alpha,const real beta,const int size);
