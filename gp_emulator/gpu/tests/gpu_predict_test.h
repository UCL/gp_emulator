#include <stdio.h>
#include <stdlib.h>
#include "../gpu_predict.h"
#include "CUnit/Basic.h"
#include <cuda.h>




static real *expX, *expXsqrt, *inputs, *testing, *invQ, *invQt;
static real *cdist_a, *cdist_expa;
static real *cdist_test_var1, *cdist_test_var2, *cdist_test_var3;
static real *var_test1;

static real *mu, *var, *deriv;

static  int M, N, D;
#define epsilon 1e-3

real *readTestData(char *file_name, int M, int N, int D, int size);


void testInitArray(void);

void testVecTimesMat(const real *c_vec,const  real *c_mat, const real *c_res,const int vec_len, const int mat_nrows, const int mat_ncols);
void testCdist(const real *in1,const real *in2, const real *res, const int in1_nrows, const int in2_nrows, const int in_ncols);
void testMatrixExp(const real *mat, const real *res, const real alpha,const real beta,const int size);
void testCublasgemm(const real *c_mat1, const real *c_mat2, const real *c_res, const int mat1_nrows, const int mat1_ncols, const int mat2_nrows, const int mat2_ncols);
void testPredict(const real *expX, const real *inputs, const real *invQt, const real *invQ, const real *testing, const real *mu, const real *var, const real *deriv, int M, int N, int D);
