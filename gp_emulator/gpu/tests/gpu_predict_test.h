#include <stdio.h>
#include <stdlib.h>
#include "../gpu_predict.h"
#include "CUnit/Basic.h"
#include <cuda.h>

extern real *expX, *expXsqrt, *in_train, *in_predict, *invQ, *invQt;
extern real *cdist_a, *cdist_expa;
extern real *cdist_test_var1, *cdist_test_var2, *cdist_test_var3;
extern real *error_test1;

extern real *result_t, *error_t, *deriv_t;

extern int Npredict_t, Ntrain_t, Ninputs_t;
extern int theta_size_t;

#ifdef DOUBLE__PRECISION
    #define EPSILON_AVG 1e-6
    #define EPSILON_MAX 1e-2
#else
    #define EPSILON_AVG 1e-3
    #define EPSILON_MAX 1
#endif


real *readTestData(char *file_name, int size);
void compare_result(const real *test_val, const real *origin_val, 
        const int len, const real epsilon_average, 
        const real epsilon_max, char var_name[]);
void testInitArray(void);

void testVecTimesMat(const real *c_vec,
        const real *c_mat, const real *c_res,
        const int vec_len, const int mat_nrows, 
        const int mat_ncols);
void testCdist(const real *in1, const real *in2, 
        const real *res, const int in1_nrows, 
        const int in2_nrows, const int in_ncols);
void testMatrixExp(const real *mat, 
        const real *res, const real alpha,
        const real beta,const int size);
void testCublasgemm(const real *c_mat1, 
        const real *c_mat2, const real *c_res, 
        const int mat1_nrows, const int mat1_ncols, 
        const int mat2_nrows, const int mat2_ncols);
/*
void testPredict(real *expX, real *expXsqrt, 
        real *invQt, real *invQ, real *in_predict, real *in_train, 
        int Npredict_t, int  Ntrain_t, int Ninputs_t, int theta_size_t, 
        real *result_t, real *error_t, real *deriv_t);*/
void testPredict(void);
