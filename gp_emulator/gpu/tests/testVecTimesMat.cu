#include "gpu_predict_test.h"

void testVecTimesMat(const real *c_vec,const  real *c_matrix, const real *c_res,const int vec_len, const int matrix_nrows, const int matrix_ncols)
{
    real *d_vec, *d_matrix, *d_res;
    real *c_res_gpu;
    real *c_matrix_T;

    CU_ASSERT (vec_len == matrix_ncols);
    cudaMalloc((void **)&d_vec, sizeof(real) * vec_len );
    cudaMalloc((void **)&d_matrix, sizeof(real) * matrix_nrows * matrix_ncols );
    cudaMalloc((void **)&d_res, sizeof(real) * matrix_nrows * matrix_ncols );
    
    c_matrix_T = (real *)malloc( sizeof(real) * matrix_nrows * matrix_ncols);
    c_res_gpu = (real *)malloc( sizeof(real) * matrix_nrows * matrix_ncols );

    c_matrix_T = computeTranspose(c_matrix, matrix_ncols, matrix_nrows);

    cublasCheckErrors(cublasSetVector( vec_len, sizeof(real), c_vec, 1, d_vec, 1 ));
    cublasCheckErrors(cublasSetMatrix( matrix_nrows, matrix_ncols, sizeof(real), c_matrix_T, matrix_nrows, d_matrix, matrix_nrows));
    
    gpu_vectorTimesMatrix( d_matrix, d_vec, d_res, matrix_nrows, matrix_ncols);

    cudaMemcpy(c_res_gpu, d_res, sizeof(real) * matrix_nrows * matrix_ncols, cudaMemcpyDeviceToHost);
    c_res_gpu = computeTranspose(c_res_gpu, matrix_nrows, matrix_ncols);
    compare_result( c_res_gpu, c_res, matrix_nrows * matrix_ncols, EPSILON_AVG, EPSILON_MAX, "RESULT");

    free(c_res_gpu);
    free(c_matrix_T);

    cudaFree(d_vec);
    cudaFree(d_matrix);
    cudaFree(d_res);

}
    
