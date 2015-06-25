/*********************************************//**
 * row sum:
 * return vector rowSum(matrix)
 *********************************************/
#include "gpu_predict.h"

real* gpu_rowSum(const real *matrix, const int matrix_nrows,const int matrix_ncols)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    real alpha = 1.f;
    real beta = 0.f;
    real *vec_one;
    real *d_result;

    cudaMalloc((void **)&vec_one, sizeof(real) * matrix_ncols);
    cudaMalloc((void **)&d_result, sizeof(real) * matrix_ncols);

    gpu_init_array(vec_one, 1.0, matrix_ncols);
    gpu_init_array(d_result, 0.0, matrix_ncols);

    cublasCheckErrors(CUBLAS_GEMV(handle, CUBLAS_OP_T, matrix_nrows, matrix_ncols, &alpha, matrix, matrix_nrows, vec_one, 1, &beta, d_result, 1));

    cudaFree(vec_one);
    cublasDestroy(handle);
    return d_result;
}
