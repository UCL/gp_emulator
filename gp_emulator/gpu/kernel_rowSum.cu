/*********************************************//**
 * row sum:
 * return vector rowSum(A)
 *********************************************/
#include "gpu_predict.h"

real* gpu_rowSum(const real *A, const int A_nrows,const int A_ncols)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat=cublasCreate(&handle);

    real alpha = 1.f;
    real beta = 0.f;
    real *vec_one;
    real *d_var;

    cudaMalloc((void **)&vec_one, sizeof(real) * A_ncols );
    cudaMalloc((void **)&d_var, sizeof(real) * A_ncols);

    gpu_init_array(vec_one, 1.0, A_ncols);
    gpu_init_array(d_var, 0.0, A_ncols);

    cublasCheckErrors(CUBLAS_GEMV(handle, CUBLAS_OP_T, A_nrows, A_ncols, &alpha, A, A_nrows, vec_one, 1, &beta, d_var, 1));

    cudaFree(vec_one);
    cublasDestroy(handle);
    return d_var;
}
