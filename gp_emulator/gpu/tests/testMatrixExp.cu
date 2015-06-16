#include "gpu_predict_test.h"


void testMatrixExp(const real *mat, const real *res, const real alpha,const real beta,const int size)
{
    real *d_mat;
    real *gpu_res;

    gpu_res = (real *)malloc( sizeof(real) * size );
    cudaMalloc( (void **)&d_mat, sizeof(real) * size );

    cudaMemcpy( d_mat, mat, sizeof(real) * size, cudaMemcpyHostToDevice );
    gpu_matrixExp( d_mat, alpha, beta, size );
    cudaMemcpy( gpu_res, d_mat, sizeof(real) * size, cudaMemcpyDeviceToHost);
    compare_result( gpu_res, res, size, EPSILON_AVG, EPSILON_MAX, "RESULT");
    
    cudaFree(d_mat);
    free(gpu_res);
}
    
