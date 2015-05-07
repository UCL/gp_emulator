#include "gpu_predict_test.h"


void testMatrixExp(const real *mat, const real *res, const real alpha,const real beta,const int size)
{
    int i;
    real error;
    real *d_mat;
    real *gpu_res;

    gpu_res = (real *)malloc(sizeof(real) * size);
    cudaMalloc((void **)&d_mat, sizeof(real) * size);

    cudaMemcpy(d_mat, mat, sizeof(real) * size, cudaMemcpyHostToDevice);
    int nblocks , nthreads;
    nthreads = 512;
    nblocks = ceil( float(size) / nthreads);
    
    gpu_matrixExp<<< nblocks, nthreads  >>> (d_mat, alpha, beta);

    cudaMemcpy( gpu_res, d_mat, sizeof(real) * size, cudaMemcpyDeviceToHost);
    
    for( i = 0; i < size; i++ )
    {
        error = abs( gpu_res[i] - res[i] );
        CU_ASSERT( error < 1e-6 );
    }

    cudaFree(d_mat);
    free(gpu_res);
}
    
