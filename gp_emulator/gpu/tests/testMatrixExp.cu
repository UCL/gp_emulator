#include "gpu_predict_test.h"


void testMatrixExp(const real *mat, const real *res, const real alpha,const real beta,const int size)
{
    int i;
    int error = 0;
    real *d_mat;
    real *gpu_res;

    gpu_res = (real *)malloc(sizeof(real) * size);
    cudaMalloc((void **)&d_mat, sizeof(real) * size);

    cudaMemcpy(d_mat, mat, sizeof(real) * size, cudaMemcpyHostToDevice);
//    int nblocks , nthreads;
//    nthreads=1000; 
//    nblocks=ceil(float(size) / float(CUDA_BLOCK) / 1000);
    
    gpu_matrixExp(d_mat, alpha, beta, size);

    cudaMemcpy( gpu_res, d_mat, sizeof(real) * size, cudaMemcpyDeviceToHost);
    
    for( i = 0; i < size; i++ )
    {
        if(abs( gpu_res[i] - res[i] ) / res[i] > epsilon)
            error++;
    }
    
    if( error != 0)
        printf("MatrixExp error [%d/%d]", error,size);
    CU_ASSERT( error == 0 );
    
    cudaFree(d_mat);
    free(gpu_res);
}
    
