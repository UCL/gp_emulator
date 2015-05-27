#include "gpu_predict_test.h"

void testCublasgemm(const real *c_mat1, const real *c_mat2, const real *c_res, const int mat1_nrows, const int mat1_ncols, const int mat2_nrows, const int mat2_ncols)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat=cublasCreate(&handle);
    

    int i;
    real e;

    real *d_mat1, *d_mat2, *d_res;
    real *c_gpu_res;
    c_gpu_res = (real *)malloc( sizeof(real) * mat2_nrows * mat2_ncols);
    
    cudaMalloc( (void **)&d_mat1, sizeof(real) * mat1_nrows * mat1_ncols );
    cudaMalloc( (void **)&d_mat2, sizeof(real) * mat2_nrows * mat2_ncols );
    cudaMalloc( (void **)&d_res, sizeof(real) * mat1_nrows * mat2_ncols );


    real *c_mat1_T, *c_mat2_T;
    c_mat1_T = (real *)malloc(sizeof(real)* mat1_ncols * mat1_nrows);
    c_mat2_T = (real *)malloc(sizeof(real)* mat2_nrows * mat2_nrows);

/*

    for(i=0;i<mat1_ncols * mat1_nrows;i++)
        c_mat1_T[i] = c_mat1[i];

    for(i=0;i<mat2_ncols*mat2_nrows;i++)
        c_mat2_T[i] = c_mat2[i];

    
    
    
   computeTranspose(c_mat1_T, mat1_nrows, mat1_ncols); 
   //computeTranspose(c_mat2_T, mat2_ncols, mat2_nrows);
    


    cudaMemcpy( d_mat1, c_mat1_T, sizeof(real) * mat1_nrows * mat1_ncols, cudaMemcpyHostToDevice);
    cudaMemcpy( d_mat2, c_mat2_T, sizeof(real) * mat1_nrows * mat2_ncols, cudaMemcpyHostToDevice);


    real alpha = 1.f;
    real beta = 0.f;
    cublasCheckErrors( CUBLAS_GEMM( handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                mat1_nrows, mat2_ncols,  mat1_ncols,
                &alpha, 
                d_mat1, mat1_nrows, 
                d_mat2, mat2_ncols, 
                &beta, 
                d_res, 250 ));

    cudaMemcpy( c_gpu_res, d_res, sizeof(real) * mat1_nrows *  mat2_ncols, cudaMemcpyDeviceToHost);

 

    computeTranspose(c_gpu_res, mat1_nrows, mat2_ncols); 
    int error;
    for(i = 0; i < mat1_nrows * mat2_ncols; i++)
    {
        if( abs( c_gpu_res[i] - c_res[i] ) > epsilon )
            error++;
    }
    
    if( error != 0 )
        printf("  ERROR: cublasgemm [%d/%d]   ", error, mat1_nrows*mat2_ncols);
    CU_ASSERT( error == 0);

*/



    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_res);
    free(c_gpu_res);
}
