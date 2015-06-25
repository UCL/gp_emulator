#include "gpu_predict_test.h"

void testCublasgemm(const real *c_mat1, const real *c_mat2, const real *c_res, 
        const int mat1_nrows, const int mat1_ncols, const int mat2_nrows, 
        const int mat2_ncols)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    

    int i;

    real *d_mat1, *d_mat2, *d_res;
    real *c_gpu_res;
    c_gpu_res = (real *)malloc( sizeof(real) * mat2_nrows * mat2_ncols);
    
    cudaMalloc( (void **)&d_mat1, sizeof(real) * mat1_nrows * mat1_ncols );
    cudaMalloc( (void **)&d_mat2, sizeof(real) * mat2_nrows * mat2_ncols );
    cudaMalloc( (void **)&d_res, sizeof(real) * mat1_nrows * mat2_ncols );


    real *c_mat1_T;
    
    
   c_mat1_T = computeTranspose(c_mat1, mat1_nrows, mat1_ncols); 

    cudaMemcpy( d_mat1, c_mat1_T, sizeof(real) * mat1_nrows * mat1_ncols, cudaMemcpyHostToDevice);
    cudaMemcpy( d_mat2, c_mat2, sizeof(real) * mat1_nrows * mat2_ncols, cudaMemcpyHostToDevice);


    real alpha = 1.f;
    real beta = 0.f;
    cublasCheckErrors( CUBLAS_GEMM( handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                mat1_nrows, mat2_ncols,  mat1_ncols,
                &alpha, 
                d_mat1, mat1_nrows, 
                d_mat2, mat2_ncols, 
                &beta, 
                d_res, mat1_ncols ));

    cudaMemcpy( c_gpu_res, d_res, sizeof(real) * mat1_nrows *  mat2_ncols, cudaMemcpyDeviceToHost);
    c_gpu_res = computeTranspose(c_gpu_res, mat1_nrows, mat2_ncols); 
    compare_result( c_gpu_res, c_res, mat1_nrows * mat2_ncols, EPSILON_AVG, EPSILON_MAX, "RESULTS");
    
    free(c_mat1_T);
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_res);
    free(c_gpu_res);
}
