#include "gpu_predict_test.h"

void testCublasgemm(const real *c_matrix1, const real *c_matrix2, const real *c_result, 
        const int matrix1_nrows, const int matrix1_ncols, const int matrix2_nrows, 
        const int matrix2_ncols)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    

    real *d_matrix1, *d_matrix2, *d_result;
    real *c_gpu_result;
    c_gpu_result = (real *)malloc( sizeof(real) * matrix2_nrows * matrix2_ncols);
    
    cudaMalloc( (void **)&d_matrix1, sizeof(real) * matrix1_nrows * matrix1_ncols );
    cudaMalloc( (void **)&d_matrix2, sizeof(real) * matrix2_nrows * matrix2_ncols );
    cudaMalloc( (void **)&d_result, sizeof(real) * matrix1_nrows * matrix2_ncols );


    real *c_matrix1_T;
    
    
   c_matrix1_T = computeTranspose(c_matrix1, matrix1_nrows, matrix1_ncols); 

    cudaMemcpy( d_matrix1, c_matrix1_T, sizeof(real) * matrix1_nrows * matrix1_ncols, cudaMemcpyHostToDevice);
    cudaMemcpy( d_matrix2, c_matrix2, sizeof(real) * matrix1_nrows * matrix2_ncols, cudaMemcpyHostToDevice);


    real alpha = 1.f;
    real beta = 0.f;
    CUBLAS_GEMM( handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                matrix1_nrows, matrix2_ncols,  matrix1_ncols,
                &alpha, 
                d_matrix1, matrix1_nrows, 
                d_matrix2, matrix2_ncols, 
                &beta, 
                d_result, matrix1_ncols );

    cudaMemcpy( c_gpu_result, d_result, sizeof(real) * matrix1_nrows *  matrix2_ncols, cudaMemcpyDeviceToHost);
    c_gpu_result = computeTranspose(c_gpu_result, matrix1_nrows, matrix2_ncols); 
    compare_result( c_gpu_result, c_result, matrix1_nrows * matrix2_ncols, EPSILON_AVG, EPSILON_MAX, "RESULTS");
    
    free(c_matrix1_T);
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);
    free(c_gpu_result);
}
