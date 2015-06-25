#include "gpu_predict_test.h"

void testCdist( const real *matrix1,const real *matrix2, const real *result, 
        const int matrix1_nrows, const int matrix2_nrows, const int matrix_ncols )
{
    int i;
    real *matrix1_T, *matrix2_T, *gpu_result;
    real *d_matrix1, *d_matrix2, *d_result; 

    gpu_result = (real *)malloc( sizeof(real) * matrix1_nrows * matrix2_nrows ); 
    matrix1_T = computeTranspose( matrix1, matrix_ncols, matrix1_nrows );
    matrix2_T = computeTranspose( matrix2, matrix_ncols, matrix2_nrows );
    
    /*GPU part*/
    for( i = 0; i < matrix1_nrows * matrix2_nrows; i++ )
        gpu_result[i] = 0;
    
    
    cudaMalloc((void **)&d_matrix1, sizeof(real) * matrix1_nrows * matrix_ncols );
    cudaMalloc((void **)&d_matrix2, sizeof(real) * matrix2_nrows * matrix_ncols );
    cudaMalloc((void **)&d_result, sizeof(real) * matrix2_nrows * matrix1_nrows );
    cublasCheckErrors(cublasSetMatrix( matrix1_nrows, matrix_ncols, sizeof(real), matrix1_T, matrix1_nrows, d_matrix1, matrix1_nrows) );
    cublasCheckErrors(cublasSetMatrix( matrix2_nrows, matrix_ncols, sizeof(real), matrix2_T, matrix2_nrows, d_matrix2, matrix2_nrows) );
    cublasCheckErrors(cublasSetMatrix( matrix2_nrows, matrix1_nrows, sizeof(real), gpu_result, matrix2_nrows, d_result,matrix2_nrows) );
    gpu_cdist(d_matrix1, d_matrix2, d_result, matrix1_nrows, matrix_ncols, matrix2_nrows, matrix_ncols);
    cudaMemcpy(gpu_result, d_result, sizeof(real) * matrix2_nrows * matrix1_nrows, cudaMemcpyDeviceToHost);
    
    compare_result(gpu_result, result, matrix1_nrows * matrix2_nrows, EPSILON_AVG, EPSILON_MAX, "RESULT");
    
    free(matrix1_T);
    free(matrix2_T);
    free(gpu_result);
   
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);
}
