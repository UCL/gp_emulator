#include "gpu_predict_test.h"


void testVecTimesMat(const real *c_vec,const  real *c_mat, const real *c_res,const int vec_len, const int mat_nrows, const int mat_ncols,  const dim3 nblock, const dim3 nthreads)
{
    real *d_vec, *d_mat, *d_res;
    real *c_res_gpu;

    real *c_mat_T;


    CU_ASSERT (vec_len == mat_ncols);
    int i;
    cudaMalloc((void **)&d_vec, sizeof(real) * vec_len );
    cudaMalloc((void **)&d_mat, sizeof(real) * mat_nrows * mat_ncols );
    cudaMalloc((void **)&d_res, sizeof(real) * mat_nrows * mat_ncols );
    
    c_mat_T = (real *)malloc( sizeof(real) * mat_nrows * mat_ncols);
    c_res_gpu = (real *)malloc( sizeof(real) * mat_nrows * mat_ncols );

    for( i = 0; i < mat_nrows * mat_ncols; i++ )
        c_mat_T[i] = c_mat[i];
    computeTranspose(c_mat_T, mat_ncols, mat_nrows);


    cublasCheckErrors(cublasSetVector( vec_len, sizeof(real), c_vec, 1, d_vec, 1 ));
    cublasCheckErrors(cublasSetMatrix( mat_nrows, mat_ncols, sizeof(real), c_mat_T, mat_nrows, d_mat, mat_nrows));
    
    gpu_vectorTimesMatrix <<< nblock, nthreads >>> ( d_mat, d_vec, d_res, mat_nrows );

    cudaMemcpy(c_res_gpu, d_res, sizeof(real) * mat_nrows * mat_ncols, cudaMemcpyDeviceToHost);
    computeTranspose(c_res_gpu, mat_nrows, mat_ncols);
    
    int error = 0;
    for( i = 0; i < mat_nrows * mat_ncols; i++)
    {
        if( abs(c_res[i] - c_res_gpu[i]) > epsilon )
            error++;
    }
    
    CU_ASSERT( error == 0);

    free(c_res_gpu);
    free(c_mat_T);

    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_res);

}
    
