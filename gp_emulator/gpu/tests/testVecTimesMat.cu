#include "gpu_predict_test.h"


void testVecTimesMat(real *c_vec,  real *c_mat, const real *c_res,const int vec_len, const int mat_nrows, const int mat_ncols,  const dim3 nblock, const dim3 nthreads)
{
    real *d_vec, *d_mat, *d_res;
    real *c_res_gpu;
    
    CU_ASSERT (vec_len == mat_ncols);
    int i;
    int epsilon;
    cudaMalloc((void **)&d_vec, sizeof(real) * vec_len );
    cudaMalloc((void **)&d_mat, sizeof(real) * mat_nrows * mat_ncols );
    cudaMalloc((void **)&d_res, sizeof(real) * mat_nrows * mat_ncols );
    
    c_res_gpu = (real *)malloc( sizeof(real) * mat_nrows * mat_ncols );


    computeTranspose(c_mat, mat_ncols, mat_nrows);

    cublasCheckErrors(cublasSetVector( vec_len, sizeof(real), c_vec, 1, d_vec, 1 ));
    cublasCheckErrors(cublasSetMatrix( mat_nrows, mat_ncols, sizeof(real), c_mat, mat_nrows, d_mat, mat_nrows));
    
    gpu_vectorTimesMatrix <<< nblock, nthreads >>> ( d_mat, d_vec, d_res, mat_nrows );

    cudaMemcpy(c_res_gpu, d_res, sizeof(real) * mat_nrows * mat_ncols, cudaMemcpyDeviceToHost);
    computeTranspose(c_res_gpu, mat_nrows, mat_ncols);
    
    for( i = 0; i < mat_nrows * mat_ncols; i++)
    {
        epsilon = abs( c_res[i] - c_res_gpu[i] );
        CU_ASSERT( epsilon < 1e-6 );
//        if( epsilon < 1e-6 )
//            printf("res [%d] = %f   gpu result [%d] = %f\n",i, c_res[i], i, c_res_gpu[i]);
    }

}
    
