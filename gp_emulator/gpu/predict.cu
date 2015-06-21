/*********************************************//** 
 * CUDA implementation of derived from GaussianProcess
 * in python code. 
 * 
 * Sinan Shi 
 *********************************************/
#include "gpu_predict.h"
#include <stdlib.h>
#define debug

/*********************************//*
 * Euclidian distance calculation
 * 1) res_mv1 = theta_exp_sqrt_{,Ninputs} * train
 * 2) res_mv2 = theta_exp_sqrt_{,Ninputs} * predict
 * 3) dist_matrix = cidist(res_mv1, res_mv2)
 * 4) dist_matrix = -0.5 * exp(expX_{Ninputs})
 * Notice: distance is equivalent to distance^T (a^T) in python
 ********************************/ 
real *compute_distance(const real *d_predict, const real *d_train, 
                      const real *c_theta_exp,const real *d_theta_exp, 
                      const real *d_theta_exp_sqrt,
                      const int Npredict, const int Ntrain, const int Ninputs)
{
    real *d_res_mv1, *d_res_mv2, *d_dist_matrix;
    cudaMalloc((void **)&d_res_mv1, sizeof(real) * Ntrain * Ninputs);
    cudaMalloc((void **)&d_res_mv2, sizeof(real) * Npredict * Ninputs);
    cudaMalloc((void **)&d_dist_matrix, sizeof(real) * Ntrain * Npredict);

    gpu_vectorTimesMatrix(d_train, d_theta_exp_sqrt, d_res_mv1, Ntrain, Ninputs);
    gpu_vectorTimesMatrix(d_predict, d_theta_exp_sqrt, d_res_mv2, Npredict, Ninputs);
    gpu_init_array( d_dist_matrix, 0.0, Npredict * Ntrain );
    gpu_cdist(d_res_mv1, d_res_mv2, d_dist_matrix, Ntrain, Ninputs, Npredict, Ninputs);
    gpu_matrixExp(d_dist_matrix, -0.5, c_theta_exp[Ninputs], Ntrain * Npredict);

    cudaFree(d_res_mv1);
    cudaFree(d_res_mv2);
    return(d_dist_matrix);
}


/*********************************//*
 * compute result:
 * c_result = dist_matrix * invQt (dot product)
 ********************************/
void compute_result(cublasHandle_t handle, real *c_result, const real *d_dist_matrix, const real *d_invQt,
                const int Npredict, const int Ntrain, const int Ninputs)
{
    real *d_result;
    cudaMalloc((void **)&d_result, sizeof(real) * Npredict);
    real alpha = 1.f;
    real beta = 0.f;
    cublasCheckErrors(CUBLAS_GEMV(handle, CUBLAS_OP_N, Npredict, Ntrain, &alpha, d_dist_matrix, 
                Npredict, d_invQt, 1, &beta, d_result, 1));
    cudaMemcpy(c_result, d_result, sizeof(real) * Npredict, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

real *gpu_transpose(cublasHandle_t handle, real *d_matrix, const int nrow, const int ncol)
{
    real *d_matrix_T;
    real alpha = 1.f;
    real beta = 0.f;
    cudaMalloc((void **)&d_matrix_T, sizeof(real) * nrow * ncol );
    cublasCheckErrors(CUBLAS_GEAM(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                nrow, ncol, &alpha, d_matrix, ncol, &beta, 
                d_matrix, nrow, d_matrix_T, nrow));
    return( d_matrix_T );
}
  
/*********************************
 * compute error:
 * c_error = b - rowsum(a * dot(invQ, d_dist_matrix_T))
 * arguments d_invQ, d_dist_matrix have been freed.
 ********************************/
void compute_error(cublasHandle_t handle, real *c_error,
        real *d_dist_matrix,const  real *d_dist_matrix_T, 
        real *d_invQ, const real *c_theta_exp,
        const int Npredict, const int Ntrain, const int Ninputs)
{
    real alpha = 1.f;
    real beta = 0.f;
    real *d_res_dot;
    real *d_error;
    
    cudaMalloc((void **)&d_res_dot, sizeof(real) * Ntrain * Npredict);
    cublasCheckErrors(CUBLAS_GEMM(
                handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                Ntrain, Npredict, Ntrain, 
                &alpha, d_invQ, Ntrain, 
                d_dist_matrix, Npredict, 
                &beta, d_res_dot, Ntrain));    // dot(invQ, d_dist_matrix_T)
    gpu_elementwiseMult(d_dist_matrix_T, d_res_dot, Ntrain * Npredict);
    d_error = gpu_rowSum(d_res_dot, Ntrain, Npredict);
    gpu_scalarMinusVec(d_error, c_theta_exp[Ninputs], Npredict );
 
    cudaMemcpy(c_error, d_error, sizeof(real) * Npredict, cudaMemcpyDeviceToHost);

    cudaFree(d_dist_matrix);
    cudaFree(d_invQ);
    cudaFree(d_error);
    cudaFree(d_res_dot);
}

/*********************************//*
 * compute deriv:
 *  
 ********************************/
void compute_deriv( cublasHandle_t handle, real *c_deriv, const real *c_theta_exp, 
        real *d_predict, real *d_train,
        const real *d_invQt, const real *d_dist_matrix_T,
        const int Npredict, const int Ntrain, const int Ninputs)
{
    int i;
    real alpha;
    real beta = 0.f;
 
    real *d_deriv, *d_aa;
    cudaMalloc((void **)&d_deriv, sizeof(real) * Npredict );
    cudaMalloc((void **)&d_aa, sizeof(real) * Ntrain * Npredict);
    real *ptr_train, *ptr_predict, *ptr_deriv;
    ptr_train = d_train;
    ptr_predict = d_predict;
    ptr_deriv = c_deriv;

     for( i = 0; i < Ninputs; ++i){
        gpu_crossMinus(ptr_train, ptr_predict, d_aa, Ntrain, Npredict );
        ptr_train = ptr_train + Ntrain;
        ptr_predict = ptr_predict + Npredict;
        alpha = c_theta_exp[i];
        gpu_elementwiseMult(d_dist_matrix_T, d_aa, Ntrain * Npredict);
        cublasCheckErrors(CUBLAS_GEMV(handle, CUBLAS_OP_T, Ntrain, Npredict, &alpha, d_aa, Ntrain, d_invQt, 1, &beta, d_deriv,1));

        cudaMemcpy(ptr_deriv, d_deriv, sizeof(real) * Npredict, cudaMemcpyDeviceToHost);
        ptr_deriv = ptr_deriv + Npredict;
     }

     cudaFree(d_deriv);
     cudaFree(d_aa);
}




 
/*********************************************//**
 * predict function:
 * CUDA version of the corresponding python code
 * GaussianPredict::predict
 *********************************************/
extern "C"{
void predict(const real *c_theta_exp, const real *c_train,const real *c_invQt,
        const real *c_invQ, const real *c_predict,  
        real *c_result, real *c_error, real *c_deriv,
        const int Npredict,const int Ntrain, const int  Ninputs, const int theta_size)
{
    int i;
    cublasHandle_t handle;
    cublasCreate(&handle);

    real *c_theta_exp_sqrt;
    c_theta_exp_sqrt = (real *)malloc( sizeof(real) * theta_size );
    for( i=0; i < theta_size; i++ )
    {
        c_theta_exp_sqrt[i] = sqrt( c_theta_exp[i] );
    }


    //define device vector and matrices
    real *d_train, *d_theta_exp, *d_theta_exp_sqrt, *d_invQt, *d_invQ, *d_predict;

    //allocate and copy vector on device 
    cudaMalloc( (void **)&d_theta_exp, sizeof(real) * theta_size );
    cudaMalloc( (void **)&d_theta_exp_sqrt, sizeof(real) * theta_size );
    cudaMalloc( (void **)&d_invQt, sizeof(real) * Ntrain);
    cublasCheckErrors(cublasSetVector( 
                theta_size, sizeof(real), c_theta_exp_sqrt, 1, d_theta_exp_sqrt, 1 ));
    cublasCheckErrors(cublasSetVector( 
                theta_size, sizeof(real), c_theta_exp, 1, d_theta_exp, 1 ));
    cublasCheckErrors(cublasSetVector( 
                Ntrain, sizeof(real), c_invQt, 1, d_invQt, 1));

    //allocate and copy matrix on device
    cudaMalloc( (void **)&d_train, sizeof(real) * Ntrain * Ninputs );
    cudaMalloc( (void **)&d_invQ, sizeof(real) * Ntrain * Ntrain );
    cudaMalloc( (void **)&d_predict, sizeof(real) * Npredict * Ninputs );
    cublasCheckErrors(cublasSetMatrix( 
                Ntrain, Ninputs, sizeof(real), c_train, Ntrain, d_train, Ntrain ));
    cublasCheckErrors(cublasSetMatrix( 
                Ntrain, Ntrain, sizeof(real), c_invQ, Ntrain, d_invQ, Ntrain ));
    cublasCheckErrors(cublasSetMatrix( 
                Npredict, Ninputs, sizeof(real), c_predict, Npredict, d_predict, Npredict));

    real *d_dist_matrix, *d_dist_matrix_T;
    
    d_dist_matrix = compute_distance(d_predict, d_train, 
            c_theta_exp, d_theta_exp, 
            d_theta_exp_sqrt, 
            Npredict, Ntrain, Ninputs);
    compute_result(handle, c_result, 
            d_dist_matrix, d_invQt, 
            Npredict, Ntrain, Ninputs);
    d_dist_matrix_T = gpu_transpose(handle, d_dist_matrix, Ntrain, Npredict);
    compute_error(handle, c_error, d_dist_matrix, 
            d_dist_matrix_T, d_invQ, c_theta_exp,
            Npredict, Ntrain, Ninputs); 
    compute_deriv( handle, c_deriv, c_theta_exp, 
            d_predict, d_train, d_invQt, d_dist_matrix_T,
            Npredict, Ntrain, Ninputs);
    
    cudaFree(d_invQt);
    cudaFree(d_dist_matrix_T);
    cudaFree(d_train);
    cudaFree(d_theta_exp);
    cudaFree(d_predict);
}
}
