/*********************************************//** 
 * CUDA implementation of derived from GaussianProcess
 * in python code. 
 * 
 * Sinan Shi 
 *********************************************/
#include "gpu_predict.h"
#include <stdlib.h>
#define debug
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
    cublasCheckErrors(cublasSetVector( theta_size, sizeof(real), c_theta_exp_sqrt, 1, d_theta_exp_sqrt, 1 ));
    cublasCheckErrors(cublasSetVector( theta_size, sizeof(real), c_theta_exp, 1, d_theta_exp, 1 ));
    cublasCheckErrors(cublasSetVector( Ntrain, sizeof(real), c_invQt, 1, d_invQt, 1));

    //allocate and copy matrix on device
    cudaMalloc( (void **)&d_train, sizeof(real) * Ntrain * Ninputs );
    cudaMalloc( (void **)&d_invQ, sizeof(real) * Ntrain * Ntrain );
    cudaMalloc( (void **)&d_predict, sizeof(real) * Npredict * Ninputs );
    cublasCheckErrors(cublasSetMatrix( Ntrain, Ninputs, sizeof(real), c_train, Ntrain, d_train, Ntrain ));
    cublasCheckErrors(cublasSetMatrix( Ntrain, Ntrain, sizeof(real), c_invQ, Ntrain, d_invQ, Ntrain ));
    cublasCheckErrors(cublasSetMatrix( Npredict, Ninputs, sizeof(real), c_predict, Npredict, d_predict, Npredict));
    

        
    /*********************************
     * Euclidian distance calculation
     * a = cidist(sqrt(expX_{,Ninputs}) * inputs * sqrt(expX_{,Ninputs} * testing)))
     * Npredictotice: a is equivalent to a^T in python due to column major fashion
     ********************************/ 
    real *d_res_temp1, *d_res_temp2, *d_a;
    cudaMalloc((void **)&d_res_temp1, sizeof(real) * Ntrain * Ninputs);
    cudaMalloc((void **)&d_res_temp2, sizeof(real) * Npredict * Ninputs);
    cudaMalloc((void **)&d_a, sizeof(real) * Ntrain * Npredict);

    gpu_vectorTimesMatrix(d_train, d_theta_exp_sqrt, d_res_temp1, Ntrain, Ninputs);
    gpu_vectorTimesMatrix(d_predict, d_theta_exp_sqrt, d_res_temp2, Npredict, Ninputs);
    gpu_init_array( d_a, 0.0, Npredict * Ntrain );
    gpu_cdist(d_res_temp1, d_res_temp2, d_a, Ntrain, Ninputs, Npredict, Ninputs);
    gpu_matrixExp(d_a, -0.5, c_theta_exp[Ninputs], Ntrain * Npredict);

    cudaFree(d_res_temp1);
    cudaFree(d_res_temp2);

    /*********************************
     * compute mu:
     * mu = a * invQt (dot product)
     ********************************/
    real *d_result;
    cudaMalloc((void **)&d_result, sizeof(real) * Npredict);
    real alpha = 1.f;
    real beta = 0.f;
    /*cublasCheckErrors(*/CUBLAS_GEMV(handle, CUBLAS_OP_N, Npredict, Ntrain, &alpha, d_a, Npredict, d_invQt, 1, &beta, d_result, 1);//);
    cudaMemcpy(c_result, d_result, sizeof(real) * Npredict, cudaMemcpyDeviceToHost);
       
    
   /*********************************
    * compute var:
    * var = b - rowsum(a * dot(invQ, a_T))
    ********************************/
    real *d_temp_dot, *d_a_T;
    real *d_error;

    cudaMalloc((void **)&d_temp_dot, sizeof(real) * Ntrain * Npredict);
    cudaMalloc((void **)&d_a_T, sizeof(real) * Ntrain * Npredict);

    cublasCheckErrors(CUBLAS_GEMM(handle, CUBLAS_OP_N, CUBLAS_OP_T, Ntrain, Npredict, Ntrain, &alpha, d_invQ, Ntrain, d_a, Npredict, &beta, d_temp_dot, Ntrain));
    cublasCheckErrors(CUBLAS_GEAM(handle, CUBLAS_OP_T, CUBLAS_OP_N, Ntrain, Npredict, &alpha, d_a, Npredict, &beta, d_a, Ntrain, d_a_T, Ntrain));
    
    cudaFree(d_a);
    cudaFree(d_invQ);
    
    gpu_elementwiseMult(d_a_T, d_temp_dot, Ntrain * Npredict);
    d_error = gpu_rowSum(d_temp_dot, Ntrain, Npredict);
    gpu_scalarMinusVec(d_error, c_theta_exp[Ninputs], Npredict );
    cudaMemcpy(c_error, d_error, sizeof(real) * Npredict, cudaMemcpyDeviceToHost);

    cudaFree(d_error);
    cudaFree(d_temp_dot);


   /*********************************
    * compute deriv:
    * 
    ********************************/
    real *d_deriv;
    cudaMalloc((void **)&d_deriv, sizeof(real) * Npredict );
   
    real *d_aa;
    cudaMalloc((void **)&d_aa, sizeof(real) * Ntrain * Npredict);
    real *ptr_inputs, *ptr_testing, *ptr_deriv;
    ptr_inputs = d_train;
    ptr_testing = d_predict;
    ptr_deriv = c_deriv;

     for( i = 0; i < Ninputs; ++i){
        gpu_crossMinus(ptr_inputs, ptr_testing, d_aa, Ntrain, Npredict );
        ptr_inputs = ptr_inputs + Ntrain;
        ptr_testing = ptr_testing + Npredict;
        alpha = c_theta_exp[i];
        gpu_elementwiseMult(d_a_T, d_aa, Ntrain * Npredict);
        cublasCheckErrors(CUBLAS_GEMV(handle, CUBLAS_OP_T, Ntrain, Npredict, &alpha, d_aa, Ntrain, d_invQt, 1, &beta, d_deriv,1));

        cudaMemcpy(ptr_deriv, d_deriv, sizeof(real) * Npredict, cudaMemcpyDeviceToHost);
        ptr_deriv = ptr_deriv + Npredict;
     }

     cudaFree(d_result);
     cudaFree(d_invQt);
     cudaFree(d_a_T);
     cudaFree(d_aa);
     cudaFree(d_train);
     cudaFree(d_deriv);
     cudaFree(d_theta_exp);
     cudaFree(d_predict);
   
}
}
