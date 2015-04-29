#include "gpu_predict.h"


extern "C"{
void predict(real *c_theta_exp, real *c_inputs,real *c_invQt,real *c_invQ, real *c_testing, int N, int NN, int  D, int theta_size)
{
    printf("start Gaussian process prediction: (N=%d,nn=%d,D=%d,theta_size=%d)\n",N,NN,D,theta_size);   

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat=cublasCreate(&handle);

    //define device vector and matrices
    real *d_inputs, *d_theta_exp, *d_invQt, *d_invQ, *d_testing;
    real *d_aa, *d_deriv;


    //allocate and copy vector on device 
    cudaMalloc( (void **)&d_theta_exp, sizeof(real) * theta_size );
    cudaMalloc( (void **)&d_invQt, sizeof(real) * NN);
    cublasCheckErrors(cublasSetVector( theta_size, sizeof(real), c_theta_exp, 1, d_theta_exp, 1 ));
    cublasCheckErrors(cublasSetVector( NN, sizeof(real), c_invQt, 1, d_invQt, 1));


    //allocate and copy matrix on device
    cudaMalloc( (void **)&d_inputs, sizeof(real) * NN * D );
    cudaMalloc( (void **)&d_invQ, sizeof(real) * NN * NN );
    cudaMalloc( (void **)&d_testing, sizeof(real) * N * D );
    cublasCheckErrors(cublasSetMatrix( NN, D, sizeof(real), c_inputs, NN, d_inputs, NN ));
    cublasCheckErrors(cublasSetMatrix( NN, NN, sizeof(real), c_invQ, NN, d_invQ, NN ));
    cublasCheckErrors(cublasSetMatrix( N, D, sizeof(real), c_testing, N, d_testing, N));
    



    
    dim3 threads(10,20); 
    gpu_cdist<<<1,threads>>>(d_testing);    

    cublasCheckErrors(cublasGetMatrix (N, D, sizeof(real), d_testing, N, c_testing, N));
   
    cudaFree(d_theta_exp);
    cudaFree(d_testing);
    cudaFree(d_aa);
    //free(c_theta);
    //free((char*) c_inputs);
    //return Py_BuildValue ("i",1);
}
}
