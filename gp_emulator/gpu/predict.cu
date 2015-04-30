#include "gpu_predict.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

/*********************************************//** 
 * vector matrix elementwise multiplication
 *********************************************/
__global__ 
void gpu_vectorTimesMatrix(real *A, const real * v,/* real *res,*/ int A_ld, int v_len)
{
    int ix, iy;
   // ix = blockIdx.x * blockDim.x + threadIdx.x;
   // iy = blockIdx.y * blockDim.y + threadIdx.y;
    ix = threadIdx.x;
    iy = blockIdx.x;

    A[IDX2C(ix, iy, A_ld)] = A[IDX2C(ix, iy, A_ld)] * v[ix];
}


extern "C"{
void predict(real *c_theta_exp, real *c_inputs,real *c_invQt,real *c_invQ, real *c_testing, int N, int M, int  D, int theta_size)
{
    printf("start Gaussian process prediction: (N=%d,nn=%d,D=%d,theta_size=%d)\n",N,M,D,theta_size);   

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat=cublasCreate(&handle);

    //define device vector and matrices
    real *d_inputs, *d_theta_exp, *d_invQt, *d_invQ, *d_testing;
    real *d_aa, *d_deriv;


    //allocate and copy vector on device 
    cudaMalloc( (void **)&d_theta_exp, sizeof(real) * theta_size );
    cudaMalloc( (void **)&d_invQt, sizeof(real) * M);
    cublasCheckErrors(cublasSetVector( theta_size, sizeof(real), c_theta_exp, 1, d_theta_exp, 1 ));
    cublasCheckErrors(cublasSetVector( M, sizeof(real), c_invQt, 1, d_invQt, 1));


    //allocate and copy matrix on device
    cudaMalloc( (void **)&d_inputs, sizeof(real) * M * D );
    cudaMalloc( (void **)&d_invQ, sizeof(real) * M * M );
    cudaMalloc( (void **)&d_testing, sizeof(real) * N * D );
    cublasCheckErrors(cublasSetMatrix( M, D, sizeof(real), c_inputs, M, d_inputs, M ));
    cublasCheckErrors(cublasSetMatrix( M, M, sizeof(real), c_invQ, M, d_invQ, M ));
    cublasCheckErrors(cublasSetMatrix( N, D, sizeof(real), c_testing, N, d_testing, N));
    


        
    dim3 threads(D, N);
    gpu_vectorTimesMatrix<<<N, D>>>(d_testing, d_theta_exp, D, D);

    //gpu_cdist<<<1,threads>>>(d_testing,N,);    

    cublasCheckErrors(cublasGetMatrix (N, D, sizeof(real), d_testing, N, c_testing, N));
   
    cudaFree(d_theta_exp);
    cudaFree(d_testing);
    cudaFree(d_aa);
    //free(c_theta);
    //free((char*) c_inputs);
    //return Py_BuildValue ("i",1);
}
}
