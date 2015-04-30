#include "gpu_predict.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

/*********************************************//** 
 * vector matrix elementwise multiplication
 *********************************************/
__global__ 
void gpu_vectorTimesMatrixSqrt(const real *A, const real * v, real *res, int A_ld)
{
    int ix, iy;
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
//    ix=threadIdx.x;
//    iy=blockIdx.y;
    res[IDX2C(ix, iy, A_ld)] = sqrt(A[IDX2C(ix, iy, A_ld)] * v[ix]);
}


__global__
void gpu_cdist(const real *input1, const real *input2, real *output, int In1_ld, int In2_ld)
{
    int ix, iy, iz;
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    iz = blockIdx.z * blockDim.z + threadIdx.z;


    output[IDX2C(ix, iz, In1_ld)]+=pow(input1[IDX2C(iy, iz, In1_ld)] - input2[IDX2C(ix, iz, In2_ld)],2);
}


__global__
void gpu_init_zero(real *vec)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    vec[ix] = 0;
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
    real *d_a;//, *d_deriv;

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
    
    //allocate memory to results matrices
    cudaMalloc((void **)&d_a, sizeof(real) * M * N);

        
    /*********************
     *cdist
     *********************/
    real *d_res_temp1, *d_res_temp2;
    cudaMalloc((void **)&d_res_temp1, sizeof(real) * M * D);
    cudaMalloc((void **)&d_res_temp2, sizeof(real) * N * D);
    

    dim3 nthread(5,D);
    dim3 nblock(M/5,1);
    gpu_vectorTimesMatrixSqrt<<<M, D>>>(d_inputs, d_theta_exp, d_res_temp1, D);
    nthread.x=D; nthread.y=50;
    nblock.x=1; nblock.y=N/50;
    gpu_vectorTimesMatrixSqrt<<<nblock, nthread>>>(d_testing, d_theta_exp, d_res_temp2, D);
    
    

    nthread.x=1;   nthread.y=50;    nthread.z=D;
    nblock.x=N;    nblock.y=M/50;     nblock.z=1;

    
    gpu_init_zero<<<N*M/512,512>>>(d_a);
    gpu_cdist<<<nblock,nthread>>>(d_res_temp1, d_res_temp2, d_a, D, D);
    
    cudaFree(d_res_temp1);
    cudaFree(d_res_temp2);

    //gpu_cdist<<<1,threads>>>(d_testing,N,);    

    cublasCheckErrors(cublasGetMatrix (N, D, sizeof(real), d_res_temp2, N, c_testing, N));
   
    cudaFree(d_theta_exp);
    cudaFree(d_testing);
    //cudaFree(d_aa);
    //free(c_theta);
    //free((char*) c_inputs);
    //return Py_BuildValue ("i",1);
}
}
