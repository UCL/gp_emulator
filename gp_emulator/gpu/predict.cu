#include "gpu_predict.h"
#include <stdlib.h>
//#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IDX2D(i,j,ld) (((j)*(ld))+(i))  //!! keep it in column major for coping with cublas column major fashion.
#define debug 
// x -> i -> col
// y -> j -> row
// leading dimension should always be column
/*********************************************//** 
 * vector matrix elementwise multiplication
 *********************************************/
__global__ 
void gpu_vectorTimesMatrix(const real *A, const real * v, real *res, int A_ld)
{
    int ix, iy;
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    res[IDX2D(ix, iy, A_ld)] = A[IDX2D(ix, iy, A_ld)] * v[iy];
}


// ix -> M; iy -> N; iz -> D

__global__
void gpu_cdist(const real *input1, const real *input2, real *output, int In1_ld, int In2_ld, int Out_ld, int D)
{
    int ix, iy, iz;
    ix = blockIdx.x * blockDim.x + threadIdx.x;//N
    iy = blockIdx.y * blockDim.y + threadIdx.y;//M
    iz = blockIdx.z * blockDim.z + threadIdx.z;
    output[IDX2D(ix, iy, Out_ld)] += pow(input1[IDX2D(iy, iz, In1_ld)] - input2[IDX2D(ix, iz, In2_ld)],2);
}


__global__
void gpu_init_array(real *vec, const real init_val)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    vec[ix] = init_val;
}
// further tests needed: ix exceed M*N
__global__
void gpu_matrixExp(real *matrix, real alpha, real beta)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    matrix[ix] = beta * exp( alpha * matrix[ix]);
}


void computeTranspose(real *matrix, const  int size_in, const  int size_out)
{
    real * temp;
    temp = ( real *)malloc(sizeof(real) * size_in * size_out);

    for ( int i = 0; i < size_in * size_out; ++i)
        temp[i] = matrix[i];

    for (int y = 0; y < size_out; ++y)
    {
        for (int x = 0; x < size_in; ++x)
        {
            matrix[(x * size_out) + y] = temp[(y * size_in) + x];                                                                

        }   
    }
}

__global__
void gpu_elementwiseMult(const real *v1, real *v2)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    v2[ix] = v2[ix] * v1[ix];
}


__global__
void gpu_scalarMinusVec(real *vec, const real scalar)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    vec[ix] = scalar - vec[ix];
}


real* gpu_rowSum(const real *A, const int A_nrows,const int A_ncols)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat=cublasCreate(&handle);
    
    real alpha = 1.f;
    real beta = 0.f;
    real *vec_one;
    real *d_var;
    
    cudaMalloc((void **)&vec_one, sizeof(real) * A_ncols );
    cudaMalloc((void **)&d_var, sizeof(real) * A_ncols);
    
    gpu_init_array<<< ceil(float(A_ncols)/512), 512 >>>(vec_one, 1);
    gpu_init_array<<< ceil(float(A_ncols)/512),512 >>>(d_var, 0);

    cublasCheckErrors(cublasDgemv(handle, CUBLAS_OP_T, A_nrows, A_ncols, &alpha, A, A_nrows, vec_one, 1, &beta, d_var, 1));
    
    cudaFree(vec_one);
    cublasDestroy(handle);
    return d_var;
}

extern "C"{
void predict(real *c_theta_exp, real *c_inputs,real *c_invQt,real *c_invQ, real *c_testing,  int N, int M, int  D, int theta_size)
{
    printf("start Gaussian process prediction: (N=%d,nn=%d,D=%d,theta_size=%d)\n",N,M,D,theta_size);   
    int i;
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    stat=cublasCreate(&handle);

    real *c_theta_exp_sqrt;
    c_theta_exp_sqrt = (real *)malloc( sizeof(real) * theta_size );
    for( i=0; i < theta_size; i++ )
    {
        c_theta_exp_sqrt[i] = sqrt( c_theta_exp[i] );
    }


    //define device vector and matrices
    real *d_inputs, *d_theta_exp, *d_theta_exp_sqrt, *d_invQt, *d_invQ, *d_testing;
    real *d_a;//, *d_deriv;


    //allocate and copy vector on device 
    cudaMalloc( (void **)&d_theta_exp, sizeof(real) * theta_size );
    cudaMalloc( (void **)&d_theta_exp_sqrt, sizeof(real) * theta_size );
    cudaMalloc( (void **)&d_invQt, sizeof(real) * M);
    cublasCheckErrors(cublasSetVector( theta_size, sizeof(real), c_theta_exp_sqrt, 1, d_theta_exp_sqrt, 1 ));
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
    


    dim3 nthread(1,D);
    dim3 nblock(M,1);
    gpu_vectorTimesMatrix<<<nblock, nthread>>>(d_inputs, d_theta_exp_sqrt, d_res_temp1, M);
    nthread.x=1; nthread.y=D;
    nblock.x=N; nblock.y=1;
    gpu_vectorTimesMatrix<<<nblock, nthread>>>(d_testing, d_theta_exp_sqrt  , d_res_temp2, N);
    gpu_init_array<<<ceil(float(N)*float(M)/512),512>>>(d_a, 0);

    nthread.x=1;   nthread.y=5;    nthread.z=1;
    nblock.x=N;    nblock.y=M/5;     nblock.z=D;
    gpu_cdist<<<nblock,nthread>>>(d_res_temp1, d_res_temp2, d_a, M, N, N, D);

    gpu_matrixExp<<<ceil(float(M)*float(N)/512),512>>>(d_a, -0.5, c_theta_exp[D]);
   
    
    real *d_mu;
    cudaMalloc((void **)&d_mu, sizeof(real) * N);
    real alpha = 1.f;
    real beta = 0.f;

   // if( sizeof(real) == sizeof(float) )
   //     cublasCheckErrors(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, d_a, N, d_invQt, 1, &beta, d_mu, 1));
    if( sizeof(real) == sizeof(double) )
        cublasCheckErrors(cublasDgemv(handle, CUBLAS_OP_N, N, M, &alpha, d_a, N, d_invQt, 1, &beta, d_mu, 1));
    
    real *temp_dot, *d_a_T;
    cudaMalloc((void **)&temp_dot, sizeof(real) * M * N);
    cudaMalloc((void **)&d_a_T, sizeof(real) * M * N);

    cublasCheckErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, M, &alpha, d_invQ, M, d_a, N, &beta, temp_dot, M));
    cublasCheckErrors(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &alpha, d_a, N, &beta, d_a, M, d_a_T, M));
    gpu_elementwiseMult<<< N, M >>>(d_a_T, temp_dot);
    real *d_var;
    d_var = gpu_rowSum(temp_dot, M, N);
    gpu_scalarMinusVec<<< N, M >>>(d_var, c_theta_exp[D]);







#define debug            
#ifdef debug
    real *temp_;
    temp_ = (real *)malloc( sizeof(real) * N * M);
    //cublasGetMatrix(M, N, sizeof(real), temp_dot, M, temp_,M);
    cudaMemcpy(temp_, d_var, sizeof(real)* N, cudaMemcpyDeviceToHost);
//    computeTranspose(temp_, M, N);
    printf("b=%f\n", c_theta_exp[D]);
    for( i = 0; i < 10 ; ++i )
       printf("%.4f|", temp_[i]);
#endif
    
    cudaFree(d_var); 
    cudaFree(d_theta_exp);
    cudaFree(d_testing);
}
}
