/*********************************************//** 
 * CUDA implementation of derived from GaussianProcess
 * in python code. 
 * 
 * Sinan Shi (UCL) 
 *********************************************/
#include "gpu_predict.h"
#include <stdlib.h>
#define debug 


/*********************************************//** 
 * Do the following operation:
 * - matrix_{i} = beta * e^{alpha * matrix_{i}}
 *********************************************/
__global__
void gpu_matrixExp(real *matrix, real alpha, real beta)
{
    int i;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for( i = 0; i < CUDA_BLOCK; ++i )
        matrix[ix * CUDA_BLOCK + i] = beta * exp( alpha * matrix[ix * CUDA_BLOCK + i]);
}

/*********************************************//** 
 * vector elementwise multiplication
 *********************************************/
__global__
void gpu_elementwiseMult(const real *v1, real *v2, const int size)
{
    int i, index;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for( i = 0; i < CUDA_BLOCK; ++i )
    {
        index = ix * CUDA_BLOCK + i;
        if( index < size)
        {
            v2[index] = v2[index] * v1[index];
        }
    }


}

/*********************************************//** 
 * vector scalar operation, update vec by,
 * vec = scalar - vec
 *********************************************/
__global__
void gpu_scalarMinusVec(real *vec, const real scalar)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    vec[ix] = scalar - vec[ix];

}

/*********************************************//**
 * row sum:
 * return vector rowSum(A)
 *********************************************/
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
   
    gpu_init_array(vec_one, 1.0, A_ncols);
    gpu_init_array(d_var, 0.0, A_ncols);

    cublasCheckErrors(CUBLAS_GEMV(handle, CUBLAS_OP_T, A_nrows, A_ncols, &alpha, A, A_nrows, vec_one, 1, &beta, d_var, 1));
    
    cudaFree(vec_one);
    cublasDestroy(handle);
    return d_var;
}

/*********************************************//**
 * getAa
 * aa_{ix, iy} = inputs_{ix} - testing_{iy} 
 *********************************************/
__global__
void gpu_getAa(const real *inputs,const real *testing, real *aa, const int aa_nrows, const int aa_ncols, const int aa_ld)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    aa[IDX2D(ix, iy, aa_ld)] = inputs[ix] - testing[iy];
}




/*********************************************//**
 * predict function:
 * This is the corresponding CUDA function of
 * GaussianPredict:predict in python code.
 *********************************************/
extern "C"{
void predict(const real *c_theta_exp, const real *c_inputs,const real *c_invQt,const real *c_invQ, const real *c_testing,  
        real *c_mu, real *c_var, real *c_deriv,const int N,const int M, const int  D, const int theta_size)
{
    int i,j;
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
    

        
    /*********************************
     * Euclidian distance calculation
     * a = cidist(sqrt(expX_{,D}) * inputs * sqrt(expX_{,D} * testing)))
     * Notice: a is equivalent to a^T in python due to column major fashion
     ********************************/ 
    dim3 nthread, nblock;
    real *d_res_temp1, *d_res_temp2, *d_a;
    cudaMalloc((void **)&d_res_temp1, sizeof(real) * M * D);
    cudaMalloc((void **)&d_res_temp2, sizeof(real) * N * D);
    cudaMalloc((void **)&d_a, sizeof(real) * M * N);
   

    gpu_vectorTimesMatrix(d_inputs, d_theta_exp_sqrt, d_res_temp1, M, D);
    gpu_vectorTimesMatrix(d_testing, d_theta_exp_sqrt, d_res_temp2, N, D);
    gpu_init_array( d_a, 0.0, N * M );
    gpu_cdist(d_res_temp1, d_res_temp2, d_a, M, D, N, D);

    
    nthread.x=1000; nthread.y=1; nthread.z=1;
    nblock.x=ceil(float(N) * float(M) / float(CUDA_BLOCK)/1000); nblock.y=1; nblock.z=1;
    gpu_matrixExp<<<nblock,nthread>>>(d_a, -0.5, c_theta_exp[D]);

    cudaFree(d_res_temp1);
    cudaFree(d_res_temp2);
int kk;
#undef debug
#ifdef debug
real *h_a;
h_a=(real *)malloc(sizeof(real) * M * N);
cudaMemcpy(h_a, d_a, sizeof(real) * M * N, cudaMemcpyDeviceToHost);

for(kk =0; kk<10; kk++)
  printf("%f|", h_a[kk]);
  printf("\n");
#endif



      

    /*********************************
     * compute mu:
     * mu = a * invQt (dot product)
     ********************************/
    real *d_mu;
    cudaMalloc((void **)&d_mu, sizeof(real) * N);
    real alpha = 1.f;
    real beta = 0.f;
    cublasCheckErrors(CUBLAS_GEMV(handle, CUBLAS_OP_N, N, M, &alpha, d_a, N, d_invQt, 1, &beta, d_mu, 1));
    cudaMemcpy(c_mu, d_mu, sizeof(real) * N, cudaMemcpyDeviceToHost);
       
    
   /*********************************
    * compute var:
    * var = b - rowsum(a * dot(invQ, a_T))
    ********************************/
    real *d_temp_dot, *d_a_T;
    real *d_var;

    cudaMalloc((void **)&d_temp_dot, sizeof(real) * M * N);
    cudaMalloc((void **)&d_a_T, sizeof(real) * M * N);

    cublasCheckErrors(CUBLAS_GEMM(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, M, &alpha, d_invQ, M, d_a, N, &beta, d_temp_dot, M));
    cublasCheckErrors(CUBLAS_GEAM(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &alpha, d_a, N, &beta, d_a, M, d_a_T, M));
    cudaFree(d_a);
    cudaFree(d_invQ);
    gpu_elementwiseMult<<< nblock, nthread >>>(d_a_T, d_temp_dot, M * N);

    d_var = gpu_rowSum(d_temp_dot, M, N);
    
    gpu_scalarMinusVec<<<N/1000, 1000 >>>(d_var, c_theta_exp[D]);
    cudaMemcpy(c_var, d_var, sizeof(real) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_var);
    cudaFree(d_temp_dot);


   /*********************************
    * compute deriv:
    * 
    ********************************/
    real *d_deriv;
    cudaMalloc((void **)&d_deriv, sizeof(real) * N );

    dim3 nblocks_getaa(M, N/1000);
    dim3 nthreads_getaa(1,1000);
    real *d_aa;
    cudaMalloc((void **)&d_aa, sizeof(real) * M * N);
    real *ptr_inputs, *ptr_testing, *ptr_deriv;
    ptr_inputs = d_inputs;
    ptr_testing = d_testing;
    ptr_deriv = c_deriv;

     for( i = 0; i < D; ++i){
        gpu_getAa <<< nblocks_getaa, nthreads_getaa >>>(ptr_inputs, ptr_testing, d_aa, M, N, M);
        ptr_inputs = ptr_inputs + M;
        ptr_testing = ptr_testing + N;
        alpha = c_theta_exp[i];
        gpu_elementwiseMult <<< nblock, nthread >>>(d_a_T, d_aa, M * N);
        cublasCheckErrors(CUBLAS_GEMV(handle, CUBLAS_OP_T, M, N, &alpha, d_aa, M, d_invQt, 1, &beta, d_deriv,1));

        cudaMemcpy(ptr_deriv, d_deriv, sizeof(real) * N, cudaMemcpyDeviceToHost);
#define debug
#undef debug
#ifdef debug
        for( j = 0; j < 10 ; ++j )
            printf("%.4f|", ptr_deriv[j]);
        printf("\n");
#endif
        ptr_deriv = ptr_deriv + N;

     }

     cudaFree(d_mu);
     cudaFree(d_invQt);
     cudaFree(d_a_T);
     cudaFree(d_aa);
     cudaFree(d_inputs);
     cudaFree(d_deriv);
     cudaFree(d_theta_exp);
     cudaFree(d_testing);
   
}
}
