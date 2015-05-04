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
    //for (iz = 0; iz < D; iz++)
    //{
        output[IDX2D(ix, iy, Out_ld)] += pow(input1[IDX2D(iy, iz, In1_ld)] - input2[IDX2D(ix, iz, In2_ld)],2);
    //}

}


__global__
void gpu_init_zero(real *vec)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    vec[ix] = 0;
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
    

// check result of gpu_vectorTimesMatrix
#undef debug
#ifdef debug

#endif


    gpu_init_zero<<<ceil(float(N)*float(M)/512),512>>>(d_a);
//#undef debug
#ifdef debug
    real * debug_zero;
    debug_zero = (real *)malloc(sizeof(real) * N * M);
    cublasCheckErrors(cublasGetMatrix(N, M, sizeof(real), d_a, N, debug_zero, N));  
    for( i = 0; i < M * N; i ++)
    {
        if( debug_zero[i] != 0 )
        {
            printf( "[ERROR!] : gpu_init_zero.\n" );
            exit ( EXIT_FAILURE );            
        }

    }
    free( debug_zero );

#endif
    nthread.x=1;   nthread.y=M;    nthread.z=1;
    nblock.x=N;    nblock.y=1;     nblock.z=D;

    gpu_cdist<<<nblock,nthread>>>(d_res_temp1, d_res_temp2, d_a, M, N, N, D);

#define debug
#ifdef debug 
    real *debug_res_temp1, *debug_res_temp2;
    debug_res_temp1 = (real *)malloc(sizeof(real) * M * D);
    debug_res_temp2 = (real *)malloc(sizeof(real) * N * D);
    cublasCheckErrors(cublasGetMatrix(M, D, sizeof(real), d_res_temp1, M, debug_res_temp1, M));  
    cublasCheckErrors(cublasGetMatrix(N, D, sizeof(real), d_res_temp2, N, debug_res_temp2, N));

    real *temp_c_a;
    temp_c_a = (real *)malloc( sizeof(real) * N * M);
    cudaMemcpy(temp_c_a, d_a, sizeof(real) * N * M, cudaMemcpyDeviceToHost);


    computeTranspose(debug_res_temp1, M, D);
    computeTranspose(debug_res_temp2, N, D);
    // printf("res_temp1\n");
    // for( i = 0 ; i < M * D; i ++)
    // {
    //     printf( "%.4f|", debug_res_temp1[i] );
    //     if( i % 15 == 0 )
    //         printf( "\n" );
    // }
    // printf("\nres_temp2\n");
    // for( i = 0; i < N * D; i ++)
    // {
    //     printf( "%.4f|", debug_res_temp2[i] );
    //     if( i % 15 == 0 )
    //         printf( "\n" );
    // }
    // printf("\n");


    //computeTranspose( temp_c_a, N, M );
    for( i = 0; i < M * N; i++ )
    {
        if(i%15==0)
            printf("\n");
        printf("%.4f|", temp_c_a[i]);
    }
    free( temp_c_a );
    free(debug_res_temp1);
    free(debug_res_temp2);

#endif

    
   // cudaFree(d_res_temp1);
   // cudaFree(d_res_temp2);

    //gpu_cdist<<<1,threads>>>(d_testing,N,);    
   

       
    
//    free(temp_c_a);
    cudaFree(d_theta_exp);
    cudaFree(d_testing);

    //cudaFree(d_aa);
    //free(c_theta);
    //free((char*) c_inputs);
    //return Py_BuildValue ("i",1);
}
}
