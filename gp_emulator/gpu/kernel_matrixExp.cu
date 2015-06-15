/*********************************************//**
 * Do the following operation:
 * - matrix_{i} = beta * e^{alpha * matrix_{i}}
 *********************************************/
#include "gpu_predict.h"
__global__
void kernel_matrixExp(real *matrix, const real alpha, const real beta, const int size)
{
    int i, index;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for( i = 0; i < CUDA_BLOCK; ++i )
    {
        index = ix * CUDA_BLOCK + i;
        if( index < size)
            matrix[ index ] = beta * exp( alpha * matrix[ index ]);
    }
}

void gpu_matrixExp( real *matrix,const real alpha,const real beta, const int size )
{
    int nthread, nblock;

    if( size < float(1024) / float(CUDA_BLOCK) )
    {
        printf("gpu_matrixExp: size = %d [ < 1024 / CUDA_BLOCK ].\n", size);
        exit(EXIT_FAILURE);
    }

    if( size > 65536 * 1024 )
    {
        printf("gpu_matrixExp: size = %d [ > 65536 * 1024 / CUDA_BLOCK ].\n", size);
        exit(EXIT_FAILURE);
    }
    
    nthread = 1024; 
    nblock = ceil( float(size) / float(CUDA_BLOCK) / float(nthread) );
    kernel_matrixExp<<<nblock,nthread>>>(matrix, alpha, beta, size);
}
