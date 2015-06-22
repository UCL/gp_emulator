/*********************************************//**
 * vector elementwise multiplication
 * vector2_{i} = vector2_{i} * vector1_{i}
 *********************************************/
#include "gpu_predict.h"
__global__
void kernel_elementwiseMult(const real *vector1, real *vector2, const int size)
{
    int i, index;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for( i = 0; i < CUDA_BLOCK; ++i )
    {
        index = ix * CUDA_BLOCK + i;
        if( index < size)
        {
            vector2[index] = vector2[index] * vector1[index];
        }
    }
}


void gpu_elementwiseMult( const real *vector1, real *vector2, const int size )
{
    int nthread, nblock;

    if( size < float(MAX_NUM_THREAD) / float(CUDA_BLOCK) )
    {
        printf("gpu_elementwiseMult: size = %d [ < MAX_NUM_THREAD / CUDA_BLOCK ].\n", size);
        exit(EXIT_FAILURE);
    }

    nthread = MAX_NUM_THREAD;
    nblock = ceil( float(size) / float(CUDA_BLOCK) / float(nthread) );
    
    if( nblock > MAX_NUM_BLOCK )
    {
        printf("gpu_elementwiseMult: nblock outside the range of [1, MAX_NUM_BLOCK]\n");
        exit(EXIT_FAILURE);
    }
    
    kernel_elementwiseMult<<<nblock,nthread>>>(vector1, vector2, size);
}
