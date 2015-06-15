/*********************************************//**
 * vector elementwise multiplication
 * v2_{i} = v2_{i} * v1_{i}
 *********************************************/
#include "gpu_predict.h"
__global__
void kernel_elementwiseMult(const real *v1, real *v2, const int size)
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


void gpu_elementwiseMult( const real *v1, real *v2, const int size )
{
    int nthread, nblock;

    if( size < float(1024) / float(CUDA_BLOCK) )
    {
        printf("gpu_elementwiseMult: size = %d [ < 1024 / CUDA_BLOCK ].\n", size);
        exit(EXIT_FAILURE);
    }

    if( size > 65536 * 1024 )
    {
        printf("gpu_elementwiseMult: size = %d [ > 65536 * 1024 / CUDA_BLOCK ].\n", size);
        exit(EXIT_FAILURE);
    }

    nthread = 1024;
    nblock = ceil( float(size) / float(CUDA_BLOCK) / float(nthread) );
    kernel_elementwiseMult<<<nblock,nthread>>>(v1, v2, size);
}
