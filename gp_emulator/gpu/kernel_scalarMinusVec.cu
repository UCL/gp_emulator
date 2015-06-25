/*********************************************//**
 * vector scalar operation, update vec by,
 * vec = scalar - vec
 *********************************************/
#include "gpu_predict.h"
__global__
void kernel_scalarMinusVec(real *vec, const real scalar, const int size)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if( ix < size )
        vec[ix] = scalar - vec[ix];
}


void gpu_scalarMinusVec( real *matrix, const real scalar, const int size )
{
    int nthread, nblock;
    if( size > MAX_NUM_BLOCK * MAX_NUM_THREAD )
    {
        printf("gpu_scalarMinusVec: size = %d [ > MAX_NUM_BLOCK * MAX_NUM_THREAD ].\n", size);
        exit(EXIT_FAILURE);
    }
    
    if( size < MAX_NUM_THREAD )
    {
        nthread = size;
        nblock = 1;
    }
    else
    {
        nthread = MAX_NUM_THREAD;
        nblock = ceil( float(size) / float(nthread) );
    }

    kernel_scalarMinusVec<<<nblock,nthread>>>(matrix, scalar, size);
}


