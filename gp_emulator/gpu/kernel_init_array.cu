/*********************************************//** 
 * initialise an array [real] with an assigned 
 * value init_val. 
 *********************************************/
#include "gpu_predict.h"
__global__
void kernel_init_array(real *vec, const real init_val, const int vec_len)
{
    int i, index;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for( i = 0; i < CUDA_BLOCK; ++i )
    {
        index = ix * CUDA_BLOCK + i;
        if( index < vec_len)
        {
            vec[index] = init_val;
        }
    }
}

void gpu_init_array(real *vec, const int init_val, const int vec_len)
{
    int nthread, nblock;

    if( CUDA_BLOCK * vec_len < 1024)
    {
        nthread = ceil( vec_len / CUDA_BLOCK );
        nblock = 1;
    }
    else
    {
        if( vec_len / CUDA_BLOCK > 65536 * 1024 ) // largest block size for sm_2.0
        {
            printf("gpu_init_array: vector length / CUDA_BLOCK = %d > 1024 * 1024.\n", vec_len/CUDA_BLOCK);
            exit(EXIT_FAILURE);
        }
        nthread = 1024;
        nblock = ceil( float(vec_len) / nthread / float(CUDA_BLOCK) );
    }
    kernel_init_array<<< nblock, nthread >>>(vec, init_val, vec_len);
}




