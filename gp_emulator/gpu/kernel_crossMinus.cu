#include "gpu_predict.h"
/*********************************************//**
 * cross minus:
 * aa_{ix, iy} = inputs_{ix} - testing_{iy}
 *********************************************/
__global__
void kernel_crossMinus(const real *vec1,const real *vec2, real *matrix_result, const int vec1_len, const int vec2_len)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < vec1_len && iy < vec2_len)
        matrix_result[IDX2D(ix, iy, vec1_len)] = vec1[ix] - vec2[iy];
}



void gpu_crossMinus(const real *vec1, const real *vec2, real *matrix_result, const int vec1_len, const int vec2_len)
{
    dim3 nthread, nblock;
    if( vec2_len < MAX_NUM_THREAD )
        nthread.y = vec2_len;
    else
        nthread.y = MAX_NUM_THREAD;
    
    nthread.x = 1;
    nblock.x = vec1_len;
    nblock.y = ceil( float(vec2_len) / float(nthread.y) );

    kernel_crossMinus<<< nblock, nthread >>>(vec1, vec2, matrix_result, vec1_len, vec2_len);
}




