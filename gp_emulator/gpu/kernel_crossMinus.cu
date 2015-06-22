#include "gpu_predict.h"
/*********************************************//**
 * cross minus:
 * aa_{ix, iy} = inputs_{ix} - testing_{iy}
 *********************************************/
__global__
void kernel_crossMinus(const real *v1,const real *v2, real *mat_res, const int v1_len, const int v2_len)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < v1_len && iy < v2_len)
        mat_res[IDX2D(ix, iy, v1_len)] = v1[ix] - v2[iy];
}



void gpu_crossMinus(const real *v1, const real *v2, real *mat_res, const int v1_len, const int v2_len)
{
    dim3 nthread, nblock;
    if( v2_len < MAX_NUM_THREAD )
        nthread.y = v2_len;
    else
        nthread.y = MAX_NUM_THREAD;
    
    nthread.x = 1;
    nblock.x = v1_len;
    nblock.y = ceil( float(v2_len) / float(nthread.y) );

    kernel_crossMinus<<< nblock, nthread >>>(v1, v2, mat_res, v1_len, v2_len);
}




