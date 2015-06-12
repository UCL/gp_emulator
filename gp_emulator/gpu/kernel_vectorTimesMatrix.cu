/*********************************************//** 
 * vector matrix elementwise multiplication
 * res_{ix,iy} = A_{ix,iy} * v_{iy}
 * A_ld: leading dimension of matrix A
 * A_sd: secondary dimension of matrix A
 *********************************************/
#include "gpu_predict.h"
__global__ 
void kernel_vectorTimesMatrix(const real *A, const real * v, real *res, int A_ld, int A_sd)
{
    int ix, iy;
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    if( ix < A_ld && iy < A_sd)
    res[IDX2D(ix, iy, A_ld)] = A[IDX2D(ix, iy, A_ld)] * v[iy];
}


extern "C"{
void gpu_vectorTimesMatrix(const real *A, const real *v, real *res, int nrows, int ncols)
{
    dim3 nthread, nblock;
    if( nrows > 1000 )
    {
        nthread.x = 100;
        nthread.y = ncols;
        nblock.x = ceil(float(nrows)/100);
        nblock.y = 1;
    }
    else
    {
        nthread.x = 1;
        nthread.y = ncols;
        nblock.x = nrows;
        nblock.y = 1;
    }
    kernel_vectorTimesMatrix<<<nblock, nthread>>>(A, v, res, nrows, ncols);
}
}



