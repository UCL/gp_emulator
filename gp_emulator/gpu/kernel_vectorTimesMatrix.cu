/*********************************************//** 
 * vector matrix elementwise multiplication
 * res_{ix,iy} = matrix_{ix,iy} * v_{iy}
 * matrix_lead_dim: leading dimension of matrix matrix
 * matrix_second_dim: secondary dimension of matrix matrix
 *********************************************/
#include "gpu_predict.h"
#define VTM_THREADX 100 //best thread number in x dim for vectorTimesMatrix
__global__ 
void kernel_vectorTimesMatrix(const real *matrix, const real * vector, real *res, int matrix_lead_dim, int matrix_second_dim)
{
    int ix, iy;
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    if( ix < matrix_lead_dim && iy < matrix_second_dim)
        res[IDX2D(ix, iy, matrix_lead_dim)] = matrix[IDX2D(ix, iy, matrix_lead_dim)] * vector[iy];
}


void gpu_vectorTimesMatrix(const real *matrix, const real *vector, real *res, int nrows, int ncols)
{
    dim3 nthread, nblock;
    if( nrows > MIN_NPREDICT )
    {
        nthread.x = VTM_THREADX;
        nthread.y = ncols;
        nblock.x = ceil(float(nrows)/VTM_THREADX);
        nblock.y = 1;
    }
    else
    {
        nthread.x = 1;
        nthread.y = ncols;
        nblock.x = nrows;
        nblock.y = 1;
    }
    kernel_vectorTimesMatrix<<<nblock, nthread>>>(matrix, vector, res, nrows, ncols);
}



