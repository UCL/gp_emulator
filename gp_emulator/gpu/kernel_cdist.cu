/*********************************************//** 
 * Squared Euclidiean distance function: 
 * - Equivalent to scipy cdist() function
 * - In1_ld and In2_ld, are leading dimention of the input1 and input2, 
 *   in our case it should be always the column. 
 * 
 *********************************************/
#include "gpu_predict.h"
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define CDIST_NTHREAD_X 256
#define CDIST_NTHREAD_Y 4
#define CDIST_NTHREAD_Z 1

__global__
void kernel_cdist(const real *input1, const real *input2, real *output, const int nrow1, const int nrow2, const int ncol)
{
    int ix, iy, iz;
    ix = blockIdx.x * blockDim.x + threadIdx.x;//N
    iy = blockIdx.y * blockDim.y + threadIdx.y;//M
    iz = blockIdx.z * blockDim.z + threadIdx.z;
    if( ix < nrow2 && iy < nrow1 && iz < ncol)
        output[IDX2D(ix, iy, nrow2)] += pow(input1[IDX2D(iy, iz, nrow1)] - input2[IDX2D(ix, iz, nrow2)],2);
}


void gpu_cdist(const real *input1, const real *input2, real *output, const int nrow1, const int ncol1, const int nrow2, const int ncol2)
{
    if( nrow2 < MIN_NPREDICT )
    {
        printf("gpu_cdist: nrow2(Npredict) < %d\n", MIN_NPREDICT);
        exit(EXIT_FAILURE);
    }

    dim3 nthread, nblock;
    nthread.x = CDIST_NTHREAD_X;
    nthread.y = CDIST_NTHREAD_Y;
    nthread.z = CDIST_NTHREAD_Z;

    nblock.x = ceil( float(nrow2) / float(nthread.x) );
    nblock.y = ceil( float(nrow1) / float(nthread.y) );
    nblock.z = ceil( float(ncol1) / float(nthread.z) );
   
    kernel_cdist<<<nblock,nthread>>>(input1, input2, output, nrow1, nrow2, ncol1);
}



