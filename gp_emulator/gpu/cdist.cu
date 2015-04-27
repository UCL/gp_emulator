//#include<cuda.h>
#include<stdio.h>
#include "gpu_predict.h"


 


__global__ void gpu_cdist(float *res, float *a, float *b)
{
/*
    int i;
     int ix = threadIdx.x+blockDim.x*blockIdx.x;
     int iy = blockIdx.y * blockDim.y + threadIdx.y;
     int nx = N;
     int ny = M;
     for(i = 0; i < ny; i++)
     {
     //res[nx*ix+iy]+=pow((a[ny*ix+i] - b[ny*iy+i]),2);
        // res[%(N)s*ix+iy]=sqrt(pow((a[%(M)s*ix]-b[%(M)s*iy]),2) + pow((a[%(M)s*ix+1]-b[%(M)s*iy+1]),2));
     }
     
     res[nx*ix+iy]=sqrt(res[nx*ix+iy]);
*/

}


