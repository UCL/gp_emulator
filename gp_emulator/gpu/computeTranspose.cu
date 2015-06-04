#include "gpu_predict.h"
//ld_in is the leading dimension of input matrix
//ld_out is the leading dimension of the output matrix
extern "C"{
real *computeTranspose(const real *matrix, const  int ld_in, const  int ld_out)
{
    int x, y;
    real * temp;
    temp = ( real *)malloc(sizeof(real) * ld_in * ld_out);
//    for ( i = 0; i < ld_in * ld_out; ++i)
//        temp[i] = matrix[i];

    for ( y = 0; y < ld_out; ++y )
    {
        for ( x = 0; x < ld_in; ++x )
        {
            temp[(x * ld_out) + y] = matrix[(y * ld_in) + x];                                                                

        }   
    }
    return(temp);
}
}
