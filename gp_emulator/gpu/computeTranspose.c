#include "gpu_predict.h"
//size_in is the leading dimension of input matrix
//size_out is the leading dimension of the output matrix
void computeTranspose(real *matrix, const  int size_in, const  int size_out)
{
    real * temp;
    temp = ( real *)malloc(sizeof(real) * size_in * size_out);

    for ( int i = 0; i < size_in * size_out; ++i)
        temp[i] = matrix[i];

    for (int y = 0; y < size_out; ++y)
    {
        for (int x = 0; x < size_in; ++x)
        {
            matrix[(x * size_out) + y] = temp[(y * size_in) + x];                                                                

        }   
    }
}
