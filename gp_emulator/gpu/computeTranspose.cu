#include "gpu_predict.h"

real *computeTranspose(const real *matrix, const  int lead_dim_in, const  int lead_dim_out)
{
    int x, y;
    real * temp;
    temp = ( real *)malloc(sizeof(real) * lead_dim_in * lead_dim_out);

    for ( y = 0; y < lead_dim_out; ++y )
    {
        for ( x = 0; x < lead_dim_in; ++x )
        {
            temp[(x * lead_dim_out) + y] = matrix[(y * lead_dim_in) + x];                                                                

        }   
    }
    return(temp);
}
