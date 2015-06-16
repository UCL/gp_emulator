#include "gpu_predict_test.h"
//#define max(a,b) (((a)>(b))?(a):(b))

void compare_result( const real *test_val, const real *origin_val, const int len, const real epsilon_average, const real epsilon_max, char var_name[])
{
    int i;
    real e = 0.0 ;
    real error = 0.0;
    real error_max = 0.0;

    for( i = 0; i < len; ++i )
    {
        e = abs( test_val[i] - origin_val[i] ) / abs( origin_val[i] );
        error += e;
        error_max = max( error_max, e );
    } 
    error = error / len;

    if(error > epsilon_average || error_max > epsilon_max)
        printf("        [%s] Average error = %.1e, Max error = %.1e\n",var_name, error, error_max);
    CU_ASSERT(error < epsilon_average);
    CU_ASSERT(error_max < epsilon_max);

}


