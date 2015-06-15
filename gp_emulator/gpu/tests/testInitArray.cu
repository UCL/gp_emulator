#include"gpu_predict_test.h"

void testInitArray(void)
{
    real *vec_c, *vec_g;
    real val;
    int len;
    int i, test, error;
    vec_c = (real *)malloc( sizeof(real) * 1e7 );
    cudaMalloc( (void **)&vec_g, sizeof(real) * 1e7 );
 
    for( test = 0; test < 6; ++test )
    {
        val = pow( 10.0, test-5);
        len = pow( 10.0, test );
        gpu_init_array(vec_g, val, len);
        cudaMemcpy(vec_g, vec_c, sizeof(real) * len, cudaMemcpyDeviceToHost);
        
        error = 0;
        for( i = 0; i < len; ++i )
        {
            if( ( vec_c[i] - val ) / val > 1e-15 )
                error++;
        }
        CU_ASSERT(error == 0);
    }

    free(vec_c);
    cudaFree(vec_g);
}
