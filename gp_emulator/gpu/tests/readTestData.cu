#include "gpu_predict_test.h"
#include <string.h>


real *readTestData(char *file_name, const int size)
{
    int i;
    char path[400];
    sprintf(path, "data/");
    strcat(path,file_name);
    
    double *data_raw;
    real *data;
    
    data_raw = (double *)malloc(sizeof(double) * size);    
    data = (real *)malloc(sizeof(real) * size);
    FILE *file_ptr;
    file_ptr = fopen(path,"rb");
    fread(data_raw, sizeof(double), size, file_ptr);

    for( i = 0; i < size; ++i )
        data[i] = (real)data_raw[i]; 

    fclose(file_ptr);
    return(data);

}



