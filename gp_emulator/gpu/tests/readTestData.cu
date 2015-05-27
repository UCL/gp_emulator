#include "gpu_predict_test.h"
#include <string.h>


real *readTestData(char *file_name, int M, int N, int D, int size)
{
    
    char path[400];
    sprintf(path, "data/");
    strcat(path,file_name);
    
    real *data;
    
    
    data = (real *)malloc(sizeof(real) * size);
    FILE *file_ptr;
    file_ptr = fopen(path,"rb");
    fread(data, sizeof(real), size, file_ptr);
    fclose(file_ptr);
    return(data);

}



