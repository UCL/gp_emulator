/* A file to test imorting C modules for handling arrays to Python */

#include "Python.h"
#include "arrayobject.h"
//#include "C_arraytest.h"
#include <math.h>
#include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <time.h>
/* Includes, cuda */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


/*
* he
*/
//#ifndef GPU_PREDICT_H
//#define GPU_PREDICT_H


void hi();
//void hello();


static PyObject *comp();

static PyObject *predict(PyObject *self, PyObject *args);

PyArrayObject *pyvector(PyObject *objin);
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);

//#endif

