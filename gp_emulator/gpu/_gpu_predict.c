
#include "gpu_predict.h"


#define real double

// PyArrayObject *pymatrix(PyObject *objin);
// double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
// double **ptrvector(long n);
// void free_Carrayptrs(double **v);
// int  not_doublematrix(PyArrayObject *mat);



/* ==== Set up the methods table ====================== */
static PyMethodDef gpuMethods[] = {
    {"predict_wrap", predict_wrap, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
//Module name must be _C_arraytest in compile and linked 
void init_gpu_predict()  {
	(void) Py_InitModule("_gpu_predict", gpuMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}





/* #### Vector Utility functions ######################### */

/* ==== Make a Python Array Obj. from a PyObject, ================
     generates a double vector w/ contiguous memory which may be a new allocation if
     the original was not a double type or contiguous 
  !! Must DECREF the object returned from this routine unless it is returned to the
     caller of this routines caller using return PyArray_Return(obj) or
     PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pyvector(PyObject *objin)  {
    return (PyArrayObject *) PyArray_ContiguousFromObject(objin,
        NPY_DOUBLE, 1,1);
}
/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
    int i,n;
    
    n=arrayin->dimensions[0];
    return (double *) arrayin->data;  /* pointer to arrayin data as double */
}
/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
    return 1 if an error and raise exception */ 
int  not_doublevector(PyArrayObject *vec)  {
    if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
        PyErr_SetString(PyExc_ValueError,
            "In not_doublevector: array must be of type Float and 1 dimensional (n).");
        return 1;  }
    return 0;
}



/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */

double **ptrvector(long n)  {
    double **v;
    v=(double **)malloc((size_t) (n*sizeof(double)));
    if (!v)   {
        printf("In **ptrvector. Allocation of memory for double array failed.");
        exit(0);  }
    return v;
}


double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
    double **c, *a;
    int i,n,m;
    
    n=arrayin->dimensions[0];
    m=arrayin->dimensions[1];
    c=ptrvector(n);
    a=(double *) arrayin->data;  /* pointer to arrayin data as double */
    for ( i=0; i<n; i++)  {
        c[i]=a+i*m;  }
    return c;
}





PyObject *predict_wrap ( PyObject *self, PyObject *args )
{
    int N,NN,D,theta_size;
    
    PyArrayObject *py_theta_exp, *py_inputs, *py_invQt, *py_invQ, *py_testing;
    real *c_theta_exp;
    real *c_inputs, *c_invQt, *c_invQ, *c_testing;


    PyArg_ParseTuple ( args, "O!O!O!O!O!iiii",
        &PyArray_Type, &py_theta_exp,
        &PyArray_Type, &py_inputs,
        &PyArray_Type, &py_invQt,
        &PyArray_Type, &py_invQ,
        &PyArray_Type, &py_testing,
        &N,&NN,&D,&theta_size);

    c_theta_exp = pyvector_to_Carrayptrs( py_theta_exp );
    c_inputs = pyvector_to_Carrayptrs( py_inputs );
    c_invQt = pyvector_to_Carrayptrs( py_invQt );
    c_invQ = pyvector_to_Carrayptrs( py_invQ );
    c_testing = pyvector_to_Carrayptrs( py_testing );
    
    predict(c_theta_exp, c_inputs, c_invQt, c_invQ, c_testing, N, NN, D, theta_size);
    return Py_BuildValue ( "O", PyArray_Return ( py_testing ) );
}



























