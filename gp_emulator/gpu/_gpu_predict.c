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

#define N 10000
#define M 250



//__global__ 
//void hello(char *a, int *b) 
//{
//    a[threadIdx.x] += b[threadIdx.x];
//}
 

//  __global__ void gpu_cdist(float *res, float *a, float *b)
// {
//     // int i;
//     int ix = threadIdx.x+blockDim.x*blockIdx.x;
//     // int iy = blockIdx.y * blockDim.y + threadIdx.y;
//     // //int iz = blockIdx.z * blockDim.y + threadIdx.z;
//     // int nx = N;
//     // int ny = M;

//     // for(i = 0; i < ny; i++)
//     // {
//     //     res[nx*ix+iy]+=pow((a[ny*ix+i] - b[ny*iy+i]),2);
//     //     //   res[%(N)s*ix+iy]=sqrt(pow((a[%(M)s*ix]-b[%(M)s*iy]),2) + pow((a[%(M)s*ix+1]-b[%(M)s*iy+1]),2));
//     // }

//     // res[nx*ix+iy]=sqrt(res[nx*ix+iy]);

// }



static PyObject *comp();

static PyObject *predict(PyObject *self, PyObject *args);


PyArrayObject *pyvector(PyObject *objin);
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);


// PyArrayObject *pymatrix(PyObject *objin);
// double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
// double **ptrvector(long n);
// void free_Carrayptrs(double **v);
// int  not_doublematrix(PyArrayObject *mat);



/* ==== Set up the methods table ====================== */
static PyMethodDef gpuMethods[] = {
	{"comp", comp, METH_VARARGS},
	{"predict", predict, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
//Module name must be _C_arraytest in compile and linked 
void init_gpu_predict()  {
	(void) Py_InitModule("_gpu_predict", gpuMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

static PyObject *predict ( PyObject *self, PyObject *args )
{
    int D;
    
    PyArrayObject *py_theta, *py_inputs, *py_invQt, *py_invQ, *py_testing;
    
    double *c_theta;
    double **c_inputs, **c_invQt, **c_invQ, **c_testing;
    
    printf("I'm C\n");
    PyArg_ParseTuple ( args, "iO!O!O!O!O!", &D,
        &PyArray_Type, &py_theta,
        &PyArray_Type, &py_inputs,
        &PyArray_Type, &py_invQt,
        &PyArray_Type, &py_invQ,
        &PyArray_Type, &py_testing);

    
    c_theta = pyvector_to_Carrayptrs( py_theta );
    c_inputs = pymatrix_to_Carrayptrs( py_inputs );
    c_invQt = pymatrix_to_Carrayptrs( py_invQt );
    c_invQ = pymatrix_to_Carrayptrs( py_invQ );
    c_testing = pymatrix_to_Carrayptrs( py_testing );


    //double **g_theta, **g_inputs, **g_in











    //free(c_theta);
    //free((char*) c_inputs);
    //return Py_BuildValue ("i",1);
    return Py_BuildValue ( "O", PyArray_Return ( py_testing ) );

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




































/* ---------------
for testing
------------------*/
int cpu_dot(const float *x, const float *y, float *z){
    int i;
    z[0]=0;
    for(i=0;i<N;i++){
        z[0]=z[0]+x[i]*y[i];
    }
    return(0);
}



int check(cublasStatus_t status){
    if (status != CUBLAS_STATUS_SUCCESS)
    {
    
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    
    }
    return(0);
}

static PyObject *comp()
{


	printf("hello world\n");
    clock_t start,diff;
    cublasHandle_t handle;
    cublasStatus_t status;
    float *x;
    float *y;
    float *z,*zg;
    float *d_x, *d_y, *d_z;
    int i;

    status = cublasCreate(&handle);
    x=(float *)malloc(N * sizeof(x[0]));
    y=(float *)malloc(N * sizeof(y[0]));
    z=(float *)malloc(sizeof(float));
    zg=(float *)malloc(  sizeof(float));


    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));
    cudaMalloc((void **)&d_z, sizeof(float));


    /*initialise*/
    for(i=0;i<N;i++){
        x[i]=0.001;
        y[i]=0.001;
        zg[0]=-9999;
        z[0]=-9999;
    }

    
    /*CPU dot product*/
    start=clock();
    cpu_dot(x,y,z);
    diff=clock()-start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken (CPU) %d seconds %d milliseconds\n", msec/1000, msec%1000);
    


    /*GPU dot product*/
    start=clock();
    cublasSetVector(N,sizeof(float),x,1,d_x,1);
    cublasSetVector(N,sizeof(float),y,1,d_y,1);
    check(cublasSdot(handle,N,d_x,1,d_y,1,zg));
    diff=clock()-start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken (GPU) %d seconds %d milliseconds\n", msec/1000, msec%1000);
    

    /*check result*/
    printf("%f - %f\n",zg[0],z[0]);





    free(x);
    free(y);
    free(z);
    free(zg);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return Py_BuildValue("i", 1);
}


// // static PyObject *rowx2(PyObject *self, PyObject *args)
// // {
// // 	printf("Hello world\n");
// // }

// /* #### Matrix Extensions ############################## */

// /* ==== Row x 2 function - manipulate matrix in place ======================
//     Multiply the 2nd row of the input by 2 and put in output
//     interface:  rowx2(mat1, mat2)
//                 mat1 and mat2 are NumPy matrices
//                 Returns integer 1 if successful                        */
// static PyObject *rowx2(PyObject *self, PyObject *args)
// {
// 	PyArrayObject *matin, *matout;  // The python objects to be extracted from the args
// 	double **cin, **cout;           // The C matrices to be created to point to the 
// 	                                //   python matrices, cin and cout point to the rows
// 	                                //   of matin and matout, respectively
// 	int i,j,n,m;
	
// 	/* Parse tuples separately since args will differ between C fcns */
// 	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &matin,
// 		&PyArray_Type, &matout))  return NULL;
// 	if (NULL == matin)  return NULL;
// 	if (NULL == matout)  return NULL;
	
// 	/* Check that objects are 'double' type and matrices
// 	     Not needed if python wrapper function checks before call to this routine */
// 	if (not_doublematrix(matin)) return NULL;
// 	if (not_doublematrix(matout)) return NULL;
		
// 	/* Change contiguous arrays into C ** arrays (Memory is Allocated!) */
// 	cin=pymatrix_to_Carrayptrs(matin);
// 	cout=pymatrix_to_Carrayptrs(matout);
	
// 	/* Get matrix dimensions. */
// 	n=matin->dimensions[0];
// 	m=matin->dimensions[1];
	
// 	/* Operate on the matrices  */
// 	for ( i=0; i<n; i++)  {
// 		for ( j=0; j<m; j++)  {
// 			if (i==1) cout[i][j]=2.0*cin[i][j];
// 	}  }
		
// 	/* Free memory, close file and return */
// 	free_Carrayptrs(cin);
// 	free_Carrayptrs(cout);
// 	return Py_BuildValue("i", 1);
// }





// PyArrayObject *pymatrix(PyObject *objin)  {
// 	return (PyArrayObject *) PyArray_ContiguousFromObject(objin,
// 		NPY_DOUBLE, 2,2);
// }



// void free_Carrayptrs(double **v)  {
// 	free((char*) v);
// }

// int  not_doublematrix(PyArrayObject *mat)  {
// 	if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2)  {
// 		PyErr_SetString(PyExc_ValueError,
// 			"In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
// 		return 1;  }
// 	return 0;
// }

