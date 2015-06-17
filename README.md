GP emulators
==============

:Info: Gaussian process (GP) emulators for Python
:Author: J Gomez-Dans <j.gomez-dans@ucl.ac.uk>
:Date: $Date: 2015-03-17 16:00:00 +0000  $
:Description: README file

This repository contains an implementation of GPs for emulation in Python. Although many different implementations exist, this particular one deals with fast GP predictions for large number of input vectors, where the training data sets are typically modest (e.g. less than 300 samples). Access to the emulation's partial derivatives and Hessian matrix is calculated, and training is also taken care of.

Requirements:
---------
* python ( 2.7 or later )
* numpy
* scipy
* GPU predict module:
    * cmake 
    * CUDA 5.0 or later
    * CUnit

Install without GPU predict:
---------
```bash
cd gp_emulator
python setup.py install
```

Install with GPU predict:
---------
1. install python modules
    ```bash
python setup.py install
```
2. compile CUDA modules
    ```bash
cd build
cmake ..
make 
```
3. export the python path following the cmake instruction, e.g.
    ```bash
    export PYTHONPATH=$PYTHONPATH:/home/sinan/gp-emulator/emulator/build/lib```
    
Tests (only with predict):
----------
1. Unit testing ([unit_tests.py](https://github.com/UCL/gp_emulator/blob/master/tests/unit_tests.py)):
   * random inputs generated by python predict
   * operate unit testings of GPU functions. 
   * compare GPU and python outputs
2. benchmark ([benchmark.py](https://github.com/UCL/gp_emulator/blob/master/tests/benchmark.py))
   * obtain speedup of GPU predict  
   * random inputs
3. testing emulator ([testing_emulator](https://github.com/UCL/gp_emulator/blob/master/tests/test_perband_emulator.py))
   * run emulator with and without GPU
   
