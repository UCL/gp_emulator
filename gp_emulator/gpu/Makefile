
CC = nvcc
BUILDPATH= ../../build/lib
OBJ = _gpu_predict
INC = -I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include/numpy
LIB = -lcuda -lcudart -lcublas
NVLIB = -L/Developer/NVIDIA/CUDA-7.0/lib/


$(BUILDPATH)/$(OBJ).so:  $(BUILDPATH)/$(OBJ).o
	gcc -bundle -flat_namespace -undefined suppress -o $@ $<  $(LIB) $(NVLIB)

$(BUILDPATH)/$(OBJ).o:  $(OBJ).c
	nvcc  -c $< -o $@ $(NVLIB) $(LIB) $(INC)

clean:
	rm -r $(BUILDPATH)/*.o $(BUILDPATH)/*.so #$(BUILDPATH)/gp_emulator



