SRCS= test1.c  testVecTimesMat.cu testCdist.cu\
      readTestData.c ../computeTranspose.c ../predict.cu

OBJS = test1.o readTestData.o testVecTimesMat.o testCdist.o \
       ../computeTranspose.o ../predict.o
GPUCC = nvcc
CC = gcc
BUILDPATH= #../../build/lib/
INC = -I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include/numpy
LIB = -link  -lcunit -lcuda -lcudart -lcublas
NVLIB = -L/Developer/NVIDIA/CUDA-7.0/lib/




all: test

test: $(OBJS)
	$(GPUCC)  -o $@ $^ $(INC) $(NVLIB) $(LIB)  

%.o:%.c
	$(GPUCC) -x cu  -dc $^  -o $@   $(INC) $(LIB) $(NVLIB)

%.o:%.cu
	$(GPUCC)  -dc $^ -o $@ $(INC) $(NVLIB) $(NVLIB)

clean:
	rm *.o test

#
