SRCS= test1.c readTestData.c
OBJS = test1.o readTestData.o
GPUCC = nvcc
CC = gcc
BUILDPATH= #../../build/lib/
INC = #-I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include/numpy
LIB = -lcunit #-lcuda -lcudart -lcublas
NVLIB = #-L/Developer/NVIDIA/CUDA-7.0/lib/




all: test

test: $(OBJS)
	$(CC) -o $@ $^ $(INC) $(NVLIB) $(LIB)  

%.o:%.c
	$(CC)  -c $^  -o $@   

%.o:%.cu
	$(CC)  -c $^ -o $@ 

clean:
	rm *.o test

#
