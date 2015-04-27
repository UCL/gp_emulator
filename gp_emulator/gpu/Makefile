SRCS= _gpu_predict.c cdist.cu
OBJS = _gpu_predict.o cdist.o
SHARE = _gpu_predict.so #$(SRCS:.c=.so) $(CUDASRCS:.cu=.so)
GPUCC = nvcc
CC = gcc
BUILDPATH= ../../build/lib/
INC = -I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include/numpy
LIB = -lcuda -lcudart -lcublas
NVLIB = -L/Developer/NVIDIA/CUDA-7.0/lib/




all: $(SHARE)


%.so:$(OBJS)
	$(CC) -bundle -flat_namespace -undefined suppress -o $(BUILDPATH)$@ $^ $(INC) $(NVLIB) $(LIB)

%.o:%.c
	$(GPUCC)  -c $^  -o $@  $(INC) $(LIB) $(NVLIB)

%.o:%.cu
	$(GPUCC) -c $^ -o $@ $(INC) $(NVLIB) $(LIB)

clean:
	rm *.o $(BUILDPATH)*.so
