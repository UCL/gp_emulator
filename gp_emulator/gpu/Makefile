SRCS= _gpu_predict.c predict.cu cdist.cu computeTranspose.c
OBJS = _gpu_predict.o cdist.o predict.o computeTranspose.o
SHARE = _gpu_predict.so #$(SRCS:.c=.so) $(CUDASRCS:.cu=.so)
GPUCC = nvcc
CC = gcc
BUILDPATH= ../../build/lib/
INC = -I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include/numpy
LIB = -lcuda -lcudart -lcublas
NVLIB = -L/Developer/NVIDIA/CUDA-7.0/lib/




all: $(SHARE) $(OBJS)


%.so:$(OBJS)
	$(CC) -shared -fpic -flat_namespace -undefined suppress -o $(BUILDPATH)$@ $^ $(INC) $(NVLIB) $(LIB)

%.o:%.c
	$(GPUCC)  -c $^  -o $@  $(INC) $(LIB) $(NVLIB)  --compiler-options '-fPIC' 

%.o:%.cu
	$(GPUCC)  -c $^ -o $@ $(INC) $(NVLIB) $(LIB)  --compiler-options '-fPIC' 

clean:
	rm *.o $(BUILDPATH)*.so

#
