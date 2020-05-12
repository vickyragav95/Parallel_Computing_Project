all: cg.c cg_cuda.cu
	mpicc -g cg.c -c -o cg.o
	nvcc -g -G -arch=sm_70 cg_cuda.cu -c -o cg_cuda.o
	mpicc -g cg.o cg_cuda.o -o cg-cuda-mpi-exe \
	 -L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++
