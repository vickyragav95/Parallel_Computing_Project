#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 1024

// function declarations
extern "C" void matrix_vector_product_onrank(double **M_vec_onrank, int **rowp_onrank, int **colm_onrank, int nrows_onrank, double **x, double **Mp, int myrank, int nnz_onrank, int N);
extern "C" void vector_scaled_addition_onrank(double **vector1,double alpha,double **vector2,int start_row, int nrows_onrank,double **result_vector);
extern "C" double vector_dot_product_onrank(double **vector1,double **vector2,int nrows_onrank, int start_row);
extern "C" double global_vector_dot_product_onrank(double **vector1,int start_row, double **vector2,int nrows_onrank);
extern "C" void calc_residual_onrank(double **rhs,double **Mx0,int start_row,int nrows_onrank,double **residual);
extern "C" void allocate_globalMemory_CUDA(int N);
extern "C" void allocate_onrankMemory_CUDA(int nrows_onrank, int nnz_onrank);
extern int *rowp;
extern double *rhs;
extern double *x0;
extern double *p;
extern int *rowp_onrank;
extern double *M_vec_onrank;
extern int *colm_onrank;
extern double *d_vector1, *d_vector2;

// allocate memory for all partial data on each rank
void allocate_onrankMemory_CUDA(int nrows_onrank, int nnz_onrank){
    cudaMallocManaged( &rowp_onrank, ((nrows_onrank+1) * sizeof(int)));
    cudaMallocManaged( &M_vec_onrank, (nnz_onrank * sizeof(double)));
    cudaMallocManaged( &colm_onrank,  (nnz_onrank * sizeof(int)));
}


// allocate memory for all full vectors (the same copy is available in each ranks)
void allocate_globalMemory_CUDA(int N){
    cudaMallocManaged( &rowp, ((N+1) * sizeof(int)));
    cudaMallocManaged( &rhs, (N * sizeof(double)));
    cudaMallocManaged( &x0,  (N * sizeof(double)));
    cudaMallocManaged( &p,   (N * sizeof(double)));
}


// kernel to compute matrix vector product of sparse matrix
// each thread will perform the calculations for one row
__global__ void csrmul_kernel(double *M, int *rowp, int *colm, int num_rows, double *x, double *y)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if( row<num_rows )
    {
        int row_begin = rowp[row];
        int row_end = rowp[row+1];
        //Multiply row
        double sum =0;
        for(int jj=row_begin; jj < row_end; jj++)
        {
            sum += M[jj] * x[colm[jj]];
        }
        y[row] = sum;
    }
}

// kernel to compute vector addition
// each thread will do the addition on one row
__global__ void scaled_addition_kernel(double *v1,double alpha,double *v2,int start_row, int nrows_onrank,double *result)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < nrows_onrank)
        result[index] = v1[index] + alpha * v2[index+start_row];
}

/*
// dot product kernel without reduction
__global__ void dot(double *v1, double *v2, double *c, int nrows_onrank, int start_row){
	__shared__ double temp[BLOCKSIZE];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < nrows_onrank)
		temp[threadIdx.x] = v1[index+start_row] * v2[index];
	__syncthreads();

	if(threadIdx.x == 0){
		double sum =0;
		for(int i=0; i< BLOCKSIZE; i++){
			sum += temp[i];
		}
		atomicAdd(c, sum);
	}
}
*/

// dot product kernel with reduction
// eahc thread will first compute the element by elemt product and store in a shared allocate_globalMemory_CUDA
// the sum is computed using reduction
__global__ void dot(double *v1, double *v2, double *c, int nrows_onrank, int start_row)
{
    __shared__ double temp[BLOCKSIZE];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    while (index < nrows_onrank)
    {
        temp[threadIdx.x] = v1[index+start_row] * v2[index];
        index += blockDim.x * gridDim.x;
    }
    __syncthreads();
    // reduction to compute the sum 
    int i = blockDim.x/2;
    while(i!=0){
        if (tid < i){
            temp[tid] += temp[tid + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (tid == 0){
    atomicAdd(c, temp[0]);
    }

}


// kernel to compute the inital residual
__global__ void residual_kernel(double *R, double *MX, double *residual,int nrows_onrank, int start_row)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < nrows_onrank)
        residual[index] = R[index+start_row] - MX[index];
}


// function to launch residual computing kernel
void calc_residual_onrank(double **R,
                          double **MX,
                          int start_row,
                          int nrows_onrank,
                          double **residual)
{
    unsigned int blocksize = BLOCKSIZE; // or any size up to 512
    unsigned int nblocks = (nrows_onrank+blocksize-1)/blocksize;
    residual_kernel<<<nblocks, blocksize>>>(*R, *MX, *residual, nrows_onrank, start_row);
}

// function to launch dot product kernel
double vector_dot_product_onrank(double **vector1,
                                 double **vector2,
                                 int nrows_onrank,
                                 int start_row)
{
    double *c;
    c = (double*) malloc(sizeof(double));
    // Executing kernel
    unsigned int blocksize = BLOCKSIZE; // or any size up to 512
    unsigned int nblocks = (nrows_onrank+blocksize-1)/blocksize;
    dot<<<nblocks, blocksize>>>(*vector1, *vector2, d_c, nrows_onrank, start_row);
    cudaDeviceSynchronize();
    return *c;
}

// function to launch vector scaled addition kernel
void vector_scaled_addition_onrank(double **vector1,
                                   double alpha,
                                   double **vector2,
                                   int start_row,
                                   int nrows_onrank,
                                   double **result)
{
    unsigned int blocksize = BLOCKSIZE; // or any size up to 512
    unsigned int nblocks = (nrows_onrank+blocksize-1)/blocksize;
    scaled_addition_kernel<<<nblocks,blocksize>>>(*vector1, alpha, *vector2,start_row, nrows_onrank, *result);
    cudaDeviceSynchronize();
}

// function to launch matrix vector product kernel
void matrix_vector_product_onrank(double **M,
                                  int **rowp,
                                  int **colm,
                                  int nrows_onrank,
                                  double **x,
                                  double **result_vector,
                                  int myrank,
                                  int nnz_onrank,int N)
{
    unsigned int blocksize = BLOCKSIZE; // or any size up to 512
    unsigned int nblocks = (nrows_onrank+blocksize-1)/blocksize;
    csrmul_kernel<<<nblocks,blocksize>>>(*M, *rowp, *colm, nrows_onrank,*x,*result_vector);
    cudaDeviceSynchronize();
}
