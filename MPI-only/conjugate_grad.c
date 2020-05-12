#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<unistd.h>
#include<mpi.h>

// global variables
double *rhs, *x0, *p;
int *rowp, N;
#define INITIAL_GUESS 1.0

// code to calculate ticks and time on POWER9 systems
typedef unsigned long long ticks;
static __inline__ ticks getticks(void)
{
  unsigned int tbl, tbu0, tbu1;

  do {
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);

  return (((unsigned long long)tbu0) << 32) | tbl;
}

// routine to allocate memory and read the partial data on each rank
void allocate_rank_rows(int start_row,
						int end_row,
						double **M_vec_onrank,
						char *matrix_filename,
						int **rowp_onrank,
						char *rowp_filename,
						int **colm_onrank,
						char *colm_filename)
{
	int start_index, end_index, nnz_onrank, nrows_onrank;
	start_index = rowp[start_row];
	end_index = rowp[end_row+1]-1;
	nrows_onrank = end_row-start_row+1;
	nnz_onrank = end_index-start_index+1;

	int *tmp_rowp_onrank = malloc((nrows_onrank+1)*sizeof(int));
	double *tmp_M_vec_onrank = malloc(nnz_onrank*sizeof(double));
	int *tmp_colm_onrank = malloc(nnz_onrank*sizeof(int));

	MPI_Status status;
	MPI_Offset offset;

	MPI_File fh_matrix, fh_rowp, fh_colm;
	MPI_File_open(MPI_COMM_WORLD, matrix_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_matrix);
	MPI_File_open(MPI_COMM_WORLD, rowp_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_rowp);
	MPI_File_open(MPI_COMM_WORLD, colm_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_colm);

	offset = start_index*sizeof(double);
	MPI_File_read_at(fh_matrix, offset, tmp_M_vec_onrank, nnz_onrank, MPI_DOUBLE, &status);
	MPI_File_close(&fh_matrix);

	offset = start_index*sizeof(int);
	MPI_File_read_at(fh_colm, offset, tmp_colm_onrank, nnz_onrank, MPI_INT, &status);
	MPI_File_close(&fh_colm);

	offset = start_row*sizeof(int);
	MPI_File_read_at(fh_rowp, offset, tmp_rowp_onrank, nrows_onrank+1, MPI_INT, &status);
	MPI_File_close(&fh_rowp);
	MPI_Barrier(MPI_COMM_WORLD);

	int i;
	for (i=0; i<nrows_onrank+1; i++){
		tmp_rowp_onrank[i] -= rowp[start_row];
	}

	*rowp_onrank = tmp_rowp_onrank;
	*M_vec_onrank = tmp_M_vec_onrank;
	*colm_onrank = tmp_colm_onrank;
}

// computes partial matrix vector product on each rank
void matrix_vector_product_onrank(double *M_vec_onrank,
								  int *rowp_onrank,
								  int *colm_onrank,
								  int nrows_onrank,
								  double *rhs,
								  double *result_vector,
							  	  int myrank)
{
	int i, j, nnz_row, first_rowindex, index_onrank;
	for (i=0; i<nrows_onrank; i++){
		result_vector[i] = 0.0;
		nnz_row = rowp_onrank[i+1]-rowp_onrank[i];
		first_rowindex = rowp_onrank[i];
		for (j=0; j<nnz_row; j++){
			index_onrank = j+first_rowindex;
			result_vector[i] += M_vec_onrank[index_onrank]*rhs[colm_onrank[index_onrank]];
		}
	}
}

// calculates partial residual
void calc_residual_onrank(double *rhs,
						  double *Mx0,
						  int start_row,
						  int nrows_onrank,
						  double *residual)
{
	int i;
	for (i=0; i<nrows_onrank; i++){
		residual[i] = rhs[start_row+i] - Mx0[i];
	}
}

// calculates partial vector scled addition
void vector_scaled_addition_onrank(double *vector1,
								   double alpha,
								   double *vector2,
								   int nrows_onrank,
								   double *result_vector)
{
	int i;
	for (i=0; i<nrows_onrank; i++){
		result_vector[i] = vector1[i] + alpha*vector2[i];
	}
}

// calculates partial dot product
double vector_dot_product_onrank(double *vector1,
								 double *vector2,
							 	 int nrows_onrank)
{
	int i;
	double partial_sum = 0.0;
	for (i=0; i<nrows_onrank; i++){
		partial_sum += vector1[i]*vector2[i];
	}
	return partial_sum;
}

void vector_swap( double **pA, double **pB)
{
  // swapping pointers
  double *temp = *pA;
  *pA = *pB;
  *pB = temp;
}

int main(int argc, char *argv[]){
    // Refer to CUDA MPI code for detailed comments
	unsigned long long start = 0;
	unsigned long long finish = 0;
	start = getticks();

	char matrix_filename[256];
	char rowp_filename[256];
	char colm_filename[256];
	char rhs_filename[256];

	if (argc!=6){
		printf("ERROR: Invalid input...\n");
		printf("Execute the code with the following 6 commandline arguments:\n");
		printf("\tN                \t Size of the Linear system\n");
		printf("\tmatrix-file-name \t Binary File where the sparse matrix elements is stored as vector array\n");
		printf("\trowp-file-name   \t Binary File where location of the first non-zero element of each row in vectorized sparse matrix\n");
		printf("\tcolm-file-name   \t Binary File where column indices of non-zero elements is stored\n");
		printf("\trhs-file-name    \t Binary File where rhs array of the linear system is stored\n");
		exit(0);
	}

	N = atoi(argv[1]);
	strcpy(matrix_filename, argv[2]);
	strcpy(rowp_filename, argv[3]);
	strcpy(colm_filename, argv[4]);
	strcpy(rhs_filename, argv[5]);

	int numranks, myrank;
	MPI_Status status;
	MPI_Offset offset;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int i;
	rowp = (int*) malloc((N+1)*sizeof(int));
	rhs = (double*) malloc(N*sizeof(double));
	x0 = (double*) malloc(N*sizeof(double));
	p = (double*) malloc(N*sizeof(double));
	for (i=0; i<N; i++){
		x0[i] = INITIAL_GUESS;
	}

	MPI_File fh_rowp, fh_rhs;
	MPI_File_open(MPI_COMM_WORLD, rowp_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_rowp);
	MPI_File_read(fh_rowp, rowp, N+1, MPI_INT, &status);
	MPI_File_close(&fh_rowp);
	MPI_File_open(MPI_COMM_WORLD, rhs_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_rhs);
	MPI_File_read(fh_rhs, rhs, N, MPI_DOUBLE, &status);
	MPI_File_close(&fh_rhs);

	int rank_num_rows = N/numranks;
	int reminder_rows = N%numranks;

	int start_row, end_row, nrows_onrank;
	start_row = myrank*rank_num_rows;
	if (myrank==numranks-1){
		end_row = start_row + rank_num_rows + reminder_rows - 1;
	}
	else{
		end_row = start_row + rank_num_rows-1;
	}

	nrows_onrank = end_row-start_row+1;

	double *M_vec_onrank;
	int *rowp_onrank, *colm_onrank;

	allocate_rank_rows(start_row, end_row, &M_vec_onrank, matrix_filename, &rowp_onrank, rowp_filename, &colm_onrank, colm_filename);

	MPI_Barrier(MPI_COMM_WORLD);

	double *x, *r, *xnew, *pnew, *rnew, *Mp;
	x = (double*) malloc(nrows_onrank*sizeof(double));
	for (i=0; i<nrows_onrank; i++){
		x[i]=INITIAL_GUESS;
	}
	r = (double*) malloc(nrows_onrank*sizeof(double));
	xnew = (double*) malloc(nrows_onrank*sizeof(double));
	rnew = (double*) malloc(nrows_onrank*sizeof(double));
	pnew = (double*) malloc(nrows_onrank*sizeof(double));
	Mp = (double*) malloc(nrows_onrank*sizeof(double));

	matrix_vector_product_onrank(M_vec_onrank, rowp_onrank, colm_onrank, nrows_onrank, x0, Mp, myrank);
	calc_residual_onrank(rhs, Mp, start_row, nrows_onrank, r);

	char update_filename[256];
	strcpy(update_filename, "update_vector");

	MPI_File fh;
	MPI_File_open(MPI_COMM_WORLD, update_filename, MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
	offset = start_row*sizeof(double);
	MPI_File_write_at(fh, offset, r, nrows_onrank, MPI_DOUBLE, &status);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_File_seek(fh,0,MPI_SEEK_SET);
	MPI_File_read_all(fh, p, N, MPI_DOUBLE, &status);

	double epsilon = 1.0;
	int iter = 0;
    // iterative solver -- same algorithm as in CUDA-MPI code
	while (epsilon>1e-7){
		matrix_vector_product_onrank(M_vec_onrank, rowp_onrank, colm_onrank, nrows_onrank, p, Mp, myrank);

		double rr, rr_sum, pMp, pMp_sum, alpha, beta;
		rr = vector_dot_product_onrank(r, r, nrows_onrank);
		//  numerator
		if (myrank>0){
			MPI_Send(&rr, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
		else{
			double tmp_partialsum;
			rr_sum=0;
			for (i=1; i<numranks; i++){
				MPI_Recv(&tmp_partialsum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
				rr_sum += tmp_partialsum;
			}
			rr_sum = rr_sum+rr;
		}
		MPI_Bcast(&rr_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		pMp = vector_dot_product_onrank(p+start_row, Mp, nrows_onrank);
		//  denominator
		if (myrank>0){
			MPI_Send(&pMp, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
		else{
			double tmp_partialsum;
			pMp_sum=0;
			for (i=1; i<numranks; i++){
				MPI_Recv(&tmp_partialsum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
				pMp_sum += tmp_partialsum;
			}
			pMp_sum = pMp_sum+pMp;
		}
		MPI_Bcast(&pMp_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		alpha = rr_sum/pMp_sum;


		vector_scaled_addition_onrank(x, alpha, p+start_row, nrows_onrank, xnew);
		vector_scaled_addition_onrank(r, -alpha, Mp, nrows_onrank, rnew);

		double rnewrnew, rnewrnew_sum;
		rnewrnew = vector_dot_product_onrank(rnew, rnew, nrows_onrank);
		//  numerator
		if (myrank>0){
			MPI_Send(&rnewrnew, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
		else{
			double tmp_partialsum;
			rnewrnew_sum=0;
			for (i=1; i<numranks; i++){
				MPI_Recv(&tmp_partialsum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
				rnewrnew_sum += tmp_partialsum;
			}
			rnewrnew_sum = rnewrnew_sum+rnewrnew;
		}
		MPI_Bcast(&rnewrnew_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		beta = rnewrnew_sum/rr_sum;

		vector_scaled_addition_onrank(rnew, beta, p+start_row, nrows_onrank, pnew);

		MPI_File_write_at(fh, offset, pnew, nrows_onrank, MPI_DOUBLE, &status);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_File_seek(fh,0,MPI_SEEK_SET);
		MPI_File_read_all(fh, p, N, MPI_DOUBLE, &status);

		vector_swap(&r, &rnew);
		vector_swap(&x, &xnew);
		epsilon = rnewrnew_sum;
		iter++;

	}

    if (myrank==0){
        printf("iteration %d -- epsilon = %.16e\n", iter, epsilon);
    }

	MPI_File_close(&fh);

	char result_filename[256];
	strcpy(result_filename, "resultfile");
	MPI_File fh_result;
	MPI_File_open(MPI_COMM_WORLD, result_filename, MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh_result);
	offset = start_row*sizeof(double);
	MPI_File_write_at(fh_result, offset, x, nrows_onrank, MPI_DOUBLE, &status);
	MPI_File_close(&fh_result);

	MPI_Finalize();

	finish = getticks();
	if (myrank==0){
		double tot_time = ((double) finish - (double) start)/(512000000.0);
		printf("Runtime: finish(%llu) - start(%llu) = %llu   Tot time = %lf\n", finish, start, (finish-start), tot_time );
	}

    // free all allocated memory
    free(rowp);
    free(rhs);
    free(x0);
    free(p);
    free(x);
    free(r);
    free(xnew);
    free(pnew);
    free(rnew);
    free(Mp);
    free(rowp_onrank);
    free(M_vec_onrank);
    free(colm_onrank);

	return 0;
}
