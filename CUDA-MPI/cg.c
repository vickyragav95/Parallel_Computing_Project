#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<mpi.h>

int N; // Size of linear system
#define INITIAL_GUESS 1.0 // inital guess for solution

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

// host function declarations
void matrix_vector_product_onrank(double **M_vec_onrank, int **rowp_onrank, int **colm_onrank, int nrows_onrank, double **x, double **Mp, int myrank, int nnz_onrank, int N);
void vector_scaled_addition_onrank(double **vector1,double alpha,double **vector2,int start_row, int nrows_onrank,double **result_vector);
double vector_dot_product_onrank(double **vector1,double **vector2,int nrows_onrank, int start_row);
void calc_residual_onrank(double **rhs,double **Mx0,int start_row,int nrows_onrank,double **residual);
void allocate_globalMemory_CUDA(int N);
void allocate_onrankMemory_CUDA(int nrows_onrank, int nnz_onrank);

// global variables
int *rowp;
double *rhs,*x0,*p;
int *rowp_onrank, *colm_onrank;
double *M_vec_onrank;

int allocate_rank_rows(int start_row,int end_row, char *matrix_filename, char *rowp_filename, char *colm_filename)
{
    int start_index, end_index, nnz_onrank, nrows_onrank;
    // index of first row on each rank
    start_index = rowp[start_row];
    // index of last row on each rank
    end_index = rowp[end_row+1]-1;
    // number of rows on each rank
    nrows_onrank = end_row-start_row+1;
    // number of non-zero terms on each rank
    nnz_onrank = end_index-start_index+1;

    // allocate memory for partitioned inputs on each rank
    allocate_onrankMemory_CUDA(nrows_onrank,nnz_onrank);

    MPI_Status status;
    MPI_Offset offset;

    MPI_File fh_matrix, fh_rowp, fh_colm;
    // open sparse matrix input file
    MPI_File_open(MPI_COMM_WORLD, matrix_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_matrix);
    // open rowp aux array for reading sparse matrix
    MPI_File_open(MPI_COMM_WORLD, rowp_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_rowp);
    // open colm aux array for reading sparse matrix
    MPI_File_open(MPI_COMM_WORLD, colm_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_colm);

    // move file pointer and read partial matrix onto each rank
    offset = start_index*sizeof(double);
    MPI_File_read_at(fh_matrix, offset, M_vec_onrank, nnz_onrank, MPI_DOUBLE, &status);
    // move file pointer and read partial colm array onto each rank
    offset = start_index*sizeof(int);
    MPI_File_read_at(fh_colm, offset, colm_onrank, nnz_onrank, MPI_INT, &status);
    // move file pointer and read partial rowp array onto each rank
    offset = start_row*sizeof(int);
    MPI_File_read_at(fh_rowp, offset, rowp_onrank, nrows_onrank+1, MPI_INT, &status);

    MPI_Barrier(MPI_COMM_WORLD);

    // offset rowp on each rank to get the pointer to first non-zero term of
    // partial sparse matrix stored in each rank
    int i;
    for (i=0; i<nrows_onrank+1; i++){
        rowp_onrank[i] -= rowp[start_row];
    }

    return nnz_onrank;
}


void vector_swap( double **pA, double **pB)
{
    // swapping pointers
    double *temp = *pA;
    *pA = *pB;
    *pB = temp;
}


int main(int argc, char *argv[]){

    unsigned long long start = 0;
    unsigned long long finish = 0;
    start = getticks();

    char matrix_filename[256];
    char rowp_filename[256];
    char colm_filename[256];
    char rhs_filename[256];

    // print usage
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

    // parsing command input and loading inputfiles
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
    // alllocate memory for all full variables
    allocate_globalMemory_CUDA(N);
    for (i=0; i<N; i++){
        x0[i] = INITIAL_GUESS;
    }
    // read full rhs file and rowp file on all ranks
    MPI_File fh_rowp, fh_rhs;
    MPI_File_open(MPI_COMM_WORLD, rowp_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_rowp);
    MPI_File_read(fh_rowp, rowp, N+1, MPI_INT, &status);
    MPI_File_open(MPI_COMM_WORLD, rhs_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_rhs);
    MPI_File_read(fh_rhs, rhs, N, MPI_DOUBLE, &status);

    // calculate number of rows to be handled in each rank (equal for ranks 0 to numranks-2)
    int rank_num_rows = N/numranks;
    // rank numranks-1 will get the reminder rows
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

    int nnz_onrank =  allocate_rank_rows(start_row, end_row, matrix_filename, rowp_filename, colm_filename);

    MPI_Barrier(MPI_COMM_WORLD);

    // declare variables for linear solver and allocate memory
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

    // calculate initial residual based on initial guess solution
    matrix_vector_product_onrank(&M_vec_onrank, &rowp_onrank, &colm_onrank, nrows_onrank, &x0, &Mp, myrank, nnz_onrank, N);
    calc_residual_onrank(&rhs,&Mp, start_row, nrows_onrank,&r);

    char update_filename[256];
    strcpy(update_filename, "update_vector");

    // write partial residual from each rank to common file
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, update_filename, MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    offset = start_row*sizeof(double);
    MPI_File_write_at(fh, offset, r, nrows_onrank, MPI_DOUBLE, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_seek(fh,0,MPI_SEEK_SET);
    // read the full residual vector on each rank from the same file
    MPI_File_read_all(fh, p, N, MPI_DOUBLE, &status);

    // initialize convergence tolerance
    double epsilon = 1.0;
    int iter = 0;

    // iterative linear solver loop
    while (epsilon>1e-7){
        matrix_vector_product_onrank(&M_vec_onrank, &rowp_onrank, &colm_onrank, nrows_onrank, &p, &Mp, myrank, nnz_onrank, N);
        double rr, rr_sum, pMp, pMp_sum, alpha, beta;
        // compute dot product of residual
        rr = vector_dot_product_onrank(&r, &r, nrows_onrank,0);
        //  compute numerator of alpha -- solution update step size
        if (myrank>0){
            // send partial dot products to master ranks
            MPI_Send(&rr, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        else{
            double tmp_partialsum;
            rr_sum=0;
            for (i=1; i<numranks; i++){
                // receive partial dot products from slave ranks and compute their sum
                MPI_Recv(&tmp_partialsum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                rr_sum += tmp_partialsum;
            }
            rr_sum = rr_sum+rr;
        }
        // broadcast computed dot product on master rank to all slaves
        MPI_Bcast(&rr_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        pMp = vector_dot_product_onrank(&p,&Mp,nrows_onrank,start_row);
        //  denominator of alpha
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

        // compute alpha = numerator/denominator
        alpha = rr_sum/pMp_sum;

        // update partial solution vector on each rank
        vector_scaled_addition_onrank(&x, alpha, &p,start_row, nrows_onrank, &xnew);
        // update new partial residual on each rank
        vector_scaled_addition_onrank(&r, -alpha, &Mp,0, nrows_onrank, &rnew);

        //computing beta for update vector (same procedure as alpha)
        double rnewrnew, rnewrnew_sum;
        rnewrnew = vector_dot_product_onrank(&rnew, &rnew, nrows_onrank,0);
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
        // update new partial update vector on each rank
        vector_scaled_addition_onrank(&rnew, beta, &p,start_row, nrows_onrank, &pnew);
        // write partial update vector on each rank to common update file
        MPI_File_write_at(fh, offset, pnew, nrows_onrank, MPI_DOUBLE, &status);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_File_seek(fh,0,MPI_SEEK_SET);
        // once all processors has finised writing read the full update vector
        MPI_File_read_all(fh, p, N, MPI_DOUBLE, &status);
        // swap residual and solution with new values
        vector_swap(&r, &rnew);
        vector_swap(&x, &xnew);
        // compute new error
        epsilon = rnewrnew_sum;
        //if(myrank ==0)
        //	printf("Iteration: %6d    epsilon = %.16e\n",iter,epsilon);
        iter++;
    }

    if (myrank==0){
        printf("iteration %d -- epsilon = %.16e\n", iter, epsilon);
    }

    MPI_File_close(&fh);
    // print solution to result file
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

    free(x);
    free(r);
    free(xnew);
    free(pnew);
    free(rnew);
    free(Mp);
    
    return 0;
}
