#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace std;

#define rows 1030
#define columns 515

__global__ void matrix_transpose_v1(float *A_d, float *B_d)
{

    // This is a naive v1 implementation where each thread transposes one element
    // Reads: Coaleased ; Writes: Uncoaleased

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;   //Calculating the rows with the y dimension which is slow moving 
    size_t cols = blockIdx.x * blockDim.x + threadIdx.x; //Calculating the colums with the fast moving x dimension

    if (cols < columns && row < rows)
        {
            B_d[cols * rows + row] = A_d[row  * columns + cols];
        }
}

__global__ void matrix_transpose_v2(float *A_d, float *B_d)
{
    /*This is v2 implementation where tilling is implemented. 
        Steps: Read values of matrix A into a local thread variable, then add block level thread barrier
        fill a block level shared memory matrix with transpose, add block level sync barrier;
        fill the B matrix in coaleased form from the tilled matrix.

        The tile size is defined one column padded on the left most column, this is to avoid the bank conflicts
    */
    // Declear the tile before any thread is launched -> used padding to avoid bank conflicts
    __shared__ float tile[32][33];

    size_t row_A = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col_A = blockIdx.x * blockDim.x + threadIdx.x;
    
    //Read the elements into the Tile from matrix A in row wise fashion
    if (row_A < rows && col_A < columns)
        {
            tile[threadIdx.y][threadIdx.x] =  A_d[row_A * columns + col_A];
        }
    else
        {
            tile[threadIdx.y][threadIdx.x] = 0;
        }

    __syncthreads();

    //Transpose the values inside the tile
    float value = tile[threadIdx.x][threadIdx.y];
    __syncthreads();

    //Write the values from the tile to the matrix B in coaleased form
    size_t row_B = blockIdx.x * blockDim.x + threadIdx.y;
    size_t col_B = blockIdx.y * blockDim.y + threadIdx.x;

    if (row_B < columns && col_B < rows)
        {
            B_d[row_B * rows + col_B] = value;
        }

}

void fill_matrix_random(float *matrix, int num_rows, int num_cols)
{
    //For the naive approch we will have single block of dimension rows x columns: Then we make sure that each element of the matrix is responsible for transposing single element

    static bool seeded = false;
    if (!seeded) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        seeded = true;
    }
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            float r = 1.0f + static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX) / 9.0f);
            matrix[i * num_cols + j] = r;
        }
    }
}

void matrix_transpose_CPU(const float *A, float *B, int num_rows, int num_cols)
{
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            B[j * num_rows + i] = A[i * num_cols + j];
        }
    }
}

bool compare_transpose_results(const float *cpu, const float *gpu, size_t num_elements, float epsilon = 1e-5f)
{
    for (size_t i = 0; i < num_elements; ++i) {
        if (std::fabs(cpu[i] - gpu[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main()
{
    float *A;

    size_t size_bytes;
    size_bytes= (size_t)rows * (size_t)columns * sizeof(float);

    A = (float *) malloc(size_bytes);

    fill_matrix_random(A, rows, columns);

    //Allocate device memory for GPU
    float *A_d, *B_d;

    cudaMalloc((void **)&A_d, size_bytes);
    cudaMalloc((void **)&B_d, size_bytes);

    //MemCpy to device memory from host memory
    cudaMemcpy(A_d,A,size_bytes,cudaMemcpyHostToDevice);
    
    //Kernel run and configuration

    //Block Size
    size_t blocks_rows = 32;
    size_t blocks_cols = 32;
    
    dim3 block_size(blocks_rows, blocks_cols);

    //Grid Size
    size_t grid_rows = (rows + blocks_rows - 1)/blocks_rows;
    size_t grid_cols = (columns + blocks_cols - 1)/blocks_cols;
    
    dim3 grid_size(grid_cols,grid_rows);
    
    matrix_transpose_v2<<<grid_size,block_size>>>(A_d,B_d);

    float *B_cpu = (float *)malloc(size_bytes);
    matrix_transpose_CPU(A, B_cpu, rows, columns);

    //Copy back the data from device memory to host memory
    float *B = (float *) malloc(size_bytes);

    cudaMemcpy(B,B_d,size_bytes,cudaMemcpyDeviceToHost);

    const size_t num_elements = (size_t)rows * (size_t)columns;
    const bool match = compare_transpose_results(B_cpu, B, num_elements);

    cout << "CPU vs GPU transpose match: " << (match ? "true" : "false") << endl;

    free(A);
    free(B_cpu);
    free(B);
    cudaFree(A_d);
    cudaFree(B_d);

    return match ? 0 : 1;
}