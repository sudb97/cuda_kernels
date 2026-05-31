#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdio.h>

using namespace std;

#define rows 64
#define columns 32

__global__ void increment_matrix_elements_by_one_tiled(float *A_d, float *B_d, size_t rows_A, size_t cols_A)
{
    //The matrix will have a tile of 32x32 it will increment the individual tile elements and copy back to the original matrix
    __shared__ float tile[32][32];

    //Read the each tiles from matrix A
    size_t row = threadIdx.y + blockIdx.y * blockDim.x; //Index the rows inside the block + (How many rows of block have been passed * rows in each block)
    size_t col = threadIdx.x + blockIdx.x * blockDim.y;

    if (row < rows_A && col < cols_A)
        tile[threadIdx.y][threadIdx.x] = A_d[row * cols_A + col];
    __syncthreads();

    //Increment each of the tile elements and save
    float value = tile[threadIdx.y][threadIdx.x] + 1.0f;
    __syncthreads();

    //Save the incremented value to the B matrix
    if (row < rows_A && col < cols_A)
        B_d[row * cols_A + col] = value;
}

void fill_matrix_random(float *matrix, int num_rows, int num_cols)
{
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

void increment_matrix_elements_CPU(const float *A, float *B, int num_rows, int num_cols)
{
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            B[i * num_cols + j] = A[i * num_cols + j] + 1.0f;
        }
    }
}

bool compare_increment_results(const float *cpu, const float *gpu, size_t num_elements, float epsilon = 1e-5f)
{
    for (size_t i = 0; i < num_elements; ++i) {
        if (std::fabs(cpu[i] - gpu[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

void print_first_row_and_column(const char *label, const float *matrix, int num_rows, int num_cols)
{
    cout << label << " first row: ";
    for (int j = 0; j < num_cols; ++j) {
        cout << matrix[j];
        if (j + 1 < num_cols) {
            cout << ", ";
        }
    }
    cout << endl;

    cout << label << " first column: ";
    for (int i = 0; i < num_rows; ++i) {
        cout << matrix[i * num_cols];
        if (i + 1 < num_rows) {
            cout << ", ";
        }
    }
    cout << endl;
}

int main()
{
    size_t size_bytes = (size_t)rows * (size_t)columns * sizeof(float);
    const size_t num_elements = (size_t)rows * (size_t)columns;

    float *A = (float *)malloc(size_bytes);
    fill_matrix_random(A, rows, columns);

    float *B_cpu = (float *)malloc(size_bytes);
    increment_matrix_elements_CPU(A, B_cpu, rows, columns);

    float *A_d, *B_d;
    cudaMalloc((void **)&A_d, size_bytes);
    cudaMalloc((void **)&B_d, size_bytes);

    cudaMemcpy(A_d, A, size_bytes, cudaMemcpyHostToDevice);

    size_t blocks_x = 32; //horizontal -> Matrix columns
    size_t blocks_y = 32; //vertical -> matrix rows

    dim3 block_size(blocks_x, blocks_y);
    dim3 grid_size((columns + blocks_y - 1) / blocks_y, (rows + blocks_x - 1) / blocks_x);

    increment_matrix_elements_by_one_tiled<<<grid_size, block_size>>>(A_d, B_d, (size_t)rows, (size_t)columns);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Kernel launch failed: " << cudaGetErrorString(err) << endl;
        free(A);
        free(B_cpu);
        cudaFree(A_d);
        cudaFree(B_d);
        return 1;
    }
    cudaDeviceSynchronize();

    float *B_gpu = (float *)malloc(size_bytes);
    cudaMemcpy(B_gpu, B_d, size_bytes, cudaMemcpyDeviceToHost);

    // print_first_row_and_column("CPU", B_cpu, rows, columns);
    // print_first_row_and_column("GPU", B_gpu, rows, columns);

    const bool match = compare_increment_results(B_cpu, B_gpu, num_elements);
    cout << "CPU vs GPU increment-by-one match: " << (match ? "true" : "false") << endl;

    free(A);
    free(B_cpu);
    free(B_gpu);
    cudaFree(A_d);
    cudaFree(B_d);

    return 0;
}
