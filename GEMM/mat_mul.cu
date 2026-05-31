#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cuda_runtime.h>

#include "matMul_GPU_v1.cu"
#include "matMul_GPU_v2.cu"
#include "matMul_GPU_v3.cu"

#define BLOCK_SIZE 32

using namespace std;

static inline void cuda_check_impl(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " (" << static_cast<int>(err) << ")"
                  << " at " << file << ":" << line
                  << " in " << expr << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) cuda_check_impl((expr), #expr, __FILE__, __LINE__)
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

// Fills a matrix with random floats in the range 1.0 to 10.0
void fill_matrix_random(float *matrix, int rows, int cols) {
    static bool seeded = false;
    if (!seeded) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        seeded = true;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float r = 1.0f + static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX) / 9.0f);
            matrix[i * cols + j] = r;
        }
    }
}

// CPU Multiplication Function
void matMul_CPU(float *A, float *B, float *C, size_t rows_a, size_t rows_b, size_t cols_b)
    {
        for (size_t i=0;i<rows_a;i++)
            {
                for (size_t j=0;j<cols_b;j++)
                    {
                        float sum = 0;
                        for (size_t k=0;k<rows_b;k++)
                            {
                                sum = sum + A[k + i * rows_b] * B[j + k * cols_b];
                            }
                        C[j + i * cols_b] = sum;
                    }
            }
    }

// Prints a 2D matrix of size rows x cols
void print_matrix(const float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.3f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// Find the average difference between sum of elements of two matrices
float avg_diff_bw_elements(float *A, float *B, size_t numelem)
    {
        float agg_diff = 0.0f;

        for (size_t i=0; i<numelem; i++)
            {
                agg_diff = agg_diff + (abs(A[i] - B[i]));  
            }

        return agg_diff/numelem;
    }

int main()
{
    size_t rows_a = 1024;
    size_t rows_b = 1024;
    size_t cols_b = 1024; 

    // Host memory for matrix A, B and C
    const size_t bytes_A = rows_a * rows_b * sizeof(float);
    const size_t bytes_B = rows_b * cols_b * sizeof(float);
    const size_t bytes_C = rows_a * cols_b * sizeof(float);

    float *A = (float *)malloc(bytes_A); 
    float *B = (float *)malloc(bytes_B); 
    float *C = (float *)malloc(bytes_C);

    fill_matrix_random(A,rows_a,rows_b);
    fill_matrix_random(B,rows_b,cols_b);

    /*
    clock_t start = clock();
    matMul_CPU(A,B,C,rows_a, rows_b, cols_b);
    clock_t end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "CPU Execution Time: " << cpu_time << " seconds" << endl;
    */

    // GPU V1 VERSION IMPLEMENTATION

    /* Below is the Architecture used for GPU multiplication kernel
        1. The number of threads is equal to the rows in the product matrix i.e., rows_a
        2. Each thread will fill the single row of the product matrix 

        Note: The parallelism will be in terms of the number of rows of the product matrix
    */

    uint32_t NUM_BLOCKS = static_cast<uint32_t>((rows_a + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Make the device memory
    float *A_d, *B_d, *C_d;

    CUDA_CHECK(cudaMalloc((void**)&A_d, bytes_A));
    CUDA_CHECK(cudaMalloc((void**)&B_d, bytes_B));
    CUDA_CHECK(cudaMalloc((void**)&C_d, bytes_C));

    // Copy Host to Device Memory
    CUDA_CHECK(cudaMemcpy(A_d, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, bytes_B, cudaMemcpyHostToDevice));

    cudaEvent_t start_v1, stop_v1;
    CUDA_CHECK(cudaEventCreate(&start_v1));
    CUDA_CHECK(cudaEventCreate(&stop_v1));

    CUDA_CHECK(cudaEventRecord(start_v1));

    // Run the GPU Kernel
    matMul_GPU_v1<<<NUM_BLOCKS, BLOCK_SIZE>>> (A_d, B_d, C_d, rows_a, rows_b, cols_b);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the product back from Device to Host Memory
    float *C_gpu_h;

    C_gpu_h = (float*) malloc(bytes_C);

    CUDA_CHECK(cudaMemcpy(C_gpu_h, C_d, bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop_v1));
    CUDA_CHECK(cudaEventSynchronize(stop_v1));

    float gpu_time_v1_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_v1_ms, start_v1, stop_v1));
    cout << "GPU v1 Execution Time (including memcopy): " << gpu_time_v1_ms << " ms" << endl;

    CUDA_CHECK(cudaEventDestroy(start_v1));
    CUDA_CHECK(cudaEventDestroy(stop_v1));

    // GPU MULTIPLICATION V2 VERSION

    // Now the block is 2D, that is thread is structured in a 2D form here. Pre-Decided
    int block_rows = 32;  // 32 threads in a row
    int block_cols = 32;  // 32 threads in a column 
    
    dim3 block(block_rows, block_cols);

    uint32_t rows_grid = static_cast<uint32_t>((rows_a + block_rows - 1) / block_rows); //The number of blocks in a row in the grid
    uint32_t cols_grid = static_cast<uint32_t>((cols_b + block_cols - 1) / block_cols); //The number of blocks in a column in the grid

    dim3 grid(cols_grid, rows_grid);

    cudaEvent_t start_v2, stop_v2;
    CUDA_CHECK(cudaEventCreate(&start_v2));
    CUDA_CHECK(cudaEventCreate(&stop_v2));

    CUDA_CHECK(cudaEventRecord(start_v2));

    matMul_GPU_v2<<<grid,block>>>(A_d, B_d, C_d, rows_a, rows_b, cols_b);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the product back from Device to Host Memory
    float *C_gpu_h_v2;

    C_gpu_h_v2 = (float*) malloc(bytes_C);

    CUDA_CHECK(cudaMemcpy(C_gpu_h_v2, C_d, bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop_v2));
    CUDA_CHECK(cudaEventSynchronize(stop_v2));

    float gpu_time_v2_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_v2_ms, start_v2, stop_v2));
    cout << "GPU v2 Execution Time (including memcopy): " << gpu_time_v2_ms << " ms" << endl;

    CUDA_CHECK(cudaEventDestroy(start_v2));
    CUDA_CHECK(cudaEventDestroy(stop_v2));

    // V3 implementation with coalising implementation A-> Broadcast, B-> coalsing and C-> Coalisin
    
    cudaEvent_t start_v3;
    cudaEvent_t stop_v3;

    CUDA_CHECK(cudaEventCreate(&start_v3));
    CUDA_CHECK(cudaEventCreate(&stop_v3));


    CUDA_CHECK(cudaEventRecord(start_v3));
    matMul_GPU_v3<<<grid,block>>>(A_d, B_d, C_d, rows_a, rows_b, cols_b);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    float *C_gpu_h_v3;

    C_gpu_h_v3 = (float *) malloc(bytes_C);

    CUDA_CHECK(cudaMemcpy(C_gpu_h_v3, C_d, bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop_v3));

    CUDA_CHECK(cudaEventSynchronize(stop_v3));

    float gpu_time_v3_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_v3_ms, start_v3, stop_v3));

    cout << "GPU v3 Execution Time (including memcopy): " << gpu_time_v3_ms << " ms" << endl;

    CUDA_CHECK(cudaEventDestroy(start_v3));
    CUDA_CHECK(cudaEventDestroy(stop_v3));

    cout << "GPU v2 Speedup vs GPU v1: " << (gpu_time_v1_ms / gpu_time_v2_ms) << "x" << endl;
    cout<<"Average difference between GPU v1 and GPU v2: "<<avg_diff_bw_elements(C_gpu_h, C_gpu_h_v2, rows_a*cols_b)<<endl;

    cout<<"Speedup of GPU v3 implementation vs GPU v2 implementation: "<< (gpu_time_v2_ms/gpu_time_v3_ms)<< "x" <<endl;
    cout<<"Average difference between GPU v2 and GPU v3: "<<avg_diff_bw_elements(C_gpu_h_v2, C_gpu_h_v3, rows_a*cols_b)<<endl;

    // Cleanup
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    free(A);
    free(B);
    free(C);
    free(C_gpu_h);
    free(C_gpu_h_v2);
    free(C_gpu_h_v3);

    return 0;
}
