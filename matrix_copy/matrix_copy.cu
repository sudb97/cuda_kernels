#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace std;

#define ROWS 1078
#define COLS 515
#define TILE 32

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

// Practice: tiled coalesced copy — B[i][j] = A[i][j]
// Convention: x -> column (j), y -> row (i)
__global__ void copy_tiled(const float* A_d, float* B_d, int M, int N)
{
    // TODO: __shared__ tile[TILE][TILE]
    // TODO: compute global row i, col j from blockIdx / threadIdx / blockDim
    // TODO: load tile from A (bounds check), __syncthreads()
    // TODO: store tile to B (same i, j — no transpose swap)

    __shared__ float tile[TILE][TILE];

    size_t rows = threadIdx.y + TILE * blockIdx.y;
    size_t cols = threadIdx.x + TILE * blockIdx.x;

    //Copying the 32 * 32 Tiles from A Matrix
    if (rows < M && cols < N)
        tile[threadIdx.y][threadIdx.x] = A_d[rows * N + cols];

    //__syncthreads(); //Syncs all the threads inside the block to complete

    float values;

    if (rows < M && cols < N)
        values = tile[threadIdx.y][threadIdx.x];
    //__syncthreads();

    //copying the values to the B matrix
    if (rows < M && cols < N)
        B_d[rows * N + cols] = values;
}

void fill_matrix_random(float* matrix, int num_rows, int num_cols)
{
    static bool seeded = false;
    if (!seeded) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        seeded = true;
    }
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            float r = 1.0f + static_cast<float>(std::rand()) /
                                (static_cast<float>(RAND_MAX) / 9.0f);
            matrix[i * num_cols + j] = r;
        }
    }
}

void matrix_copy_CPU(const float* A, float* B, int num_rows, int num_cols)
{
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            B[i * num_cols + j] = A[i * num_cols + j];
        }
    }
}

bool compare_copy_results(const float* cpu, const float* gpu, size_t num_elements,
                          float epsilon = 1e-5f)
{
    for (size_t k = 0; k < num_elements; ++k) {
        if (std::fabs(cpu[k] - gpu[k]) > epsilon) {
            return false;
        }
    }
    return true;
}

void print_first_row_and_column(const char* label, const float* matrix, int num_rows,
                                int num_cols)
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
    const int M = ROWS;
    const int N = COLS;

    const size_t size_bytes = (size_t)M * (size_t)N * sizeof(float);
    const size_t num_elements = (size_t)M * (size_t)N;

    float* A = (float*)malloc(size_bytes);
    fill_matrix_random(A, M, N);

    float* B_cpu = (float*)malloc(size_bytes);
    matrix_copy_CPU(A, B_cpu, M, N);

    float *A_d = nullptr, *B_d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&A_d, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size_bytes));
    CUDA_CHECK(cudaMemcpy(A_d, A, size_bytes, cudaMemcpyHostToDevice));

    // TODO: dim3 block_size — threads per block (e.g. TILE x TILE)
    // TODO: dim3 grid_size  — gridDim.x = ceil(N/TILE), gridDim.y = ceil(M/TILE)
    // copy_tiled<<<grid_size, block_size>>>(A_d, B_d, M, N);

    size_t block_rows = 32;
    size_t block_cols = 32;

    dim3 block_size(block_cols, block_rows); // dim3(x, y) = (cols, rows)

    dim3 grid_size((N + block_cols -1)/block_cols, (M + block_rows -1)/block_rows);

    copy_tiled<<<grid_size,block_size>>>(A_d,B_d,M,N);

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    float* B_gpu = (float*)malloc(size_bytes);
    CUDA_CHECK(cudaMemcpy(B_gpu, B_d, size_bytes, cudaMemcpyDeviceToHost));

    print_first_row_and_column("CPU", B_cpu, M, N);
    print_first_row_and_column("GPU", B_gpu, M, N);

    const bool match = compare_copy_results(B_cpu, B_gpu, num_elements);
    cout << "CPU vs GPU matrix copy match: " << (match ? "true" : "false") << endl;

    free(A);
    free(B_cpu);
    free(B_gpu);
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));

    return match ? 0 : 1;
}
