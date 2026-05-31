#include <cuda_runtime.h>
#include <stdio.h>

__global__ void indexing_single_block(float *A_d)
{
    //This kernel is designed to index a single block of the array A
    int kernel_number = threadIdx.y * blockDim.y + threadIdx.x;

    printf("Kernel Number: %d \n", kernel_number);
    printf("Value stored in the index A[%d]: %f \n",kernel_number, A_d[kernel_number]);

}

int main()
{
    dim3 grid(10,10);
    dim3 block(32,32);

    float *A;

    int array_rows=32;
    int array_columns=32;

    A = (float*) malloc(array_rows*array_columns*sizeof(float));

    for (int i=0; i<array_rows*array_columns; i++)
        A[i]=i;

    int block_rows = 32;
    int block_cols = 32;

    dim3 block_size(block_rows,block_cols);

    int grid_rows = (array_rows + block_rows -1 )/block_rows;
    int grid_columns = (array_columns + block_cols - 1 )/block_cols;
    
    dim3 grid_size(grid_rows, grid_columns);

    //Move the Array A to the GPU
    float *A_d;
    cudaMalloc((void **)&A_d, array_rows*array_columns*sizeof(float));
    cudaMemcpy(A_d, A, array_rows*array_columns*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    indexing_single_block<<<1, block_size>>>(A_d);
    cudaDeviceSynchronize();

    return 0;
}