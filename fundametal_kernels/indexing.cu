#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 32 //The number of threads in a block
#define NUM_ELEMENTS 1024 //The number of elements in the array

__global__ void cuda_kernel()
    {   
        int block_offset = blockIdx.x + blockIdx.y * 
        int tid= blockIdx.x * blockDim.x + threadIdx.x;

        if(tid < 1)
            {
                printf("Block Dim x: %d\n", blockDim.x);
                printf("Block Dim y: %d\n", blockDim.y);
                printf("Block Dim z: %d\n", blockDim.z);

                printf("Grid Dim x: %d\n", gridDim.x);
                printf("Grid Dim y: %d\n", gridDim.y);
                printf("Grid Dim z: %d\n", gridDim.z);
            }
    }

int main()
    {
        int num_blocks = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        dim3 blockspergrid(2,2,2);
        dim3 threadsperblock(2,2,8);

        cuda_kernel<<<blockspergrid, threadsperblock>>> ();
        cudaDeviceSynchronize();

        return 0;
    }