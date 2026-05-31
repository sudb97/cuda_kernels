#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

#define N 10000000
#define BLOCK_SIZE 256

__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if (tid < n)
            {
                c[tid]= a[tid] + b[tid];
            }
    }


void fillArrays(float *A, float *B) {
    for (int i = 0; i < N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }
}

void addVectorsCPU(float *h_A, float *h_B, float *h_C)
{
    for (int i=0;i<N;i++)
        {
            h_C[i] = h_A[i] + h_B[i];
        }
}

int main()
{   
    //Allocating host buffers
    float *h_A= (float*)malloc(N*sizeof(float));
    float *h_B= (float*)malloc(N*sizeof(float));
    float *h_C= (float*)malloc(N*sizeof(float));

    //Sending host buffers to fillArrays
    fillArrays(h_A,h_B);

    //Allocating GPU Device buffers
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, N*sizeof(float));
    cudaMalloc((void **)&d_B, N*sizeof(float));
    cudaMalloc((void **)&d_C, N*sizeof(float));

    //calculate the number of blocks
    int num_blocks = (N + BLOCK_SIZE -1)/BLOCK_SIZE;

    //transfer the memory from host to device
    cudaMemcpy(d_A,h_A,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,N*sizeof(float),cudaMemcpyHostToDevice);

    //Warmup the cpu and gpu computes
    for (int i=0; i<3;i++)
        {
            addVectorsCPU(h_A,h_B,h_C);
            addVectorsGPU<<<num_blocks,BLOCK_SIZE>>>(d_A,d_B,d_C,N);
            cudaDeviceSynchronize();
        }
    
  
    //Computing the vector addition on CPU 20 times and calculating the average time for the CPU computation
    auto start_cpu = chrono::high_resolution_clock::now();
    
    for (int i=0; i<20;i++)
        {
            addVectorsCPU(h_A,h_B,h_C);
        }
    
    auto stop_cpu = chrono::high_resolution_clock::now();
    auto duration_cpu = chrono::duration_cast<chrono::nanoseconds>(stop_cpu - start_cpu);
    auto time_cpu= duration_cpu.count()/20;
    cout<<"CPU vector addition time: "<<time_cpu<<" ns"<<endl;

    
    auto start_gpu = chrono::high_resolution_clock::now();
    //Compute the vector addition 20 times on GPU and calculate the average time taken on GPU
    for (int i=0;i<20;i++)
        {
            addVectorsGPU<<<num_blocks,BLOCK_SIZE>>>(d_A,d_B,d_C,N);
            cudaDeviceSynchronize();
        }
    auto stop_gpu = chrono::high_resolution_clock::now();
    auto duration_gpu = chrono::duration_cast<chrono::nanoseconds>(stop_gpu - start_gpu);
    auto time_gpu= duration_gpu.count()/20;
    cout<<"GPU vector addition time: "<<time_gpu<<" ns"<<endl;
    
    cout<<"Speedup in CPU vs GPU time: "<<time_cpu/time_gpu<<endl;
    //compare the average difference between the added elements of CPU and GPU
    
    //Copying the results from GPU to CPU
    float *h_C_gpu = (float*)malloc(N*sizeof(float));

    cudaMemcpy(h_C_gpu,d_C,N*sizeof(float),cudaMemcpyDeviceToHost);

    float diff=0;
    for (int i=0;i<N;i++)
        {
            diff = diff + fabs(h_C[i]-h_C_gpu[i]);
        }

    //Top 5 CPU and GPU addition vectors
    for (int i=0;i<5;i++)
        {
            cout<<"Element: "<<i<<" CPU: "<<h_C[i]<<" GPU: "<<h_C_gpu[i]<<endl; 
        }
    
    cout<<"The average difference between the CPU and GPU computed vector addition is: "<<diff/N<<endl;

    return 0;

}