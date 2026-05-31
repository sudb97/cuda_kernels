__global__ void matMul_GPU_v1(float *A, float *B, float *C, size_t rows_a, size_t rows_B, size_t cols_b)
    {
        
         /* Below is the Architecture used for GPU multiplication kernel v1

        1. The number of threads is equal to the rows in the product matrix i.e., rows_a
        2. Each thread will fill the single row of the product matrix 

        Note: The parallelism will be in terms of the number of rows of the product matrix
        */

        size_t tid = static_cast<size_t>(threadIdx.x) + static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x);

        if (tid<rows_a)
        {
        for (size_t i=0; i< cols_b; i++)
            {
                float sum = 0.0f;
                for (size_t j=0; j< rows_B; j++)
                    {
                        sum = sum + A[tid*rows_B + j] * B[j*cols_b +i];
                    }
                C[tid*cols_b + i] = sum;
            }
        }

    }
