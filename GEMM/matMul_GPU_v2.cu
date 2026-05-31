__global__ void matMul_GPU_v2(float *A, float *B, float *C, size_t rows_a, size_t rows_B, size_t cols_b)
    {
        /* In this architechture each thread will fill the single element of the product matrix
        
        Note: The tid_x will give the row of the product matrix and tid_y will give the column of the product matrix
        
        */
        size_t tid_x =  static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
        size_t tid_y =  static_cast<size_t>(blockIdx.y) * static_cast<size_t>(blockDim.y) + static_cast<size_t>(threadIdx.y);

        if (tid_x < rows_a && tid_y < cols_b)
            {
            float sum = 0.0f;

            for (size_t i=0; i< rows_B; i++)
                {
                    sum = sum + A[tid_x*rows_B + i] * B[i*cols_b + tid_y];
                }

            C[tid_x* cols_b + tid_y] = sum;
            }
    }
