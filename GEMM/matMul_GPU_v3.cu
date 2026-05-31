__global__ void matMul_GPU_v3(float *A, float *B, float *C, size_t rows_a, size_t rows_B, size_t cols_b)
    {
        /* In this architechture each thread will fill the single element of the product matrix
        
        Note: The columns will drive the which column of B for a perticular row needed for the multiplication at ith instant, row will drive which row of element A will be fetched at the warp level
        
        */
        size_t columns =  static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x); //It will drive the column of B
        size_t rows =  static_cast<size_t>(blockIdx.y) * static_cast<size_t>(blockDim.y) + static_cast<size_t>(threadIdx.y); //It will drive the rows of A

        if (rows < rows_a && columns < cols_b)
            {
            float sum = 0.0f;

            for (size_t i=0; i< rows_B; i++)
                {
                    sum = sum + A[rows*rows_B + i] * B[i*cols_b + columns];
                }

            C[rows* cols_b + columns] = sum;
            }
    }
