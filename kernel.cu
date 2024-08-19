
#ifndef __CUDACC__
  #define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <chrono>
#include <stdio.h>

// Thread block size
#define BLOCK_SIZE 16

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernelGeneral(const Matrix, const Matrix, Matrix);
__global__ void MatMulKernelPartitioned(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C, bool general)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    if( general )
        MatMulKernelGeneral<<<dimGrid, dimBlock >>>(d_A, d_B, d_C);
    else
        MatMulKernelPartitioned<<<dimGrid, dimBlock >>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernelGeneral(Matrix A, Matrix B, Matrix C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// Get a matrix element
//__device__ float GetElement(const Matrix A, int row, int col)
//{
//    return A.elements[row * A.stride + col];
//}
//// Set a matrix element
//__device__ void SetElement(Matrix A, int row, int col, float value)
//{
//    A.elements[row * A.stride + col] = value;
//}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernelPartitioned(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Collective load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub.elements[row * Asub.stride + col]; //GetElement(Asub, row, col);
        Bs[row][col] = Bsub.elements[row * Bsub.stride + col]; //GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    Csub.elements[row * Csub.stride + col] = Cvalue; //SetElement(Csub, row, col, Cvalue);
}

int main()
{
    const size_t sz = 64*BLOCK_SIZE;

    Matrix a;
    a.width = a.height = sz;
    a.elements = new float[sz*sz];
    Matrix b;
    b.width = b.height = sz;
    b.elements = new float[sz * sz];
    Matrix c;
    c.width = c.height = sz;
    c.elements = new float[sz * sz];

    cudaError_t cudaStatus;
    int cudaDevice(0);
    int rc(0);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaInitDevice(cudaDevice, 0, 0);//cudaSetDevice(cudaDevice );
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaSetDevice #%d failed!  Do you have a CUDA-capable GPU installed?", cudaDevice);
        fprintf(stderr, "cudaInitDevice #%d failed!  Do you have a CUDA-capable GPU installed?", cudaDevice);
        rc = 1;
        goto end;
    }

    // Measured multiplication
    auto ns0 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    MatMul( a, b, c, true );
    auto ns1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    MatMul( a, b, c, false );
    auto ns2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    fprintf(stdout, "MatMul %lld ns (%llu-%llu)\n", ns1-ns0, ns1, ns0);
    fprintf(stdout, "MatMul %lld ns (%llu-%llu)\n", ns2-ns1, ns2, ns1);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MatMul launch failed #%d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        rc = 2;
        goto end;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d\n", cudaStatus);
        rc = 3;
        goto end;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        rc = 4;
        goto end;
    }

end:
    delete[] a.elements;
    delete[] b.elements;
    delete[] c.elements;

    return rc;
}
