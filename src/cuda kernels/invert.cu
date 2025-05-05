#include <cuda_runtime.h>
#include <device_launch_parameters.h>
extern "C" void launchInvertCUDA(unsigned char* data, int rows, int cols, int channels);

__global__ void invertKernel(unsigned char* data, int rows, int cols, int channels) {
    // Grab Pixel and do boundary check
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    //Calculate index of pixel in the GPU array img to place in corresponding pixel in output - when pass 2D array to GPU, it will be flattened to 1D
    int idx = (y * cols + x) * channels;
    for (int c = 0; c < channels; ++c) {
        data[idx + c] = 255 - data[idx + c];
    }
}

void launchInvertCUDA(unsigned char* data, int rows, int cols, int channels) {
    //Initialise Grid, Block, threads
    int threads = 16;
    dim3 blockSize(threads, threads);
    dim3 gridSize((cols + threads - 1) / threads, (rows + threads - 1) / threads);

    //Initialise memory in GPU
    unsigned char* d_data;
    size_t totalBytes = rows * cols * channels;

    cudaMalloc(&d_data, totalBytes);
    cudaMemcpy(d_data, data, totalBytes, cudaMemcpyHostToDevice);

    //Call GPU kernel
    invertKernel<<<gridSize, blockSize>>>(d_data, rows, cols, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, totalBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
