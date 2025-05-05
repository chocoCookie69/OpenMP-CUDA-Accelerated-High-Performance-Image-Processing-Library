#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

extern "C" void launchGaussianBlurCUDA(unsigned char* data, int rows, int cols, int channels);
#define kSize 2  // Half of 5x5 kernel
#define kernelWidth (2 * kSize + 1)

__constant__ float kernel[5][5] = {
    {1, 4, 6, 4, 1},
    {4,16,24,16,4},
    {6,24,36,24,6},
    {4,16,24,16,4},
    {1, 4, 6, 4, 1}
};

__global__ void gaussianBlurKernel(unsigned char* img, unsigned char* output, int rows, int cols, int channels) {
    // Grab Pixel and do boundary check (only take pixels which are kSize pixels away from the border)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < kSize || x >= cols - kSize || y < kSize || y >= rows - kSize)
        return;

    //Calculate index of pixel in the GPU array img to place in corresponding pixel in output - when pass 2D array to GPU, it will be flattened to 1D
    int index = (y * cols + x) * channels;
    float b = 0, g = 0, r = 0;

    for (int ky = 0; ky < kernelWidth; ++ky) {
        for (int kx = 0; kx < kernelWidth; ++kx) {
            int ix = x + kx - kSize;
            int iy = y + ky - kSize;
            int iIndex = (iy * cols + ix) * channels;

            float weight = kernel[ky][kx];
            b += img[iIndex + 0] * weight;
            g += img[iIndex + 1] * weight;
            r += img[iIndex + 2] * weight;
        }
    }

    output[index + 0] = static_cast<unsigned char>(b / 256.0f);
    output[index + 1] = static_cast<unsigned char>(g / 256.0f);
    output[index + 2] = static_cast<unsigned char>(r / 256.0f);
}

void launchGaussianBlurCUDA(unsigned char* data, int rows, int cols, int channels) {
    //Initialise memory in GPU
    size_t totalSize = rows * cols * channels;

    unsigned char* d_input, * d_output;
    cudaMalloc(&d_input, totalSize);
    cudaMalloc(&d_output, totalSize);

    cudaMemcpy(d_input, data, totalSize, cudaMemcpyHostToDevice);
    
    //Initialise Grid, Block, threads
    int threads = 16;
    dim3 block(threads, threads);
    dim3 grid((cols + threads-1) / threads, (rows + threads-1) / threads);

    //Call GPU kernel
    gaussianBlurKernel<<<grid, block>>>(d_input, d_output, rows, cols, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_output, totalSize, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}