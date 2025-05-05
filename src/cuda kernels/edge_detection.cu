#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <iostream>


extern "C" void launchEdgeDetectionCUDA(unsigned char* data, int rows, int cols, int channels);
#define kSize 1
#define kernelWidth (2 * kSize + 1)

__constant__ int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

__global__ void sobelKernel(unsigned char* input, unsigned char* output, int rows, int cols, int channels) {
    // Grab Pixel and do boundary check (only take pixels which are kSize pixels away from the border)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < kSize || x >= cols - kSize || y < kSize || y >= rows - kSize)
        return;
    
    //Calculate index of pixel in the GPU array img to place in corresponding pixel in output - when pass 2D array to GPU, it will be flattened to 1D
    int outIdx = (y * cols + x) * channels;
    int sumX = 0, sumY = 0;

    for (int ky = 0; ky < kernelWidth; ++ky) {
        for (int kx = 0; kx < kernelWidth; ++kx) {
            int ix = x + kx - kSize; // ix and iy are the neighbours of center pixel(x,y)
            int iy = y + ky - kSize;
            int index = (iy * cols + ix) * channels;// index of pixel value in the array inside GPU

            unsigned char gray = 0;
            if (channels == 3) {
                unsigned char b = input[index + 0];
                unsigned char g = input[index + 1];
                unsigned char r = input[index + 2];
                gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
            }
            else {
                gray = input[index];
            }

            sumX += Gx[ky][kx] * gray;
            sumY += Gy[ky][kx] * gray;
        }
    }

    int intensity_magnitude = (int)sqrtf(sumX * sumX + sumY * sumY);
    intensity_magnitude = intensity_magnitude > 255 ? 255 : intensity_magnitude;

    
    if (channels == 3) {
        output[outIdx + 0] = intensity_magnitude;
        output[outIdx + 1] = intensity_magnitude;
        output[outIdx + 2] = intensity_magnitude;
    }
    else {
        output[outIdx] = intensity_magnitude;
    }
}

void launchEdgeDetectionCUDA(unsigned char* data, int rows, int cols, int channels) {
    //Initialise memory in GPU
    size_t size = rows * cols * channels;

    unsigned char* d_input, * d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, data, size, cudaMemcpyHostToDevice);
    
    //Initialise Grid, Block, threads
    int threads = 16;
    dim3 block(threads, threads);
    dim3 grid((cols + threads - 1) / threads, (rows + threads - 1) / threads);

    //Call GPU kernel
    sobelKernel<<<grid, block>>>(d_input, d_output, rows, cols, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}