#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <iostream>
#define blockSize 11
#define kSize (blockSize / 2)
extern "C" void launchAdaptiveThreshold(unsigned char* data, int rows, int cols, int channels);

__global__ void adaptiveThresholdKernel(unsigned char* input, unsigned char* output, int rows, int cols, int channels) {
    // Grab Pixel and do boundary check (only take pixels which are kSize pixels away from the border)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < kSize || x >= cols - kSize || y < kSize || y >= rows - kSize)
        return;

    int sum = 0;

    for (int ky = -kSize; ky <= kSize; ++ky) {
        for (int kx = -kSize; kx <= kSize; ++kx) {
            int ix = x + kx;
            int iy = y + ky;
            int index = (iy * cols + ix) * channels;

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

            sum += gray;
        }
    }

    int count = blockSize * blockSize;
    int mean = sum / count;
    
    //Calculate index of pixel in the GPU array img to place in corresponding pixel in output - when pass 2D array to GPU, it will be flattened to 1D
    int centerIdx = (y * cols + x) * channels;

    unsigned char centerPixel = 0;
    if (channels == 3) {
        unsigned char b = input[centerIdx + 0];
        unsigned char g = input[centerIdx + 1];
        unsigned char r = input[centerIdx + 2];
        centerPixel = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
    else {
        centerPixel = input[centerIdx];
    }

    //Compare mean with existing pixel value
    unsigned char result = (centerPixel > mean - 5) ? 255 : 0;

    if (channels == 3) {
        output[centerIdx + 0] = result;
        output[centerIdx + 1] = result;
        output[centerIdx + 2] = result;
    }
    else {
        output[centerIdx] = result;
    }
}

void launchAdaptiveThreshold(unsigned char* data, int rows, int cols, int channels) {
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
    adaptiveThresholdKernel<<<grid, block>>>(d_input, d_output, rows, cols, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}