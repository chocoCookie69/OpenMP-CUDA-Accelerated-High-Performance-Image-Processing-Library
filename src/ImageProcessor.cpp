#include "ImageProcessor.h"
#include <omp.h>
#include <iostream>

// CPU Invert using OpenMP
void ImageProcessor::invertCPU(cv::Mat& image) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            for (int c = 0; c < image.channels(); ++c) {
                image.at<cv::Vec3b>(i, j)[c] = 255 - image.at<cv::Vec3b>(i, j)[c];
            }
        }
    }
}

// Forward declaration (defined in .cu)
extern "C" void launchInvertCUDA(uchar* data, int rows, int cols, int channels);
void ImageProcessor::invertCUDA(cv::Mat& image) {
    if (!image.isContinuous()) {
        std::cerr << "Image data must be continuous for CUDA processing.\n";
        return;
    }
    launchInvertCUDA(image.data, image.rows, image.cols, image.channels());
}

// Forward declaration (defined in .cu)
extern "C" void launchGaussianBlurCUDA(uchar* data, int rows, int cols, int channels);
void ImageProcessor::applyGaussianBlur(cv::Mat& image, bool useGPU) {
    if (useGPU) {
        if (!image.isContinuous()) {
            std::cerr << "Image must be continuous for CUDA processing.\n";
            return;
        }
        launchGaussianBlurCUDA(image.data, image.rows, image.cols, image.channels());
        return;
    }
    cv::Mat input = image.clone();
    cv::Mat output = image.clone();

    // 5x5 Gaussian Kernel
    const float kernel[5][5] = {
        {1,  4,  6,  4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1,  4,  6,  4, 1}
    };
    const float kernelSum = 256.0f;

    const int kSize = 2;
#pragma omp parallel for collapse(2)
    for (int i = kSize; i < input.rows - kSize; ++i) {
        for (int j = kSize; j < input.cols - kSize; ++j) {
            cv::Vec3f sum(0, 0, 0);
            for (int ki = 0; ki < 5; ++ki) {
                for (int kj = 0; kj < 5; ++kj) {
                    cv::Vec3b pixel = input.at<cv::Vec3b>(i + ki - kSize, j + kj - kSize);
                    float weight = kernel[ki][kj];
                    sum[0] += pixel[0] * weight;
                    sum[1] += pixel[1] * weight;
                    sum[2] += pixel[2] * weight;
                }
            }
            output.at<cv::Vec3b>(i, j)[0] = static_cast<uchar>(sum[0] / kernelSum);
            output.at<cv::Vec3b>(i, j)[1] = static_cast<uchar>(sum[1] / kernelSum);
            output.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(sum[2] / kernelSum);
        }
    }

    image = output;
}

extern "C" void launchEdgeDetectionCUDA(unsigned char* data, int rows, int cols, int channels);
void ImageProcessor::applyEdgeDetection(cv::Mat& image, bool useGPU) {
    if (useGPU) {
        if (!image.isContinuous()) {
            std::cerr << "Image must be continuous for CUDA processing.\n";
            return;
        }
        launchEdgeDetectionCUDA(image.data, image.rows, image.cols, image.channels());
        return;
    }

    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image.clone();

    cv::Mat edge(gray.size(), CV_8UC1);

    const int kSize = 1;
    const int rows = gray.rows;
    const int cols = gray.cols;

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

#pragma omp parallel for collapse(2)
    for (int i = kSize; i < rows - kSize; ++i) {
        for (int j = kSize; j < cols - kSize; ++j) {
            int sumX = 0, sumY = 0;

            for (int ki = -kSize; ki <= kSize; ++ki) {
                for (int kj = -kSize; kj <= kSize; ++kj) {
                    int pixel = gray.at<uchar>(i + ki, j + kj);
                    sumX += Gx[ki + kSize][kj + kSize] * pixel;
                    sumY += Gy[ki + kSize][kj + kSize] * pixel;
                }
            }

            int mag = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
            edge.at<uchar>(i, j) = static_cast<uchar>(std::min(mag, 255));
        }
    }

    cv::cvtColor(edge, image, cv::COLOR_GRAY2BGR); // convert back for visual consistency
}
extern "C" void launchAdaptiveThreshold(unsigned char* data, int rows, int cols, int channels);
void ImageProcessor::applyAdaptiveThreshold(cv::Mat& image, bool useGPU) {
    if (useGPU) {
        if (!image.isContinuous()) {
            std::cerr << "Image must be continuous for CUDA processing.\n";
            return;
        }
        launchAdaptiveThreshold(image.data, image.rows, image.cols, image.channels());
        return;
    }

    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image.clone();

    const int blockSize = 11; // 11x11 local window
    const int kSize = blockSize / 2;
    const int rows = gray.rows;
    const int cols = gray.cols;

    cv::Mat output = gray.clone();
#pragma omp parallel for collapse(2)
    for (int i = kSize; i < rows - kSize; ++i) {
        for (int j = kSize; j < cols - kSize; ++j) {
            int sum = 0;
            for (int ki = -kSize; ki <= kSize; ++ki) {
                for (int kj = -kSize; kj <= kSize; ++kj) {
                    sum += gray.at<uchar>(i + ki, j + kj);
                }
            }

            int count = blockSize * blockSize;
            int mean = sum / count;
            uchar pixel = gray.at<uchar>(i, j);

            output.at<uchar>(i, j) = (pixel > mean - 5) ? 255 : 0;
        }
    }
    cv::cvtColor(output, image, cv::COLOR_GRAY2BGR); // Convert back for consistency
}

