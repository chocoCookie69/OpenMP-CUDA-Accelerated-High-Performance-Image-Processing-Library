#pragma once
#include <opencv2/opencv.hpp>

class ImageProcessor {
public:
    // OpenMP + CUDA functions
    void invertCPU(cv::Mat& image);
    void invertCUDA(cv::Mat& image);

    void applyGaussianBlur(cv::Mat& image, bool useGPU = false);
    void applyEdgeDetection(cv::Mat& image, bool useGPU = false);
    void applyAdaptiveThreshold(cv::Mat& image, bool useGPU = false);
};
