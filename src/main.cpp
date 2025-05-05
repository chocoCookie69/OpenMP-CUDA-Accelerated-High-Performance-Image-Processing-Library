// ImageProcesingLibrary.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>
#include "ImageProcessor.h"
#include <chrono>


int main() {
    /*  //Invert filter
    
        // Load the image
        cv::Mat image = cv::imread("E:/Program Files/Image Processing Library/ImageProcesingLibrary/ImageProcesingLibrary/data/Haardhik.jpg");
        if (image.empty()) {
            std::cerr << "Error loading image" << std::endl;
            return -1;
        }

        ImageProcessor ip;
    
        // CUDA
        std::cout << "Running invertCUDA filter...\n";
    
        auto startCUDA = std::chrono::high_resolution_clock::now();
        ip.invertCUDA(image);  // or ip.invertCPU(img)
        auto endCUDA = std::chrono::high_resolution_clock::now();

        auto durationCUDA = std::chrono::duration_cast<std::chrono::milliseconds>(endCUDA - startCUDA);
        std::cout << "Execution time using CUDA: " << durationCUDA.count() << " ms\n";

        // OpenMP
        std::cout << "Running invertOpenMP filter...\n";

        auto startOpenMP = std::chrono::high_resolution_clock::now();
        ip.invertCPU(image);  // or ip.invertCPU(img)
        auto endOpenMP = std::chrono::high_resolution_clock::now();

        auto durationOpenMP = std::chrono::duration_cast<std::chrono::milliseconds>(endOpenMP - startOpenMP);
        std::cout << "Execution time using OpenMP: " << durationOpenMP.count() << " ms\n";
    */
      //GaussianBlur

        // Load the image
        cv::Mat image = cv::imread("E:/Program Files/Image Processing Library/ImageProcesingLibrary/ImageProcesingLibrary/data/Haardhik.jpg");
        if (image.empty()) {
            std::cerr << "Error loading image" << std::endl;
            return -1;
        }

        ImageProcessor ip;
        cv::Mat cpuImage = image.clone();
        cv::Mat gpuImage = image.clone();

        std::cout << "Running OpenMP Blur...\n";
        auto startCPU = std::chrono::high_resolution_clock::now();
        ip.applyGaussianBlur(cpuImage, false);  // CPU
        auto endCPU = std::chrono::high_resolution_clock::now();

        std::cout << "Running CUDA Blur...\n";
        auto startGPU = std::chrono::high_resolution_clock::now();
        ip.applyGaussianBlur(gpuImage, true);   // GPU
        auto endGPU = std::chrono::high_resolution_clock::now();

        // Benchmarking
        auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU).count();
        auto gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(endGPU - startGPU).count();
        std::cout << "CPU (OpenMP) Time: " << cpuTime << " ms\n";
        std::cout << "GPU (CUDA) Time:  " << gpuTime << " ms\n";
        
        // Restrict Output Image to screen size
        //cv::namedWindow("Blurred Image", cv::WINDOW_NORMAL); // allow resize
        //cv::resizeWindow("Blurred Image", 960, 540);          // scale down to 50%
        
        // Display Output
        cv::imshow("CPU Blur", cpuImage);
        cv::imshow("GPU Blur", gpuImage);
    
    
    /*  //Edge Detection
        
        cv::Mat image = cv::imread("E:/Program Files/Image Processing Library/ImageProcesingLibrary/ImageProcesingLibrary/data/test1.jpg");
        if (image.empty()) {
            std::cerr << "Failed to load image\n";
            return -1;
        }

        ImageProcessor ip;
        cv::Mat cpuImage = image.clone();
        cv::Mat gpuImage = image.clone();
        // Benchmark CPU (OpenMP)
        std::cout << "Running Edge Detection: OpenMP...\n";
        auto startCPU = std::chrono::high_resolution_clock::now();
        ip.applyEdgeDetection(cpuImage, false);
        auto endCPU = std::chrono::high_resolution_clock::now();

        // Benchmark GPU (CUDA)
        std::cout << "Running Edge Detection: CUDA...\n";
        auto startGPU = std::chrono::high_resolution_clock::now();
        ip.applyEdgeDetection(gpuImage, true);
        auto endGPU = std::chrono::high_resolution_clock::now();

        // Calculate durations
        auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU).count();
        auto gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(endGPU - startGPU).count();

        std::cout << "OpenMP Time: " << cpuTime << " ms\n";
        std::cout << "CUDA Time:   " << gpuTime << " ms\n";
        
        // Show result
        cv::imshow("Edge Detection (CPU)", cpuImage);
        cv::imshow("Edge Detection (CUDA)", gpuImage);
    */

    /*  //Adaptive Thresholding
        
        cv::Mat image = cv::imread("E:/Program Files/Image Processing Library/ImageProcesingLibrary/ImageProcesingLibrary/data/test1.jpg");
        if (image.empty()) {
            std::cerr << "Failed to load image\n";
            return -1;
        }

        ImageProcessor ip;

        // Clone for both CPU and GPU tests
        cv::Mat cpuImage = image.clone();
        cv::Mat gpuImage = image.clone();

        // Benchmark CPU (OpenMP) version
        std::cout << "Running Adaptive Threshold: OpenMP...\n";
        auto startCPU = std::chrono::high_resolution_clock::now();
        ip.applyAdaptiveThreshold(cpuImage, false);  // CPU version
        auto endCPU = std::chrono::high_resolution_clock::now();

        // Benchmark GPU (CUDA) version
        std::cout << "Running Adaptive Threshold: CUDA...\n";
        auto startGPU = std::chrono::high_resolution_clock::now();
        ip.applyAdaptiveThreshold(gpuImage, true);   // GPU version
        auto endGPU = std::chrono::high_resolution_clock::now();

        // Calculate durations
        auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU).count();
        auto gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(endGPU - startGPU).count();

        std::cout << "OpenMP Time: " << cpuTime << " ms\n";
        std::cout << "CUDA Time:   " << gpuTime << " ms\n";

        // Show results
        cv::imshow("Adaptive Threshold (CPU)", cpuImage);
        cv::imshow("Adaptive Threshold (CUDA)", gpuImage);
    */

    /*  // Real Time video processing of all the above features

        // Load the image
        cv::VideoCapture cap(0);  // 0 = default webcam
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open webcam\n";
            return -1;
        }

        ImageProcessor ip;
        std::cout << "Webcam opened successfully. Starting real-time processing...\n";

        while (true) {
            cv::Mat frame;
            cap >> frame;  // Grab a new frame

            if (frame.empty()) {
                std::cerr << "Error: Blank frame grabbed\n";
                break;
            }

            cv::Mat processedFrame = frame.clone();

            // Choose any filter you want to apply here:
             ip.applyGaussianBlur(processedFrame, true);         // Real-time Gaussian Blur (CUDA)
            // ip.applyEdgeDetection(processedFrame, true);      // Real-time Edge Detection (CUDA)
            // ip.applyAdaptiveThreshold(processedFrame, true);  // Real-time Adaptive Thresholding (CUDA)

            cv::imshow("Real-Time Processing", processedFrame);

            // Break the loop if user presses ESC key (27)
            if (cv::waitKey(1) == 27) break;
        }

        cap.release();
        cv::destroyAllWindows();
    */
    
        // Exit
        std::cout << "Showing result window. Press any key to close.";
        cv::waitKey(0);
        return 0;
}
