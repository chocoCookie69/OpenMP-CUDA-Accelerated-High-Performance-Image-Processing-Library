# OpenM v/s CUDA Accelerated High Performance Image Processing Library

This project is a high-performance image processing library written in C++ and CUDA, supporting real-time filtering, transformations, and noise reduction. It is also Python-bindable via Pybind11.

##Features
- **Real-time Filtering**: Gaussian Blur, Edge Detection, Adaptive Thresholding.
- **Performance Optimizations**:
  - OpenMP multithreading on CPU.
  - CUDA acceleration for GPU-based filters.
- **Modular API** for easy integration into C++ or Python.

##Tech Stack
- C++
- CUDA
- OpenCV
- OpenMP
- Pybind11 (Python binding)
- CMake

##Benchmark Results
| Filter               | OpenMP (ms) | CUDA (ms) |
|----------------------|-------------|------------|
| Invert Image         | ~42000      | ~2000      |
| Gaussian Blur        | X           | Y          |
| Edge Detection       | X           | Y          |
| Adaptive Threshold   | X           | Y          |
