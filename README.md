# OpenM v/s CUDA Accelerated High Performance Image Processing Library

This project is a high-performance image processing library written in C++ and CUDA, supporting real-time filtering, transformations, and noise reduction. It is also Python-bindable via Pybind11.

## Features
- **Real-time Filtering**: Gaussian Blur, Edge Detection, Adaptive Thresholding.
- **Performance Optimizations**:
  - OpenMP multithreading on CPU.
  - CUDA acceleration for GPU-based filters.
- **Modular API** for easy integration into C++ or Python.

## Tech Stack
- C++
- CUDA
- OpenCV
- OpenMP
- Pybind11 (Python binding)
- CMake

## Benchmark Results
| Filter               | Image 1 (1013×1048)<br>OpenMP (ms) | Image 1 (1013×1048)<br>CUDA (ms) | Image 2 (3417×5125)<br>OpenMP (ms) | Image 2 (3417×5125)<br>CUDA (ms) |
|----------------------|-------------------------------|------------------------------|-------------------------------|------------------------------|
| Invert Image         | 171                           | 1192                         | 2982                          | 675                          |
| Gaussian Blur        | 1802                          | 1188                         | 29910                         | 1738                         |
| Edge Detection       | 325                           | 1208                         | 4413                          | 1618                         |
| Adaptive Threshold   | 2419                          | 1202                         | 38965                         | 1885                         |

## Screenshot sample from the experiments
![low res](https://github.com/user-attachments/assets/1bea891b-b715-4dba-9a86-f7782467d4d2)

![high res](https://github.com/user-attachments/assets/2a1c2250-1f6f-4c3e-95c1-3a42ab048f82)

## Observations from Benchmark Results

1. **CUDA generally outperforms OpenMP on large images.**
   - For the **high-resolution image**, CUDA shows **significant speedups**:
     - **Adaptive Threshold**: ~20× faster
     - **Gaussian Blur**: ~17× faster
     - **Invert Image**: ~4.4× faster
     - **Edge Detection**: ~2.7× faster

2. **On smaller images (1013×1048), OpenMP sometimes performs better.**
   - CUDA shows **higher overhead** on small inputs:
     - **Invert Image**: OpenMP is faster (171 ms vs. 1192 ms)
     - **Gaussian Blur**: CUDA slightly outperforms OpenMP (1188 ms vs. 1802 ms)
   - Indicates that CUDA's launch/setup overhead **dominates when image size is small**.

## Installation Requirements

### C++ (Compiler)
- Install [Visual Studio](https://visualstudio.microsoft.com/) with **Desktop development with C++** workload.
- Ensure `cl.exe` and build tools are in your system PATH.

### CUDA
- Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (Recommended: CUDA 12.x).
- After installation, add to your system's `Path` environment variable.

### OpenCV
- Download OpenCV from [https://opencv.org/releases](https://opencv.org/releases) and extract it.
- Configure the following in your project or CMake:
  - **Include directory**: `opencv/build/include`
  - **Library directory**: `opencv/build/x64/vc16/lib`
  - **DLL runtime directory**: `opencv/build/x64/vc16/bin`
- Add the DLL path to your system `Path` environment variable.

### OpenMP
- Comes bundled with Visual Studio C++ compiler.
- No separate installation needed.
- Use `#pragma omp parallel for` in code and pass `/openmp` flag if needed.

### Pybind11 (for Python binding)
Install via pip:
```bash
pip install pybind11
