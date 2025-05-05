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
|       Images         |   Image 1 (res: 1013 x 1048)  |  Image 1 (res: 3417 x 5125)   |
|----------------------|-------------------------------|-------------------------------|
| Filter               |  OpenMP (ms)   |  CUDA (ms)   |  OpenMP (ms)   |   CUDA (ms)  |
| Invert Image         |       171      |     1192     |     2982       |    675       |
| Gaussian Blur        |       1802     |     1188     |     1303       |    30351     |
| Edge Detection       |       325      |     1208     |     4413       |    1618      |
| Adaptive Threshold   |       2419     |     1202     |     38965      |    1885      |

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
