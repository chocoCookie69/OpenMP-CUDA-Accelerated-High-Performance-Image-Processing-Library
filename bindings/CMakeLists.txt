cmake_minimum_required(VERSION 3.14)
project(image_processing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)


include_directories(
    ../include
    ../src
    ../src/cuda_kernels
    pybind11/include
    "E:/Program Files/OpenCV/opencv/build/include"
    "C:/Program Files/Python312/include"
    "E:/Program Files/CUDA_Installation/include"
)
link_directories(
   "E:/Program Files/OpenCV/opencv/build/x64/vc16/lib"
   "C:/Program Files/Python312/libs"
   "E:/Program Files/CUDA_Installation/lib/x64"

)

file(GLOB CPP_SOURCES
    "../src/*.cpp"
)
file(GLOB  CUDA_SOURCES
    "../src/cuda kernels/*.cu"
)

add_library(image_processing MODULE bindings.cpp ${CPP_SOURCES} ${CUDA_SOURCES})

target_link_libraries(image_processing PRIVATE 
    opencv_world4110d.lib
    cudart.lib
    python312.lib
)
set_target_properties(image_processing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".pyd"
)