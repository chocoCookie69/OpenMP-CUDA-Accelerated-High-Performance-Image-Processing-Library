#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "../include/ImageProcessor.h"

namespace py = pybind11;

class PyImageProcessor {
private:
    ImageProcessor processor;

    py::array_t<unsigned char> process(py::array_t<unsigned char> input, std::function<void(cv::Mat&)> func) {
        auto buf = input.request();
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        int channels = buf.shape[2];

        cv::Mat image(rows, cols, (channels == 3) ? CV_8UC3 : CV_8UC1, static_cast<uchar*>(buf.ptr));
        func(image);
        return input;
    }
public:
    PyImageProcessor() : processor() {}

    py::array_t<unsigned char> invert_cpu(py::array_t<unsigned char> input) {
        return process(input, [&](cv::Mat& img) { processor.invertCPU(img); });
    }

    py::array_t<unsigned char> invert_cuda(py::array_t<unsigned char> input) {
        return process(input, [&](cv::Mat& img) { processor.invertCUDA(img); });
    }

    py::array_t<unsigned char> gaussian_blur(py::array_t<unsigned char> input, bool useGPU = false) {
        return process(input, [&](cv::Mat& img) { processor.applyGaussianBlur(img, useGPU); });
    }

    py::array_t<unsigned char> edge_detection(py::array_t<unsigned char> input, bool useGPU = false) {
        return process(input, [&](cv::Mat& img) { processor.applyEdgeDetection(img, useGPU); });
    }

    py::array_t<unsigned char> adaptive_threshold(py::array_t<unsigned char> input, bool useGPU = false) {
        return process(input, [&](cv::Mat& img) { processor.applyAdaptiveThreshold(img, useGPU); });
    }
};

PYBIND11_MODULE(image_processing, m) {
    py::class_<PyImageProcessor>(m, "ImageProcessor")
        .def(py::init<>())
        .def("invert_cpu", &PyImageProcessor::invert_cpu)
        .def("invert_cuda", &PyImageProcessor::invert_cuda)
        .def("gaussian_blur", &PyImageProcessor::gaussian_blur)
        .def("edge_detection", &PyImageProcessor::edge_detection)
        .def("adaptive_threshold", &PyImageProcessor::adaptive_threshold);
}
/*
#include <pybind11/pybind11.h>
int add(int i, int j) { return i + j; }
PYBIND11_MODULE(example, m) {
    m.def("add", &add);
}
*/