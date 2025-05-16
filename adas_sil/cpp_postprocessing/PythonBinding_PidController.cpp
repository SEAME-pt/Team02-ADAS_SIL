#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "PidController.hpp"

namespace py = pybind11;

// Helper function to convert numpy array to cv::Mat (reuse from your existing code)
cv::Mat numpy_to_mat(py::array_t<uint8_t>& array) {
    py::buffer_info info = array.request();
    
    int rows = static_cast<int>(info.shape[0]);
    int cols = static_cast<int>(info.shape[1]);
    int type = CV_8UC3;  // Assuming BGR image
    
    if (info.ndim == 2) {
        type = CV_8UC1;  // Grayscale image
    }
    
    cv::Mat image(rows, cols, type, info.ptr);
    return image.clone();  // Clone to ensure we own the data
}

// Helper function to convert cv::Mat to numpy array (reuse from your existing code)
py::array_t<uint8_t> mat_to_numpy(const cv::Mat& image) {
    py::array_t<uint8_t> array;
    
    if (image.channels() == 1) {
        array = py::array_t<uint8_t>({image.rows, image.cols});
    } else {
        array = py::array_t<uint8_t>({image.rows, image.cols, image.channels()});
    }
    
    auto buffer = array.request();
    uint8_t* ptr = static_cast<uint8_t*>(buffer.ptr);
    std::memcpy(ptr, image.data, image.total() * image.elemSize());
    
    return array;
}

// Helper function to convert cv::Point to Python tuple
py::tuple point_to_tuple(const cv::Point& p) {
    return py::make_tuple(p.x, p.y);
}

PYBIND11_MODULE(pid_controller_py, m) {
    m.doc() = "PID controller for lane following in CARLA";
    
    // Bind the PID Controller
    py::class_<PidController>(m, "PidController")
        // Default constructor (already implemented in your code)
        .def(py::init<>())
        
        // Core methods
        .def("init", &PidController::init)
        .def("steeringPID", &PidController::steeringPID)
        .def("speedAdjustment", &PidController::speedAdjustment)
        
        // Autonomous state methods
        .def("setAutonomousDriveState", &PidController::setAutonomousDriveState)
        .def("getAutonomousDriveState", &PidController::getAutonomousDriveState)
        
        // CARLA-specific methods
        .def("updateCarlaControl", &PidController::updateCarlaControl)
        .def("setCameraError", &PidController::setCameraError);

}