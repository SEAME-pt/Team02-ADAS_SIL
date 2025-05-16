#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "LaneDetector.hpp"

namespace py = pybind11;
// Helper function to convert numpy array to cv::Mat
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

// Helper function to convert cv::Mat to numpy array
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

PYBIND11_MODULE(lane_processor_py, m) {
    m.doc() = "Lane detector module with preprocessing, inference support and postprocessing";
    
    // Add module-level constants
    m.attr("WIDTH") = WIDTH;
    m.attr("HEIGHT") = HEIGHT;
    
    // Bind LaneDetector class
    py::class_<LaneDetector>(m, "LaneProcessor")
        .def(py::init<>())
        .def("preProcess", [](LaneDetector& self, py::array_t<uint8_t>& frame_array) {
            cv::Mat frame = numpy_to_mat(frame_array);
            cv::Mat result = self.preProcess(frame);  // Capture the return value
            return mat_to_numpy(result);  // Convert back to numpy and return
        })
        .def("postProcess", [](LaneDetector& self, py::array_t<uint8_t>& frame_array) {
            cv::Mat frame = numpy_to_mat(frame_array);
            self.postProcess(frame);
            return mat_to_numpy(frame);
        })
        .def("getInputData", [](LaneDetector& self) {
            // Return the preprocessed input data as a numpy array
            py::array_t<float> array({3, HEIGHT, WIDTH});
            auto buffer = array.request();
            float* ptr = static_cast<float*>(buffer.ptr);
            std::memcpy(ptr, self.getInputData(), 3 * HEIGHT * WIDTH * sizeof(float));
            return array;
        })
        .def("setOutputData", [](LaneDetector& self, py::array_t<float>& output_array) {
            // Get buffer info
            py::buffer_info info = output_array.request();
            
            // for (size_t i = 0; i < info.ndim; ++i) {
            //     std::cout << info.shape[i] << (i < info.ndim-1 ? ", " : "");
            // }
            // std::cout << "]" << std::endl;
            
            // Get pointer to data
            float* ptr = static_cast<float*>(info.ptr);
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();
            for (size_t i = 0; i < 2 * HEIGHT * WIDTH; i++) {
                min_val = std::min(min_val, ptr[i]);
                max_val = std::max(max_val, ptr[i]);
            }
            // Calculate expected size for 2 channels
            size_t expected_size = 2 * HEIGHT * WIDTH * sizeof(float);
            
            // Calculate actual size from the array info
            size_t total_elements = 1;
            for (size_t i = 0; i < info.ndim; ++i) {
                total_elements *= info.shape[i];
            }
            size_t actual_size = total_elements * sizeof(float);
            
            // std::cout << "Expected size: " << expected_size << " bytes, actual: " << actual_size << " bytes" << std::endl;
            
            // Handle different input sizes
            if (actual_size < expected_size) {
                // std::cout << "Creating new buffer with the correct size" << std::endl;
                // Create a new buffer with the right size
                std::vector<float> full_data(2 * HEIGHT * WIDTH, 0.0f);
                
                // Copy what data we have
                std::memcpy(full_data.data(), ptr, actual_size);
                
                // If we only have one channel's worth of data, duplicate it for the second channel
                if (actual_size == HEIGHT * WIDTH * sizeof(float)) {
                    // std::cout << "Duplicating single channel to create two channels" << std::endl;
                    std::memcpy(full_data.data() + HEIGHT * WIDTH, ptr, HEIGHT * WIDTH * sizeof(float));
                }
                
                // Use our new buffer
                self.setOutputData(full_data.data(), expected_size);
            } 
            else {
                // Safe to call with the expected size - we have enough data
                self.setOutputData(ptr, expected_size);

            }
        })
        // Remove the def_readonly_static lines that were causing errors
        .def_property_readonly("left_coeffs", [](const LaneDetector& self) {
            std::vector<double> coeffs;
            const cv::Mat& leftCoeffs = self.getLeftCoeffs();
            if (!leftCoeffs.empty() && leftCoeffs.rows >= 3) {
                coeffs = {
                    leftCoeffs.at<double>(0),
                    leftCoeffs.at<double>(1),
                    leftCoeffs.at<double>(2)
                };
            }
            return coeffs;
        })
        .def_property_readonly("right_coeffs", [](const LaneDetector& self) {
            std::vector<double> coeffs;
            const cv::Mat& rightCoeffs = self.getRightCoeffs();
            if (!rightCoeffs.empty() && rightCoeffs.rows >= 3) {
                coeffs = {
                    rightCoeffs.at<double>(0),
                    rightCoeffs.at<double>(1),
                    rightCoeffs.at<double>(2)
                };
            }
            return coeffs;
        })

        .def_property_readonly("left_points", [](const LaneDetector& self) {
            std::vector<py::tuple> points;
            const std::vector<cv::Point>& leftPoints = self.getLeftPoints();
            for (const auto& pt : leftPoints) {
                points.push_back(point_to_tuple(pt));
            }
            return points;
        })
        .def_property_readonly("right_points", [](const LaneDetector& self) {
            std::vector<py::tuple> points;
            const std::vector<cv::Point>& rightPoints = self.getRightPoints();
            for (const auto& pt : rightPoints) {
                points.push_back(point_to_tuple(pt));
            }
            return points;
        })
        .def_property_readonly("all_lane_points", [](const LaneDetector& self) {
            std::vector<py::tuple> points;
            const std::vector<cv::Point>& Points = self.getAllLanePoints();
            for (const auto& pt : Points) {
                points.push_back(point_to_tuple(pt));
            }
            return points;
        })
        .def_property_readonly("lane_points_visualization", [](const LaneDetector& self) {
            cv::Mat vis = self.getLanePointsVisualization();
            if (vis.empty()) {
                throw std::runtime_error("Lane points visualization not available");
            }
            
            auto rows = vis.rows;
            auto cols = vis.cols;
            auto channels = vis.channels();
            
            py::array_t<uint8_t> result({rows, cols, channels});
            auto buf = result.request();
            uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
            
            std::memcpy(ptr, vis.data, rows * cols * channels * sizeof(uint8_t));
            return result;
        });
}