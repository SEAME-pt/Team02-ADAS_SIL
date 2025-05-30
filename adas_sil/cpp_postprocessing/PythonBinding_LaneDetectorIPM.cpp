#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "LaneDetectorIPM_new.hpp"

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

PYBIND11_MODULE(lane_detector_py, m) {
    m.doc() = "Lane detector module with IPM support";
    
    // Add module-level constants
    m.attr("WIDTH") = WIDTH;
    m.attr("HEIGHT") = HEIGHT;
    
    // Bind LaneDetector class
    py::class_<LaneDetector>(m, "LaneProcessor")
        .def(py::init<>())
        .def("preProcess", [](LaneDetector& self, py::array_t<uint8_t>& frame_array) {
            cv::Mat frame = numpy_to_mat(frame_array);
            cv::Mat result = self.preProcess(frame);
            return mat_to_numpy(result);
        })
        .def("postProcess", [](LaneDetector& self, py::array_t<uint8_t>& frame_array) {
            cv::Mat frame = numpy_to_mat(frame_array);
            self.postProcess(frame);
            return mat_to_numpy(frame);
        })
        .def("visualizeBothViews", [](LaneDetector& self, py::array_t<uint8_t>& frame_array) {
            cv::Mat frame = numpy_to_mat(frame_array);
            self.visualizeBothViews(frame);
            return mat_to_numpy(frame);
        })
        // .def("detect", [](LaneDetector& self, py::array_t<uint8_t>& frame_array) {
        //     cv::Mat frame = numpy_to_mat(frame_array);
        //     self.detect(frame); // Make sure this method exists or add it
        //     return mat_to_numpy(frame);
        // })
        .def("setOutputData", [](LaneDetector& self, py::array_t<float>& output_array) {
            // Get buffer info
            py::buffer_info info = output_array.request();
            
            // Expected values
            size_t expected_elements = HEIGHT * WIDTH;
            size_t expected_size = expected_elements * sizeof(float);
            
            // Debug output
            std::cout << "Python array info: dimensions=" << info.ndim 
                    << ", shape=[";
            for (size_t i = 0; i < info.ndim; i++) {
                std::cout << info.shape[i] << (i < info.ndim - 1 ? ", " : "");
            }
            std::cout << "], size=" << info.size 
                    << ", format=" << info.format << std::endl;
            
            // Validation checks
            if (info.ndim != 1) {
                throw std::runtime_error("Expected 1D array (flattened), got " + 
                                        std::to_string(info.ndim) + "D");
            }
            
            if (info.size != expected_elements) {
                throw std::runtime_error("Array size mismatch: got " + 
                                        std::to_string(info.size) + 
                                        " elements, expected " + 
                                        std::to_string(expected_elements));
            }
            
            // Get pointer and sample data
            float* ptr = static_cast<float*>(info.ptr);
            
            // Print sample of data (first 5 elements)
            std::cout << "Data sample: [";
            for (int i = 0; i < std::min(5, static_cast<int>(info.size)); i++) {
                std::cout << ptr[i] << ", ";
            }
            std::cout << "...]" << std::endl;
            
            // Call the C++ function and track success
            try {
                self.setOutputData(ptr, expected_size);
                std::cout << "setOutputData completed successfully" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "C++ exception in setOutputData: " << e.what() << std::endl;
                throw;
            }
        })
        // Fix for left_coeffs
        .def_property_readonly("left_coeffs", [](const LaneDetector& self) -> py::object {
            try {
                const cv::Mat& coeffs = self.getLeftCoeffs();
                
                // Proper check for empty or invalid matrix
                if (coeffs.empty() || !coeffs.data) {
                    return py::none();
                }
                
                py::list coeff_list;
                
                // Handle column vector (common for polynomial coefficients)
                if (coeffs.cols == 1) {
                    for (int i = 0; i < coeffs.rows; i++) {
                        coeff_list.append(coeffs.at<double>(i, 0));
                    }
                }
                // Handle row vector
                else if (coeffs.rows == 1) {
                    for (int i = 0; i < coeffs.cols; i++) {
                        coeff_list.append(coeffs.at<double>(0, i));
                    }
                }
                // Handle unexpected matrix shape
                else {
                    // Just get the first row as a fallback
                    for (int i = 0; i < coeffs.cols; i++) {
                        coeff_list.append(coeffs.at<double>(0, i));
                    }
                }
                std::cout << "Left coefficients (binding): " << coeff_list << std::endl;
                return coeff_list;
            }
            catch (const cv::Exception& e) {
                std::cerr << "OpenCV error in left_coeffs binding: " << e.what() << std::endl;
                return py::none();
            }
            catch (const std::exception& e) {
                std::cerr << "Error in left_coeffs binding: " << e.what() << std::endl;
                return py::none();
            }
        })

        // Fix for right_coeffs
        .def_property_readonly("right_coeffs", [](const LaneDetector& self) -> py::object {
            try {
                const cv::Mat& coeffs = self.getRightCoeffs();
                
                // Proper check for empty or invalid matrix
                if (coeffs.empty() || !coeffs.data) {
                    return py::none();
                }
                
                py::list coeff_list;
                
                // Handle column vector (common for polynomial coefficients)
                if (coeffs.cols == 1) {
                    for (int i = 0; i < coeffs.rows; i++) {
                        coeff_list.append(coeffs.at<double>(i, 0));
                    }
                }
                // Handle row vector
                else if (coeffs.rows == 1) {
                    for (int i = 0; i < coeffs.cols; i++) {
                        coeff_list.append(coeffs.at<double>(0, i));
                    }
                }
                // Handle unexpected matrix shape
                else {
                    // Just get the first row as a fallback
                    for (int i = 0; i < coeffs.cols; i++) {
                        coeff_list.append(coeffs.at<double>(0, i));
                    }
                }
                
                return coeff_list;
            }
            catch (const cv::Exception& e) {
                std::cerr << "OpenCV error in right_coeffs binding: " << e.what() << std::endl;
                return py::none();
            }
            catch (const std::exception& e) {
                std::cerr << "Error in right_coeffs binding: " << e.what() << std::endl;
                return py::none();
            }
        })
        // Fix for midCoeffs
        .def_property_readonly("midCoeffs", [](const LaneDetector& self) -> py::object {
            try {
                cv::Mat coeffs = self.getMidCoeffs();
                
                // Proper check for empty or invalid matrix
                if (coeffs.empty() || !coeffs.data) {
                    return py::none();
                }
                
                py::list coeff_list;
                
                // Handle column vector (common for polynomial coefficients)
                if (coeffs.cols == 1) {
                    for (int i = 0; i < coeffs.rows; i++) {
                        coeff_list.append(coeffs.at<double>(i, 0));
                    }
                }
                // Handle row vector
                else if (coeffs.rows == 1) {
                    for (int i = 0; i < coeffs.cols; i++) {
                        coeff_list.append(coeffs.at<double>(0, i));
                    }
                }
                // Handle unexpected matrix shape
                else {
                    // Just get the first row as a fallback
                    for (int i = 0; i < coeffs.cols; i++) {
                        coeff_list.append(coeffs.at<double>(0, i));
                    }
                }
                
                return coeff_list;
            }
            catch (const cv::Exception& e) {
                std::cerr << "OpenCV error in midCoeffs binding: " << e.what() << std::endl;
                return py::none();
            }
            catch (const std::exception& e) {
                std::cerr << "Error in midCoeffs binding: " << e.what() << std::endl;
                return py::none();
            }
        })
        .def_property_readonly("all_lane_points", [](const LaneDetector& self) {
            std::vector<py::tuple> points;
            const std::vector<cv::Point>& Points = self.getAllLanePoints();
            for (const auto& pt : Points) {
                points.push_back(point_to_tuple(pt));
            }
            return points;
        })
        .def_property_readonly("bev_image", [](const LaneDetector& self) {
            cv::Mat bev = self.getBevImage();
            std::cout << "PAssei aqui" << std::endl;
            if (bev.empty()) {
                throw std::runtime_error("BEV image not available");
            }
            return mat_to_numpy(bev);
        })
        .def_property_readonly("polylines_viz", [](const LaneDetector& self) {
            cv::Mat allPolylinesViz = self.getPolyLines();
            if (allPolylinesViz.empty()) {
                throw std::runtime_error("Lane polylines visualization not available yet. Call createLanesIPM first.");
            }
            return mat_to_numpy(allPolylinesViz);
        }, "Visualization of all lane polylines before merging")

        .def_property_readonly("lane_Error", [](const LaneDetector& self) {
            const float laneError = self.getLaneError();
            return laneError;
        }, "Lane error value");
}