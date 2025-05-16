#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "IPM.hpp"

namespace py = pybind11;

// Convert numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> array) {
    py::buffer_info buf = array.request();
    
    if (buf.ndim == 3) {
        return cv::Mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
    }
    else if (buf.ndim == 2) {
        return cv::Mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
    }
    
    throw std::runtime_error("Incompatible numpy array format");
}

// Convert cv::Mat to numpy array
py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat) {
    if (mat.empty()) {
        throw std::runtime_error("Empty cv::Mat");
    }
    
    py::array_t<uint8_t> result;
    
    if (mat.channels() == 3) {
        result = py::array_t<uint8_t>({mat.rows, mat.cols, 3}, {(size_t)mat.step[0], (size_t)mat.step[1], sizeof(uint8_t)});
    }
    else {
        result = py::array_t<uint8_t>({mat.rows, mat.cols}, {(size_t)mat.step[0], sizeof(uint8_t)});
    }
    
    auto buf = result.request();
    std::memcpy(buf.ptr, mat.data, mat.total() * mat.elemSize());
    
    return result;
}

// Convert cv::Mat to numpy array (for perspective matrices)
py::array_t<double> mat_to_numpy_double(const cv::Mat& mat) {
    if (mat.empty()) {
        throw std::runtime_error("Empty cv::Mat");
    }
    
    py::array_t<double> result({mat.rows, mat.cols});
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            ptr[i*mat.cols + j] = mat.at<double>(i, j);
        }
    }
    
    return result;
}

// Helper function to convert cv::Point to Python tuple
cv::Size tuple_to_size(const py::object& obj) {
    // Handle tuples
    if (py::isinstance<py::tuple>(obj)) {
        py::tuple t = py::cast<py::tuple>(obj);
        if (py::len(t) != 2) {
            throw std::runtime_error("Size tuple must have exactly 2 elements");
        }
        return cv::Size(py::cast<int>(t[0]), py::cast<int>(t[1]));
    }
    // Handle lists
    else if (py::isinstance<py::list>(obj)) {
        py::list lst = py::cast<py::list>(obj);
        if (py::len(lst) != 2) {
            throw std::runtime_error("Size list must have exactly 2 elements");
        }
        return cv::Size(py::cast<int>(lst[0]), py::cast<int>(lst[1]));
    }
    // Handle already-converted cv::Size objects
    else if (py::isinstance<cv::Size>(obj)) {
        return py::cast<cv::Size>(obj);
    }
    
    throw std::runtime_error("Cannot convert object to cv::Size. Must be tuple or list with 2 elements.");
}

py::list points_to_list(const std::vector<cv::Point2f>& points) {
    py::list result;
    for (const auto& p : points) {
        result.append(py::make_tuple(p.x, p.y));
    }
    return result;
}

PYBIND11_MODULE(ipm_module, m) {
    m.doc() = "IPM module for Bird's Eye View conversion";
    
    py::class_<IPM>(m, "IPM")
        .def(py::init<>())
        // Update constructor binding
        .def(py::init([](py::object orig_size, py::object dst_size,
                    const std::vector<cv::Point2f>& orig_points,
                    const std::vector<cv::Point2f>& dst_points) {
            return new IPM(tuple_to_size(orig_size), tuple_to_size(dst_size),
                        orig_points, dst_points);
        }), py::arg("orig_size"), py::arg("dst_size"),
            py::arg("orig_points") = std::vector<cv::Point2f>(),
            py::arg("dst_points") = std::vector<cv::Point2f>())
        
        // Update initialize method binding
        .def("initialize", [](IPM& self, py::object orig_size, py::object dst_size,
                        const std::vector<cv::Point2f>& orig_points,
                        const std::vector<cv::Point2f>& dst_points) {
            self.initialize(tuple_to_size(orig_size), tuple_to_size(dst_size),
                        orig_points, dst_points);
        }, py::arg("orig_size"), py::arg("dst_size"),
        py::arg("orig_points") = std::vector<cv::Point2f>(),
        py::arg("dst_points") = std::vector<cv::Point2f>())
        
        
        // Apply IPM transformation
        .def("apply_ipm", [](IPM& self, py::array_t<uint8_t> image) {
            cv::Mat input = numpy_to_mat(image);
            cv::Mat result = self.applyIPM(input);
            return mat_to_numpy(result);
        }, py::arg("image"))
        
        // Apply inverse IPM transformation
        .def("apply_inverse_ipm", [](IPM& self, py::array_t<uint8_t> image) {
            cv::Mat input = numpy_to_mat(image);
            cv::Mat result = self.applyInverseIPM(input);
            return mat_to_numpy(result);
        }, py::arg("image"))
        
        // Camera calibration method
        .def("calibrate_from_camera", &IPM::calibrateFromCamera,
             py::arg("camera_height"), py::arg("camera_pitch"), 
             py::arg("horizontal_fov"), py::arg("vertical_fov"),
             py::arg("near_distance"), py::arg("far_distance"), 
             py::arg("lane_width"))
        
        // Calculate IPM points based on camera parameters
        .def("calculate_ipm_points", [](IPM& self, float camera_height, float camera_pitch,
            float horizontal_fov, float vertical_fov,
            py::object image_size, float near_distance,
            float far_distance, float lane_width) {
            return self.calculateIPMPoints(camera_height, camera_pitch, horizontal_fov,
                        vertical_fov, tuple_to_size(image_size),
                        near_distance, far_distance, lane_width);
            }, py::arg("camera_height"), py::arg("camera_pitch"),
            py::arg("horizontal_fov"), py::arg("vertical_fov"),
            py::arg("image_size"), py::arg("near_distance"),
            py::arg("far_distance"), py::arg("lane_width"))
        
        // Visualization helper
        .def_static("draw_points", [](py::array_t<uint8_t> image, std::vector<cv::Point2f> points, 
                                     py::tuple color = py::make_tuple(0, 0, 255)) {
            cv::Mat img = numpy_to_mat(image);
            cv::Scalar cv_color;
            
            if (py::len(color) >= 3) {
                cv_color = cv::Scalar(
                    py::cast<int>(color[0]),
                    py::cast<int>(color[1]),
                    py::cast<int>(color[2])
                );
            } else {
                cv_color = cv::Scalar(0, 0, 255); // Default to red
            }
            
            IPM::drawPoints(img, points, cv_color);
            return mat_to_numpy(img);
        }, py::arg("image"), py::arg("points"), py::arg("color") = py::make_tuple(0, 0, 255))

        // Properties (read-only)
        .def_property_readonly("perspective_matrix", [](const IPM& self) {
            return mat_to_numpy_double(self.getPerspectiveMatrix());
        })
        .def_property_readonly("inverse_perspective_matrix", [](const IPM& self) {
            return mat_to_numpy_double(self.getInvPerspectiveMatrix());
        })
        .def_property_readonly("orig_points", [](const IPM& self) {
            return points_to_list(self.getOrigPoints());
        }, "Original source points for the perspective transform")
        .def_property_readonly("dst_points", [](const IPM& self) {
                return points_to_list(self.getDstPoints());
        });
}