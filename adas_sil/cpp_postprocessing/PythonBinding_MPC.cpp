#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "MPController.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mpc_controller_py, m) {
    m.doc() = "Model Predictive Controller for lane following in CARLA";
    
    // Bind the Control struct
    py::class_<ModelPredictiveController::Control>(m, "MPCControl")
        .def(py::init<>())
        .def_readwrite("steering", &ModelPredictiveController::Control::steering)
        .def_readwrite("throttle", &ModelPredictiveController::Control::throttle);
    
    // Bind the MPC Controller
    py::class_<ModelPredictiveController>(m, "MPController")
        .def(py::init<>())
        .def("init", &ModelPredictiveController::init,
             py::arg("horizon"), py::arg("wheelbase"), py::arg("time_step"),
             py::arg("Q"), py::arg("R"), py::arg("Qf"))
        .def("solve", [](ModelPredictiveController& self, const Eigen::Vector4d& x0, py::object traj_coeffs) {
            std::vector<double> coeffs;
            if (!traj_coeffs.is_none()) {
                coeffs = traj_coeffs.cast<std::vector<double>>();
            }
            return self.solve(x0, coeffs);
        });
        // .def("setTrajectoryCoefficients", &ModelPredictiveController::setTrajectoryCoefficients)
        // .def("setAutonomousDriveState", &ModelPredictiveController::setAutonomousDriveState)
        // .def("getAutonomousDriveState", &ModelPredictiveController::getAutonomousDriveState);
}