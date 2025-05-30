cmake_minimum_required(VERSION 3.10)
project(LaneProcessing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
set(OpenCV_DIR "C:/Users/manue/opencv/build")
find_package(OpenCV REQUIRED)

# pybind11 (from pip installation)
set(pybind11_DIR "C:/Users/manue/Documents/SEA_ME/LaneDetection/myenv/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Eigen directory
set(EIGEN3_INCLUDE_DIR "C:/eigen-3.4.0/build") 
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(lane_processor STATIC
    LaneDetectorIPM_new.cpp
    IPM.cpp
    KalmanFilter.cpp
)

target_link_libraries(lane_processor
    ${OpenCV_LIBS}
)

# Python binding for LaneDetectorIPM
pybind11_add_module(lane_detector_py SHARED
    PythonBinding_LaneDetectorIPM.cpp
)

target_link_libraries(lane_detector_py PRIVATE
    lane_processor
    ${OpenCV_LIBS}
)

add_library(pid_controller STATIC
    PidController.cpp
)

target_link_libraries(pid_controller
    ${OpenCV_LIBS}
)

pybind11_add_module(pid_controller_py
    PythonBinding_PidController.cpp
)

target_link_libraries(pid_controller_py PRIVATE
    pid_controller
    ${OpenCV_LIBS}
) 


# Find Eigen3 (required for MPC)
find_package(Eigen3 REQUIRED)

# MPC Controller static library
add_library(mpc_controller STATIC
    MPController.cpp
)

target_link_libraries(mpc_controller
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Python binding for MPC Controller
pybind11_add_module(mpc_controller_py
    PythonBinding_MPC.cpp
)

target_link_libraries(mpc_controller_py PRIVATE
    mpc_controller
    ${OpenCV_LIBS}
    Eigen3::Eigen
)


message(STATUS "OpenCV_BIN_DIR: ${OpenCV_BIN_DIR}")

# Try to find the OpenCV bin directory
if(NOT OpenCV_BIN_DIR)
    if(EXISTS "${OpenCV_DIR}/x64/vc16/bin")
        set(OpenCV_BIN_DIR "${OpenCV_DIR}/x64/vc16/bin")
    elseif(EXISTS "${OpenCV_DIR}/bin")
        set(OpenCV_BIN_DIR "${OpenCV_DIR}/bin")
    endif()
endif()

# Copy OpenCV DLLs to output directory
if(OpenCV_BIN_DIR)
    message(STATUS "Will copy OpenCV DLLs from: ${OpenCV_BIN_DIR}")
    file(GLOB OpenCV_DLLS "${OpenCV_BIN_DIR}/opencv_*.dll")
    foreach(OpenCV_DLL ${OpenCV_DLLS})
        add_custom_command(TARGET lane_detector_py POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OpenCV_DLL}"
                "$<TARGET_FILE_DIR:lane_detector_py>")
    endforeach()
else()
    message(WARNING "Could not find OpenCV bin directory. You may need to manually copy DLLs.")
endif()


# Needed for importing modules from the build directory
set(INIT_PY "${CMAKE_CURRENT_BINARY_DIR}/__init__.py")
if(NOT EXISTS ${INIT_PY})
    file(WRITE ${INIT_PY} "")
endif()

function(setup_python_module MODULE_NAME)
    add_custom_command(TARGET ${MODULE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE_DIR:${MODULE_NAME}>/$<TARGET_FILE_NAME:${MODULE_NAME}>
                ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:${MODULE_NAME}>
    )
endfunction()

setup_python_module(lane_detector_py)
setup_python_module(pid_controller_py)
setup_python_module(mpc_controller_py)



