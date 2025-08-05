# Team02 - CARLA Simulation with Machine Learning Detection (ADAS-SIL)

A Software-in-the-Loop (SIL) testing environment for Advanced Driver Assistance Systems using the CARLA simulator with deep learning-based lane and object detection.

## Overview

This project provides a comprehensive testing framework for validating lane detection and object detection algorithms in the CARLA simulation environment. It includes:

- Integration with CARLA simulator for realistic driving scenarios
- ONNX-based machine learning models for lane and object detection
- Real-time visualization with Pygame
- Zenoh-based communication for distributed processing

## Project Structure

```
.
├── LaneDetection.py          # Lane detection using ONNX models
├── ObjectDetection.py        # Object detection and segmentation
├── run_simulation.py         # Main CARLA simulation with Zenoh publishing
├── test_detection.py         # Testing detection algorithms with CARLA
├── car_view.py              # Multi-view display client for camera feeds
├── test_pub.py              # Zenoh publisher test
├── test_sub.py              # Zenoh subscriber test
├── test_opencv.py           # OpenCV GUI testing utilities
└── requirements.txt         # Python dependencies
```

## Features

- **Lane Detection**: Deep learning-based lane detection using ONNX models with PyTorch backend
- **Object Detection**: Semantic segmentation for cars, traffic signs, pedestrians, and other objects
- **Real-time Processing**: GPU-accelerated inference with CUDA support
- **Multi-view Display**: Grid layout showing original camera, IPM view, and detection masks
- **Distributed Architecture**: Zenoh communication for publishing camera feeds and control data
- **CARLA Integration**: Full integration with CARLA simulator including vehicle control

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run CARLA Simulation with Detection

Test the detection algorithms directly in CARLA:

```bash
python test_detection.py
```

This will:
- Start a CARLA simulation in Town04
- Spawn a vehicle with autopilot
- Run lane detection or object detection (configurable)

### 2. Run Simulation with Zenoh Communication

Start the main simulation that publishes data over Zenoh:

```bash
python run_simulation.py
```

This creates:
- A CARLA simulation with 150 traffic vehicles
- Camera feed publishing via Zenoh
- Vehicle control subscriber for remote control
- Speed data publishing

### 3. View Camera Feeds

In a separate terminal, run the multi-view display client:

```bash
python car_view.py
```

This displays a grid with:
- Original camera feed
- IPM (Inverse Perspective Mapping) view
- Lane detection mask
- Object detection mask
- Traffic sign detection mask


## Configuration

### Model Paths
Update the model paths in the detection classes:
- Lane detection model: [`LaneDetection.load_model()`](LaneDetection.py)
- Object detection model: [`ObjectDetection.load_model()`](ObjectDetection.py)

### Zenoh Network Configuration
Modify IP addresses and ports in:
- [`run_simulation.py`](run_simulation.py) - Publisher configuration
- [`car_view.py`](car_view.py) - Subscriber configuration
- [`test_pub.py`](test_pub.py) and [`test_sub.py`](test_sub.py) - Test configurations

### CARLA Settings
**CARLA**: CARLA simulator installation required
Adjust simulation parameters in [`run_simulation.py`](run_simulation.py):
- Number of traffic vehicles
- Vehicle spawn location
- Camera resolution and position
- World settings (Town04 by default)

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended for real-time performance)
- **CPU**: Multi-core processor for CARLA simulation
- **Memory**: 8GB+ RAM recommended

## Architecture

The system uses a distributed architecture:

1. **CARLA Simulation** ([`run_simulation.py`](run_simulation.py)) - Generates camera data and handles vehicle control
2. **Detection Processing** ([`LaneDetection.py`](LaneDetection.py), [`ObjectDetection.py`](ObjectDetection.py)) - ML-based perception
3. **Visualization Client** ([`car_view.py`](car_view.py)) - Multi-view display system
4. **Zenoh Communication** - Real-time data distribution