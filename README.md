# ADAS-SIL-validation - Software in the Loop Testing Environment using Carla

A Software-in-the-Loop (SIL) testing environment for Advanced Driver Assistance Systems using the CARLA simulator.

## Overview

This project provides a comprehensive testing framework for validating lane detection, road segmentation, and obstacle detection algorithms in a simulated environment. It includes:

- Integration with CARLA simulator for realistic driving scenarios
- Advanced lane detection using Inverse Perspective Mapping (IPM)
- Computer vision processing pipeline with C++ components
- Lane tracking with Kalman filtering
- Visualization tools for debugging
- Real-time control algorithms


## Project Structure

- `adas_sil/` - Main package directory
  - `control/` - Vehicle control and simulation interfaces
  - `cpp_postprocessing/` - C++ accelerated processing modules
  - `models/` - ML models for perception tasks
- `carla_recordings/` - Recording storage directory
- `models/` - Pre-trained model storage
- `scripts/` - Utility scripts for dataset processing

## Features

- **Lane Detection**: Robust lane detection using computer vision techniques and IPM
- **Road Segmentation**: Semantic segmentation for road boundaries detection
- **Control Systems**: PID controllers for vehicle control
- **Video Recording**: Capability to record simulation sessions
- **Cross-language Integration**: Efficient C++ processing with Python bindings

## To run:

```sh
    # Navigate to your project directory (where setup.py is)
    cd ADAS-SIL-validation

    # Install package in development mode
    pip install -e .

    # Run using the entry point
    run-simulation
```
