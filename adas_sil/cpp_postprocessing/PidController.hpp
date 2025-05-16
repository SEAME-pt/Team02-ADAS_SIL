#pragma once

#include <memory>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
// #include <sys/time.h>

#ifdef TEST_MODE
  // Declare your custom functions
  extern "C" int custom_xbox_open(const char* path, int flags);
  extern "C" int custom_xbox_close(int fd);
  extern "C" int custom_xbox_ioctl(int fd, unsigned long request, int* arg);
  extern "C" ssize_t custom_xbox_read(int fd, void* buf, size_t count);
  extern "C" ssize_t custom_xbox_write(int fd, const void* buf, size_t count);
#endif

class PidController
{
private:

    // PID constants
    float kp_; // Proportional gain
    float ki_; // Integral gain
    float kd_; // Derivative gain
    
    // PID variables
    float prev_error_;
    float cameraError_;
    float integral_;
    double last_time_;

    float direction_;
    float speed_;
    
    // Control parameters
    float constant_speed_; // Constant speed for the car
    float max_steering_angle_; // Maximum steering angle

    float fixed_delta_time_;

    std::string autonomousDrive_;

    float lane_departure_threshold_ = 0.1f;


public:
    PidController();
    ~PidController();
    
    void init(float kp, float ki, float kd, float speed, float delta_time);
    
    float steeringPID(float error, double current_time);
    float speedAdjustment(float error);

    void updateControl(float lane_error, double current_time);

    void setAutonomousDriveState(std::string current_state);
    std::string getAutonomousDriveState() const;

    // void runCarla(); // Main control loop
    void updateCarlaControl(float error, double current_time);
    void setCameraError(float error) { cameraError_ = error; }

};
