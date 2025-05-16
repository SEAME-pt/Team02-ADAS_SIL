#include "PidController.hpp"

#ifdef TEST_MODE
// Define custom function names for testing
#define device_open custom_xbox_open
#define device_close custom_xbox_close
#define device_ioctl custom_xbox_ioctl
#define device_read custom_xbox_read
#define device_write custom_xbox_write
#define SESSION_OPEN zenoh::Session::open
#define ZENOH_CONFIG_FROM_FILE zenoh::Config::create_default()
#else
#define device_open open
#define device_close close
#define device_ioctl ioctl
#define device_read read
#define device_write write
#define SESSION_OPEN zenoh::Session::open
#define ZENOH_CONFIG_FROM_FILE zenoh::Config::from_file(configFile)
#endif

// double getCurrentTime()
// {
//     struct timeval tv;
//     gettimeofday(&tv, NULL);
//     return tv.tv_sec + tv.tv_usec * 1e-6;
// }

PidController::PidController()
{
    prev_error_  = 0.0f;
    cameraError_ = 0.0f;
    integral_    = 0.0f;
    last_time_   = 0.0f;

    constant_speed_     = 20.0f;
    max_steering_angle_ = 90.0f;

    kp_ = 1.0f;
    ki_ = 0.0f;
    kd_ = 0.0f;

    fixed_delta_time_ = 0.02f;
    autonomousDrive_  = "SAE_0";

    std::cout << "PID controller created!" << std::endl;
}

PidController::~PidController() {}

void PidController::init(float kp, float ki, float kd, float speed,
                         float delta_time)
{
    kp_               = kp;
    ki_               = ki;
    kd_               = kd;
    constant_speed_   = speed;
    fixed_delta_time_ = delta_time;

    prev_error_ = 0.0f;
    integral_   = 0.0f;

    std::cout << "PID Controller initialized with Kp=" << kp_ << ", Ki=" << ki_
              << ", Kd=" << kd_ << ", speed=" << constant_speed_
              << ", dt=" << fixed_delta_time_ << std::endl;

    autonomousDrive_ = "SAE_0";
}

float PidController::steeringPID(float error, double current_time)
{
    // dt
    double dt = current_time - last_time_;
    std::cout << "dt: " << dt << std::endl;

    // PID
    float p_term = kp_ * error;

    // Improved implementation with anti-windup
    integral_ += error * dt;
    // Limit integral term to prevent windup
    const float MAX_INTEGRAL = 10.0f; // Adjust based on your system
    integral_    = std::max(-MAX_INTEGRAL, std::min(integral_, MAX_INTEGRAL));
    float i_term = ki_ * integral_;

    float d_term = kd_ * (error - prev_error_) / dt;

    // Adjust steering
    float steering_correction = p_term + i_term + d_term;

    float direction = 90 + steering_correction;
    if (direction > 90.0f + max_steering_angle_)
    {
        direction = 90.0f + max_steering_angle_;
    }
    else if (direction < 90.0f - max_steering_angle_)
    {
        direction = 90.0f - max_steering_angle_;
    }

    prev_error_ = error;
    last_time_  = current_time;
    return direction;
}

float PidController::speedAdjustment(float error)
{
    static bool is_starting               = true;
    static auto start_time                = std::chrono::steady_clock::now();
    static const float BOOST_DURATION_SEC = 0.5f; // Duration of initial boost
    static const float BOOST_MULTIPLIER   = 1.2f; // 20% boost to starting speed

    // Check if we're in the initial starting phase
    bool apply_boost = false;
    if (is_starting)
    {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration<float>(current_time - start_time).count();

        if (elapsed < BOOST_DURATION_SEC)
        {
            apply_boost = true;
            // Log that we're applying boost
            std::cout << "Applying initial speed boost: " << elapsed
                      << " seconds" << std::endl;
        }
        else
        {
            // Boost period has ended
            is_starting = false;
            std::cout << "Initial boost complete" << std::endl;
        }
    }

    // Dynamic speed adjustment based on error
    float error_magnitude = std::fabs(error);
    std::cout << "Error magnitude: " << error_magnitude << std::endl;

    // Define speed control parameters
    const float BASE_SPEED =
        constant_speed_; // Maximum speed when error is minimal
    const float MIN_SPEED = BASE_SPEED * 0.6f; // Minimum speed (60% of max)
    const float ERROR_THRESHOLD =
        0.08f; // Error threshold where speed starts decreasing

    // Calculate dynamic speed
    float dynamic_speed;
    if (error_magnitude < ERROR_THRESHOLD)
    {
        // Linear interpolation between BASE_SPEED and slightly reduced speed
        float reduction_factor = error_magnitude / ERROR_THRESHOLD;
        dynamic_speed = BASE_SPEED - (reduction_factor * (BASE_SPEED * 0.2f));
    }
    else
    {
        // For larger errors, reduce speed more aggressively
        float excess_error = error_magnitude - ERROR_THRESHOLD;
        float reduction_factor =
            std::min(1.0f, excess_error / (ERROR_THRESHOLD * 2));
        dynamic_speed =
            BASE_SPEED * 0.8f - (reduction_factor * (BASE_SPEED * 0.2f));
    }

    // Ensure speed doesn't go below minimum
    dynamic_speed = std::max(MIN_SPEED, dynamic_speed);

    // Apply boost if in starting phase
    if (apply_boost)
    {
        dynamic_speed *= BOOST_MULTIPLIER;
    }

    return dynamic_speed * 100;
}


void PidController::setAutonomousDriveState(std::string current_state)
{
    autonomousDrive_ = current_state;
}

std::string PidController::getAutonomousDriveState() const
{
    return autonomousDrive_;
}

void PidController::updateCarlaControl(float error, double current_time)
{
    direction_ = steeringPID(
        error, current_time); // float dynamicSpeed = speedAdjustment(error);

    std::cout << "Direction: " << direction_ //<< ", Speed: " << dynamicSpeed
              << std::endl;
    // publisher_->publishCurrentGear(1);
}