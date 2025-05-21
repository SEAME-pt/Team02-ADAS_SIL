import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    import cpp_postprocessing.build.pid_controller_py as pid_controller_py
    print("Successfully imported controller modules")
except ImportError as e:
    print(f"Failed to import controller modules: {e}")


class PIDController:
    def __init__(self, kp, ki, kd, throttle, dt):
        self.pid_controller = pid_controller_py.PidController()
        self.pid_controller.init(kp, ki, kd, throttle, dt)
        print("PID controller initialized")

    def update(self, lane_error):
        # Get current time for PID computation
        current_time = time.time()
        
        # Set the camera error in the PID controller
        self.pid_controller.setCameraError(lane_error)
        
        # Calculate steering angle using PID
        steer_angle = self.pid_controller.steeringPID(lane_error, current_time)
        
        # Convert from degrees to CARLA steering (-1 to 1)
        carla_steer = (steer_angle - 90.0) / 90.0
        carla_steer = max(-1.0, min(1.0, carla_steer))
        
        # Calculate speed based on error
        speed_percentage = self.pid_controller.speedAdjustment(lane_error)
        throttle = min(speed_percentage / 100.0, 0.5)  # Limit to 70% throttle
        
        # Debug info
        print(f"PID: Error={lane_error:.1f}, Steer={carla_steer:.2f}, Throttle={throttle:.2f}")

        return carla_steer, throttle