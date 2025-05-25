import sys
import os
import numpy as np
import math

opencv_bin = "C:/Users/manue/opencv/build/x64/vc16/bin"
os.environ["PATH"] = opencv_bin + os.pathsep + os.environ["PATH"]

carla_egg = "C:/Users/manue/Documents/SEA_ME/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg"
sys.path.append(carla_egg)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    import cpp_postprocessing.build.mpc_controller_py as mpc_controller_py
    # print("Successfully imported MPC controller module")
except ImportError as e:
    print(f"Failed to import MPC controller module: {e}")

import carla

class MPCController:
    def __init__(self, vehicle = None, horizon=10, dt=0.1, target_speed_kmh=30.0):
        """
        Initialize the MPC controller wrapper
        
        Args:
            vehicle_model: Vehicle dynamics model (optional)
            horizon: Prediction horizon in steps
            dt: Time step in seconds
        """
        self.vehicle = vehicle
        self.horizon = horizon
        self.dt = dt

        self.target_speed_ms = target_speed_kmh / 3.6
        
        # Initialize MPC controller
        try:
            self.mpc_controller = mpc_controller_py.MPController()
            
            # Q matrix (state costs)
            Q = np.eye(4)
            Q[0,0] = 100.0  # x position cost 
            Q[1,1] = 100.0  # y position cost
            Q[2,2] = 10.0   # heading cost
            Q[3,3] = 1.0    # velocity cost
            
            # R matrix (control input costs)
            R = np.eye(2)
            R[0,0] = 0.1    # throttle cost
            R[1,1] = 10.0   # steering cost
            
            # Initialize parameters: horizon, wheelbase, timestep, Q, R, Qf
            self.mpc_controller.init(horizon, 2.9, dt, Q, R, Q*5.0)  # Qf is terminal cost (higher)

            self.mpc_controller.setTargetVelocity(self.target_speed_ms)

            # print("MPC controller initialized")
        except Exception as e:
            print(f"Error initializing MPC controller: {e}")
            self.mpc_controller = None

    # Add method to change target speed
    def set_target_speed(self, speed_kmh):
        """Set target speed for the MPC controller
        
        Args:
            speed_kmh: Target speed in km/h
        """
        if self.mpc_controller:
            self.target_speed_ms = speed_kmh / 3.6
            self.mpc_controller.setTargetVelocity(self.target_speed_ms)
            print(f"Target speed set to {speed_kmh} km/h")

    def set_vehicle_state(self):
        """
        Extract current vehicle state from CARLA and set it in the C++ controller
        
        Returns:
            The vehicle state vector [x, y, heading, velocity] that was set
        """
        if self.vehicle is None or self.mpc_controller is None:
            return None
            
        try:
            # Get vehicle transform and velocity
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            
            # Calculate speed in m/s
            speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
            
            # In the vehicle's reference frame, position is at origin
            x = 0.0
            y = 0.0
            
            # Convert yaw from degrees to radians
            heading = transform.rotation.yaw * math.pi / 180.0
            
            # Create state vector
            state = np.array([x, y, heading, speed], dtype=np.float64)
            
            # Set state in C++ controller
            self.mpc_controller.setVehicleState(state)
            
            return state
            
        except Exception as e:
            print(f"Error setting vehicle state: {e}")
            return None

    def get_vehicle_state(self):
        """
        Get the current vehicle state from the C++ controller
        
        Returns:
            Current vehicle state vector [x, y, heading, velocity]
        """
        if self.mpc_controller is None:
            return None
            
        try:
            return self.mpc_controller.getVehicleState()
        except Exception as e:
            print(f"Error getting vehicle state: {e}")
            return None

    def compute_control(self, lane_coeffs=None):
        """
        Compute control signals using MPC with internal vehicle state
        
        Args:
            lane_coeffs: Lane polynomial coefficients [a, b, c, d] for x = a*y³ + b*y² + c*y + d
            
        Returns:
            Tuple of (steering, throttle) control values
        """
        if self.mpc_controller is None:
            print("MPC controller not initialized")
            return 0.0, 0.0
            
        try:
            # Update vehicle state in controller
            self.set_vehicle_state()
            
            # Get vehicle state from controller
            state = self.get_vehicle_state()
            print(f"Vehicle state: {state}")
            if state is None:
                print("No vehicle state available")
                return 0.0, 0.0
            
            # Ensure coefficients are in the right format
            if lane_coeffs is None or len(lane_coeffs) < 4:
                # Default to straight path if no coefficients
                lane_coeffs = [0.0, 0.0, 0.0, 0.0]
                print("No lane coefficients provided, using default straight path")
                
            # Call the C++ solver
            control = self.mpc_controller.solve(state, lane_coeffs)
            
            # Extract control values
            steering = float(control.steering)
            throttle = float(control.throttle)
            print(f"Control values: steering={steering}, throttle={throttle}")
            # Apply limits
            steering = max(-0.7, min(0.7, steering))  # Limit to [-0.7, 0.7]
            throttle = max(0.0, min(1.0, throttle))   # Limit to [0, 1]
            
            return steering, throttle
            
        except Exception as e:
            print(f"Error in MPC control computation: {e}")
            return 0.0, 0.0



         