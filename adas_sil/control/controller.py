import sys
import os
import numpy as np
import cv2
import keyboard
import time
import pygame


opencv_bin = "C:/Users/manue/opencv/build/x64/vc16/bin"
os.environ["PATH"] = opencv_bin + os.pathsep + os.environ["PATH"]

carla_egg = "C:/Users/manue/Documents/SEA_ME/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg"
sys.path.append(carla_egg)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import cpp_postprocessing.build.pid_controller_py as pid_controller_py
    # import cpp_postprocessing.build.mpc_controller_py as mpc_controller_py
    print("Successfully imported controller modules")
except ImportError as e:
    print(f"Failed to import controller modules: {e}")

import carla

from adas_sil.control.CameraManager import CameraManager
from adas_sil.control.vehicle_controller.pid_controller import PIDController
from adas_sil.control.vehicle_controller.mpc_controller import MPCController
from adas_sil.control.display import Display

class Controller:
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world

        # Control state
        self.autonomous_mode = False
        self.control_mode = "PID"  # "PID" or "MPC"
        self._steer_cache = 0.0



        self.camera_manager = CameraManager(vehicle, world)
        self.display_manager = Display(1280, 960)
        self.pid_controller = PIDController()

        # Initialize controllers
        try:
            # Initialize PID parameters: Kp, Ki, Kd, base_speed, dt
            self.pid_controller = pid_controller_py.PidController()
            self.pid_controller.init(15.0, 0.01, 5.0, 0.5, 0.02)
            print("PID controller initialized")

            # Initialize MPC controller
            # self.mpc_controller = mpc_controller_py.MPController()
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
            # self.mpc_controller.init(10, 2.9, 0.1, Q, R, Q*5.0)  # Qf is terminal cost (higher)
            # print("MPC controller initialized")

            self.autonomous_mode = False
            self._previous_t_state = False
        except Exception as e:
            print(f"Error initializing PID controller: {e}")

        try:
            # Initialize detector
            from adas_sil.perception.Detection import Detection

            # try:
            #     self.ipm = ipm_module.IPM()
            #     # Create a new pygame surface for bird's eye view
            #     self.bev_surface = None
            #     print("IPM module initialized for bird's eye view")
            # except Exception as e:
            #     print(f"Error initializing IPM module: {e}")
            
            self.detector = Detection()
            # Initialize pygame display for visualization
            self.display = pygame.display.set_mode((1280, 960))
            pygame.display.set_caption("CARLA Camera Feed")
            self.rgb_surface = None
            self.lane_surface = None
            
            # Create output directories
            self.output_dir = 'carla_recordings'
            os.makedirs(f'{self.output_dir}/rgb', exist_ok=True)
            os.makedirs(f'{self.output_dir}/lanes', exist_ok=True)
            
            self.record_video = False
            self.video_dir = os.path.join(self.output_dir, 'videos')
            os.makedirs(self.video_dir, exist_ok=True)
            self.video_filename = os.path.join(self.video_dir, f'drive_{time.strftime("%Y%m%d-%H%M%S")}.mp4')
            self.video_fps = 20.0
            self.video_writer = None

            # Set up cameras and start recording
            self.setup_cameras()
            # self.detector.load_model(self.rgb_detcam)
                    # Set up listeners
            self.rgb_cam.listen(self.process_rgb_image)
            self.sem_cam.listen(self.process_semantic_image)
            # self.rgb_detcam.listen(self.process_rgb_with_detection)
            print("Controller initialized - recording started")
        except Exception as e:
            print(f"Error initializing controller: {e}")
            pygame.quit()
            if self.rgb_cam:
                self.rgb_cam.destroy()
            if self.sem_cam:
                self.sem_cam.destroy()
            raise


    def toggle_recording(self):
        self.record_video = not self.record_video
        if self.record_video:
            # Start a new recording
            self.video_filename = os.path.join(self.video_dir, f'drive_{time.strftime("%Y%m%d-%H%M%S")}.mp4')
            self.video_writer = None  # Will be initialized on next frame
            print(f"Video recording started: {self.video_filename}")
        else:
            # Stop recording
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                print(f"Video recording stopped: {self.video_filename}")

    def toggle_controller(self):
        """Switch between PID and MPC controllers"""
        if self.control_mode == "PID":
            self.control_mode = "MPC"
        else:
            self.control_mode = "PID"
        print(f"Controller switched to: {self.control_mode}")

    def control_car(self):
        """
        Control the car manually using keyboard input.
        Uses WASD controls similar to CARLA's manual_control.py
        
        W: accelerate
        S: brake
        A: steer left
        D: steer right
        SPACE: handbrake
        """
        
        # Control parameters
        steer_increment = 0.05
        throttle_increment = 0.1
        brake_increment = 0.2
        throttle_decay = 0.05  # Rate at which throttle decreases when not pressing W
        
        # Create a vehicle control instance
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.0
        control.steer = 0.0

        
        
        # Stop autopilot to allow manual control
        self.vehicle.set_autopilot(False)
        print("Manual control activated. Use WASD to drive.")
        print("W: accelerate, S: brake, A/D: steer, SPACE: handbrake, Q: exit")
        
        try:
            running = True
            clock = pygame.time.Clock()
            
            while running:
                try:
                    milliseconds = clock.tick_busy_loop(30)  # 30 FPS
                
                    # Process events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                print("Control deactivated")
                                running = False
                            elif event.key == pygame.K_r:
                                self.toggle_recording()
                            elif event.key == pygame.K_t:
                                # Toggle autonomous mode
                                self.autonomous_mode = not self.autonomous_mode
                                print(f"Autonomous mode: {'ON' if self.autonomous_mode else 'OFF'}")
                                self.pid_controller.setAutonomousDriveState(
                                    "SAE_4" if self.autonomous_mode else "SAE_0")
                                if hasattr(self, 'mpc_controller'):
                                    self.mpc_controller.setAutonomousDriveState(
                                    "SAE_4" if self.autonomous_mode else "SAE_0")
                            elif event.key == pygame.K_c:
                                # Toggle between controllers
                                if self.autonomous_mode:
                                    self.toggle_controller()
                    
                    # Get pressed keys
                    keys = pygame.key.get_pressed()
                
                    # Handle control based on mode
                    if self.autonomous_mode:
                        try:
                            # Get lane error from lane detector if available
                            lane_error = 0.0
                            if hasattr(self.detector, 'lane_detector') and hasattr(self.detector.lane_detector, 'lane_Error'):
                                lane_error = self.detector.lane_detector.lane_Error
                                print(f"Using lane error: {lane_error}")
                            
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
                            
                            # Apply control
                            control.steer = carla_steer
                            control.throttle = throttle
                            control.brake = 0.0
                            
                            # Debug info
                            print(f"PID: Error={lane_error:.1f}, Steer={carla_steer:.2f}, Throttle={throttle:.2f}")
                        except Exception as pid_err:
                            print(f"Error in PID control: {pid_err}")
                            self.autonomous_mode = False  # Revert to manual if error
                    else:
                        # Manual control mode - keep your existing code
                        if keys[pygame.K_w]:
                            control.throttle = min(control.throttle + 0.01, 1.0)
                            control.brake = 0.0
                        else:
                            control.throttle = 0.0                  
                        # Handle braking - same as _parse_vehicle_keys
                        if keys[pygame.K_s]:
                            control.throttle = 0.0
                            control.brake = min(control.brake + 0.2, 1.0)
                        else:
                            control.brake = 0.0
                        
                        # Handle steering with time-based increment - same as _parse_vehicle_keys
                        steer_increment = 5e-4 * milliseconds
                        if keys[pygame.K_a]:
                            if self._steer_cache > 0:
                                self._steer_cache = 0
                            else:
                                self._steer_cache -= steer_increment
                        elif keys[pygame.K_d]:
                            if self._steer_cache < 0:
                                self._steer_cache = 0
                            else:
                                self._steer_cache += steer_increment
                        else:
                            self._steer_cache = 0.0  # Immediate return to center
                        
                        # Apply steering limits and rounding
                        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
                        control.steer = round(self._steer_cache, 1)
                        
                        
                        # Handle handbrake
                        control.hand_brake = keys[pygame.K_SPACE]

                        if self.vehicle.is_at_traffic_light():
                            traffic_light = self.vehicle.get_traffic_light()
                            if traffic_light.get_state() == carla.TrafficLightState.Red:
                                traffic_light.set_state(carla.TrafficLightState.Green)
                    
                        # Apply the control to the vehicle
                    self.vehicle.apply_control(control)
                
                    # Update the display to show both camera views
                    try:
                        self.update_display()
                    except Exception as display_err:
                        print(f"Display update error: {display_err}")
                except Exception as loop_err:
                    print(f"Error in control loop: {loop_err}")
                    running = False
        except KeyboardInterrupt:
            print("Manual control stopped by user")
        except Exception as e:
            print(f"Error in manual control: {e}")
        finally:
            self.cleanup()
            # Reactivate autopilot when done
            # self.vehicle.set_autopilot(True)

    


    def cleanup(self):

        if self.record_video and self.video_writer is not None:
            self.video_writer.release()
            print(f"Video saved to {self.video_filename}")
            
        print("Cleaning up resources...")
        
        # 1. First, stop all sensor callbacks
        if hasattr(self, 'rgb_cam') and self.rgb_cam is not None:
            self.rgb_cam.stop()
            
        if hasattr(self, 'sem_cam') and self.sem_cam is not None:
            self.sem_cam.stop()

        if hasattr(self, 'rgb_detcam') and self.rgb_detcam is not None:
            self.rgb_detcam.stop()
        
        # 2. Close video writer if active
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
        # 3. After callbacks are stopped, destroy sensors
        if hasattr(self, 'rgb_cam') and self.rgb_cam is not None:
            self.rgb_cam.destroy()
            self.rgb_cam = None
            
        if hasattr(self, 'sem_cam') and self.sem_cam is not None:
            self.sem_cam.destroy()
            self.sem_cam = None

        if hasattr(self, 'rgb_detcam') and self.rgb_detcam is not None:
            self.rgb_detcam.destroy()
            self.rgb_detcam = None
        
        # 4. Finally quit pygame properly
        print("Quitting pygame...")
        pygame.display.quit()
        pygame.quit()
        
        print("Cleanup complete")