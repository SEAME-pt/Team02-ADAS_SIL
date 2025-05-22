from importlib.util import MAGIC_NUMBER
import sys
import os
import time
import pygame

from adas_sil.control.CameraManager import CameraManager
from adas_sil.control.vehicle_controller.pid_controller import PIDController
from adas_sil.control.vehicle_controller.mpc_controller import MPCController
from adas_sil.control.display import Display
from adas_sil.perception.Detection import Detection

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

class Controller:
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world

        # Control state
        self.autonomous_mode = False
        self.control_mode = "MPC"  # "PID" or "MPC"
        self._steer_cache = 0.0
        self.autonomous_mode = False
        self._previous_t_state = False

        try:
            self.display_manager = Display(1280, 960)
            self.detector = Detection()
            self.camera_manager = CameraManager(vehicle, world, self.display_manager, self.detector)

            self.pid_controller = PIDController(15.0, 0.01, 5.0, 0.5, 0.02)
            self.mpc_controller = MPCController(vehicle, 10, 0.1, 5)  # Initialize MPC controller

        except Exception as e:
            print(f"Error initializing PID controller: {e}")

        # Video recording
            # # Create output directories
            # self.output_dir = 'carla_recordings'
            # os.makedirs(f'{self.output_dir}/rgb', exist_ok=True)
            # os.makedirs(f'{self.output_dir}/lanes', exist_ok=True)
            
            # self.record_video = False
            # self.video_dir = os.path.join(self.output_dir, 'videos')
            # os.makedirs(self.video_dir, exist_ok=True)
            # self.video_filename = os.path.join(self.video_dir, f'drive_{time.strftime("%Y%m%d-%H%M%S")}.mp4')
            # self.video_fps = 20.0
            # self.video_writer = None


    # def toggle_recording(self):
    #     self.record_video = not self.record_video
    #     if self.record_video:
    #         # Start a new recording
    #         self.video_filename = os.path.join(self.video_dir, f'drive_{time.strftime("%Y%m%d-%H%M%S")}.mp4')
    #         self.video_writer = None  # Will be initialized on next frame
    #         print(f"Video recording started: {self.video_filename}")
    #     else:
    #         # Stop recording
    #         if self.video_writer is not None:
    #             self.video_writer.release()
    #             self.video_writer = None
    #             print(f"Video recording stopped: {self.video_filename}")

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
            self.running = True
            clock = pygame.time.Clock()
            
            while self.running:
                try:
                    self.milliseconds = clock.tick_busy_loop(30)  # 30 FPS
                
                    # Process events
                    for event in pygame.event.get():
                        self.general_control(control, event)
                    
                    # Get pressed keys
                    keys = pygame.key.get_pressed()
                
                    # Handle control based on mode
                    if self.autonomous_mode:
                        self.autonomous_control(control, keys)
                    else:
                        self.manual_control(control, keys)
                    if self.vehicle.is_at_traffic_light():
                        traffic_light = self.vehicle.get_traffic_light()
                        if traffic_light.get_state() == carla.TrafficLightState.Red:
                            traffic_light.set_state(carla.TrafficLightState.Green)
                    
                        # Apply the control to the vehicle
                    self.vehicle.apply_control(control)
                
                    # Update the display to show both camera views
                    try:
                        self.display_manager.update_display(self.camera_manager.rgb_surface,
                                                            self.camera_manager.lane_surface,
                                                            self.camera_manager.seg_surface,
                                                            self.camera_manager.bev_surface,
                                                            self.camera_manager.polylines_surface)
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


    def general_control(self, control, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                print("Control deactivated")
                self.running = False
            # elif event.key == pygame.K_r:
            #     self.toggle_recording()
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


    def autonomous_control(self, control, keys):

        state = self.mpc_controller.get_vehicle_state()
        if state is None:
            print("Vehicle state not available")
            return
        try:
            if self.control_mode == "PID":
                lane_error = 0.0
                # Get lane error from the detector
                if hasattr(self.detector, 'lane_detector') and hasattr(self.detector.lane_detector, 'lane_Error'):
                    lane_error = self.detector.lane_detector.lane_Error
                    control.steer, control.throttle = self.pid_controller.update(lane_error)
            elif self.control_mode == "MPC":
                try:
                    if hasattr(self.detector, 'lane_detector') and hasattr(self.detector.lane_detector, 'midCoeffs'):
                        mid_coeffs = self.detector.lane_detector.midCoeffs
                except Exception as e:
                    print(f"Error accessing midCoeffs: {e}")
                if mid_coeffs:
                    control.steer, control.throttle = self.mpc_controller.compute_control(mid_coeffs)
        except Exception as e:
            print(f"Error in autonomous control: {e}")
            self.autonomous_mode = False  # Revert to manual if error

    def manual_control(self, control, keys):

        """
            Manual control of the vehicle using keyboard input.
            Uses WASD controls similar to CARLA's manual_control.py
        """
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
        steer_increment = 5e-4 * self.milliseconds
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


    def cleanup(self):

        # if self.record_video and self.video_writer is not None:
        #     self.video_writer.release()
        #     print(f"Video saved to {self.video_filename}")
            
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