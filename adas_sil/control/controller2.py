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

class Controller:
    def __init__(self, vehicle, world):
        pygame.init()
        pygame.font.init()
        self.vehicle = vehicle
        self.world = world
        self.rgb_cam = None
        self.sem_cam = None
        self._steer_cache = 0.0

        # Track vehicle state for MPC
        self.current_speed = 0.0
        self.current_position = carla.Location(0, 0, 0)
        self.current_heading = 0.0
        
        # Controller mode selection
        self.control_mode = "PID"  # "PID" or "MPC"

        # Initialize controllers
        try:
            # Initialize PID parameters: Kp, Ki, Kd, base_speed, dt
            self.pid_controller = pid_controller_py.PidController()
            self.pid_controller.init(5.0, 0.01, 5.0, 0.5, 0.02)
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

        
    def setup_cameras(self):
        # RGB camera setup
        rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', '384')
        rgb_bp.set_attribute('image_size_y', '192')
        rgb_bp.set_attribute("fov", str(105))
        rgb_location = carla.Location(2, 0, 1.5)
        rgb_rotation = carla.Rotation(-15, 0, 0)  # Forward-facing camera
        rgb_transform = carla.Transform(rgb_location, rgb_rotation)
        self.rgb_cam = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        
        # Semantic segmentation camera setup
        sem_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute("image_size_x", '384')
        sem_bp.set_attribute("image_size_y", '192')
        sem_bp.set_attribute("fov", str(105))
        sem_location = carla.Location(2, 0, 1.5)
        sem_rotation = carla.Rotation(-15, 0, 0)  # Same as RGB camera
        sem_transform = carla.Transform(sem_location, sem_rotation)
        self.sem_cam = self.world.spawn_actor(sem_bp, sem_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        #Cam to view detection 
        # self.rgb_detcam = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

    def visualize_ipm_region(self, image):
        """Draw the source region that's being transformed to bird's eye view"""
        if not hasattr(self, 'ipm'):
            return image
        
        # Get a copy of the image to draw on
        vis_image = image.copy()
        
        # Get the four source points from the IPM
        source_points = self.ipm.orig_points
        
        # Draw the region on the original image
        points_np = np.array(source_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_image, [points_np], True, (0, 0, 255), 2)
        
        # Label the region
        cv2.putText(vis_image, "IPM Region", (points_np[0][0][0], points_np[0][0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return vis_image

    def process_rgb_image(self, image):
        # Save the image to disk
        
        # Convert CARLA image to numpy array for OpenCV
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel

        # Draw the IPM region on the original image
        array_with_region = self.visualize_ipm_region(array)
        
        # Convert BGR to RGB for pygame display
        array_rgb = cv2.cvtColor(array_with_region, cv2.COLOR_BGR2RGB)
        
        # Create pygame surface
        self.rgb_surface = pygame.surfarray.make_surface(array_rgb.swapaxes(0, 1))
        # Save frame to video if recording
        if self.record_video:
            
            # Initialize VideoWriter on first frame
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
                self.video_writer = cv2.VideoWriter(
                    self.video_filename, 
                    fourcc, 
                    self.video_fps, 
                    (image.width, image.height)
                )
                print(f"Recording video to {self.video_filename}")

            # Write the frame
            self.video_writer.write(array)

        array = array[:, :, ::-1]  # Convert BGR to RGB
        
        # Create pygame surface (swapaxes is needed as pygame uses a different coordinate system)
        # self.rgb_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        # array = array[:, :, :3]  # Remove alpha channel
        
        # if image.frame % 60 == 0:
        #     image.save_to_disk(f'carla_recordings/rgb/%08d' % image.frame)

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


    def process_rgb_with_detection(self, image):
        # Convert CARLA image to numpy array for OpenCV
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        
        # Process with detection model
        lane_img, mask_img = self.detector.processing(image, None)
        
        # Convert BGR (OpenCV) to RGB (Pygame)
        lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
        # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        
        # Create pygame surfaces
        # self.rgb_surface = pygame.surfarray.make_surface(lane_img.swapaxes(0, 1))
        self.lane_surface = pygame.surfarray.make_surface(mask_img.swapaxes(0, 1))
        
        # Save images periodically if needed
        # if image.frame % 60 == 0:
        #     image.save_to_disk(f'carla_recordings/rgb/%08d' % image.frame)
        #     cv2.imwrite(f'{self.output_dir}/lanes/lane_{image.frame:06d}.png', 
        #             cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))


    
    def process_semantic_image(self, image):
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        
        # In CARLA semantic segmentation, the tag is stored in the red channel
        semantic_tags = array[:, :, 2]
        
        # Create binary masks for road and sidewalk
        road_mask = (semantic_tags == 7).astype(np.uint8) * 255
        sidewalk_mask = (semantic_tags == 8).astype(np.uint8) * 255
        
        # Create a clean visualization image
        vis_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        
        # Add lane markings in white
        vis_img[semantic_tags == 6] = [255, 255, 255]
        
        # Extract road-sidewalk boundary edges
        # Method 1: Dilate sidewalk and find intersection with road
        kernel = np.ones((3,3), np.uint8)
        sidewalk_dilated = cv2.dilate(sidewalk_mask, kernel, iterations=1)
        road_dilated = cv2.dilate(road_mask, kernel, iterations=1)
        
        # The boundary is where dilated road and dilated sidewalk overlap
        boundary = cv2.bitwise_and(sidewalk_dilated, road_dilated)
        
        # Add curb edges in white as well
        vis_img[boundary > 0] = [255, 255, 255]
        
        # Create pygame surface for visualization
        self.seg_surface = pygame.surfarray.make_surface(vis_img.swapaxes(0, 1))
        self.semantic_processed_image = vis_img
    
        self.process_semantic_for_detection(image, vis_img)

        # # Save lane mask periodically if needed
        # if image.frame % 60 == 0:
        #     cv2.imwrite(f'{self.output_dir}/lanes/lane_{image.frame:06d}.png', 
        #             cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    def process_semantic_for_detection(self, original_image, semantic_image):

        start_time = time.time()
        """Process semantic segmentation image for lane detection"""
        if len(semantic_image.shape) == 3:
            lane_mask = cv2.cvtColor(semantic_image, cv2.COLOR_BGR2GRAY)
        else:
            lane_mask = semantic_image
        
        # Convert CARLA original image to OpenCV format  
        original_frame = self.detector.convert_Carla_image(original_image)

        # Process the mask through lane detector
        lane_img, left_points, right_points, bev_img = self.detector.process_segmentation_mask(lane_mask, original_frame)
        # If we got a valid BEV image, convert and display it
        if bev_img is not None and isinstance(bev_img, np.ndarray) and bev_img.size > 0:
            # Get the middle coefficients if available
            mid_coeffs = None
            if hasattr(self.detector.lane_detector, 'midCoeffs'):
                mid_coeffs = self.detector.lane_detector.midCoeffs
                if mid_coeffs is not None and len(mid_coeffs) >= 4:
                    # Convert from cv::Mat to Python list if needed
                    if hasattr(mid_coeffs, 'tolist'): 
                        mid_coeffs = mid_coeffs.tolist()
                        
                    # Draw the middle lane polynomial
                    bev_img = self.draw_lane_polynomial(bev_img, mid_coeffs, (0, 255, 255), 3)
                    
                    # Add text label for middle lane
                    cv2.putText(bev_img, "Mid Lane", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert to RGB for pygame
            bev_rgb = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
            self.bev_surface = pygame.surfarray.make_surface(bev_rgb.swapaxes(0, 1))
            # print("BEV image successfully received and converted to surface")
        # else:
            # print("BEV image not available from lane detector")
        # Transform only the lane points to bird's eye view
            # Get all the lane points if available
        # if hasattr(self.detector.lane_detector, 'bev_image'):
        #         bev_img = self.detector.lane_detector.bev_image
        #         if bev_img is not None:
        #             # Convert to RGB for pygame
        #             bev_rgb = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
        #             self.bev_surface = pygame.surfarray.make_surface(bev_rgb.swapaxes(0, 1))

        # Convert BGR (OpenCV) to RGB (Pygame)
        lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
        
        # Create pygame surfaces
        self.lane_surface = pygame.surfarray.make_surface(lane_img.swapaxes(0, 1))
        
        # Display polylines visualization if available
        if hasattr(self.detector, 'polylines_viz') and self.detector.polylines_viz is not None:
            polylines_rgb = cv2.cvtColor(self.detector.polylines_viz, cv2.COLOR_BGR2RGB)
            self.polylines_surface = pygame.surfarray.make_surface(polylines_rgb.swapaxes(0, 1))
            # print("Polylines visualization available")

        process_time = time.time() - start_time
        if process_time > 0.01:  # Log only if processing takes more than 100ms
            print(f"Semantic processing took {process_time:.3f} seconds")

        return lane_img

    def draw_lane_polynomial(self, bev_img, coeffs, color=(0, 255, 255), thickness=2):
        """
        Draw polynomial curve on bird's eye view image
        
        Args:
            bev_img: Bird's eye view image
            coeffs: List of polynomial coefficients [a, b, c, d] for ax^3 + bx^2 + cx + d
            color: Line color in BGR format
            thickness: Line thickness
        """
        if coeffs is None or len(coeffs) < 4 or bev_img is None:
            return bev_img
            
        # Get image dimensions
        h, w = bev_img.shape[:2]
        
        # Create points for the curve
        points = []
        for y in range(0, h, 5):  # Plot every 5 pixels
            # Calculate x value using polynomial: ax^3 + bx^2 + cx + d
            # Note: In BEV, y increases from top to bottom, so we invert the coordinates
            # for the polynomial if needed
            normalized_y = y / h  # Normalize to 0-1 range
            
            # Calculate x using polynomial coefficients
            # Adjust this calculation based on how your coefficients are defined
            x = int(coeffs[0] * y**3 + coeffs[1] * y**2 + coeffs[2] * y + coeffs[3])
            
            # Only add point if it's within image bounds
            if 0 <= x < w:
                points.append((x, y))
        
        # Draw the curve if we have enough points
        if len(points) >= 2:
            # Convert to numpy array for OpenCV
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(bev_img, [pts], False, color, thickness)
            
            # Mark the start and end points
            if len(points) > 0:
                cv2.circle(bev_img, points[0], 5, (0, 0, 255), -1)  # Red dot at start
                cv2.circle(bev_img, points[-1], 5, (255, 0, 0), -1)  # Blue dot at end
        
        return bev_img

    def update_display(self):
             
        # Clear display
        self.display.fill((0, 0, 0))
        
        # Display RGB image on left side
        if self.rgb_surface is not None:
            self.display.blit(self.rgb_surface, (0, 0))
        
        # Display lane mask on right side
        if self.lane_surface is not None:
            self.display.blit(self.lane_surface, (390, 0))
        
        # Display seg mask below
        if self.seg_surface is not None:
            self.display.blit(self.seg_surface, (0, 195))

        # if self.bev_surface is not None:
        #     self.display.blit(self.bev_surface, (1000, 0))

        # if hasattr(self.detector, 'lane_points_vis'):
        #     lane_points_vis_rgb = cv2.cvtColor(self.detector.lane_points_vis, cv2.COLOR_BGR2RGB)
        #     lane_points_surface = pygame.surfarray.make_surface(lane_points_vis_rgb.swapaxes(0, 1))
        #     self.display.blit(lane_points_surface, (384, 480))
        

            # Display bird's eye view on bottom-right
        if hasattr(self, 'bev_surface') and self.bev_surface is not None:
            self.display.blit(self.bev_surface, (390, 195))
            
            # Add label
            font = pygame.font.SysFont('Arial', 24)
            text = font.render('Bird\'s Eye View', True, (255, 255, 255))
            self.display.blit(text, (780, 490))
            # Update the display

        # Display polylines visualization in a new position (could also replace one of the above)
        if hasattr(self, 'polylines_surface') and self.polylines_surface is not None:
            self.display.blit(self.polylines_surface, (780, 0))  # Adjust position as needed
            # Add label
            font = pygame.font.SysFont('Arial', 18)
            text = font.render('Lane Polylines', True, (255, 255, 255))
            self.display.blit(text, (780, 180))

        # Display current speed
        if hasattr(self, 'current_speed'):
            speed_text = font.render(f"Speed: {self.current_speed:.1f} km/h", True, (255, 255, 255))
            self.display.blit(speed_text, (20, 60))

        pygame.display.flip()

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