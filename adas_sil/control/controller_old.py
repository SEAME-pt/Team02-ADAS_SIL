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
    import cpp_postprocessing.build.ipm_module as ipm_module
    print("Successfully imported IPM_Module")
except ImportError as e:
    print(f"Failed to import IPM_Module: {e}")

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

        try:
            # Initialize detector
            from adas_sil.perception.Detection import Detection

            try:
                # self.ipm = ipm_module.IPM()
                # Create a new pygame surface for bird's eye view
                self.bev_surface = None
                print("IPM module initialized for bird's eye view")
            except Exception as e:
                print(f"Error initializing IPM module: {e}")
            
            self.detector = Detection()
            # Initialize pygame display for visualization
            self.display = pygame.display.set_mode((1280, 960))  # Wide enough for two images side by side
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

        try:
            # Get camera parameters
            camera_height = 1.5  # rgb_location.z
            camera_pitch = 15.0  # Negate rgb_rotation.pitch as IPM uses opposite convention
            h_fov = rgb_bp.get_attribute("fov").as_float()
            
            # Get image dimensions as integers
            img_width = rgb_bp.get_attribute('image_size_x').as_int()
            img_height = rgb_bp.get_attribute('image_size_y').as_int()
            
            # Calculate vertical FOV
            v_fov = 2 * np.degrees(np.arctan( (img_height/img_width) * np.tan( np.radians(h_fov)/2 ) ))
            
            # Create size tuple
            img_size = (img_width, img_height)
            bev_size = (384, 384)   # Square bird's eye view
            
            self.ipm.initialize(img_size, bev_size)
            
            # Calibrate for CARLA camera parameters
            self.ipm.calibrate_from_camera(
                camera_height=camera_height,
                camera_pitch=camera_pitch,
                horizontal_fov=h_fov,
                vertical_fov=v_fov,
                near_distance=1.5,   # 3 meters from camera
                far_distance=15.0,   # 20 meters from camera
                lane_width=7       # Standard lane width
            )

            # try:
            #     source_points = self.ipm.orig_points
            #     print("\n==== IPM TRAPEZOID CORNERS ====")
            #     print(f"Top-left     (far left):     ({source_points[0][0]:.1f}, {source_points[0][1]:.1f})  → ({-3.5/2:.1f}m, {20:.1f}m)")
            #     print(f"Top-right    (far right):    ({source_points[1][0]:.1f}, {source_points[1][1]:.1f})  → ({3.5/2:.1f}m, {20:.1f}m)")
            #     print(f"Bottom-right (near right):   ({source_points[2][0]:.1f}, {source_points[2][1]:.1f})  → ({3.5/2:.1f}m, {3:.1f}m)")
            #     print(f"Bottom-left  (near left):    ({source_points[3][0]:.1f}, {source_points[3][1]:.1f})  → ({-3.5/2:.1f}m, {3:.1f}m)")
            #     print("===============================\n")
            # except Exception as e:
            #     print(f"Error printing trapezoid points: {e}")
            
            print("IPM calibrated with camera parameters")
        except Exception as e:
            print(f"Error calibrating IPM: {e}")
    
    def create_birds_eye_view(self, image):
        """Create bird's eye view from camera image"""
        try:
            # Check if we have a valid image and IPM is initialized
            if image is None or not hasattr(self, 'ipm'):
                return None
                
            # Convert image to numpy array if it's a CARLA image
            if hasattr(image, 'raw_data'):
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                array = array[:, :, :3]  # Remove alpha channel
            else:
                # Assume it's already a numpy array
                array = image
                
            # Apply IPM transformation to get bird's eye view
            bev_img = self.ipm.apply_ipm(array)
            
            # Return the transformed image
            return bev_img
            
        except Exception as e:
            print(f"Error creating bird's eye view: {e}")
            return None

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

    def transform_lane_points(self, points, original_size):
        """Transform lane points to bird's eye view perspective"""
        try:
            if not points or not hasattr(self, 'ipm'):
                return []
                
            # Get the perspective transformation matrix
            transform_matrix = self.ipm.perspective_matrix
                
            # Convert points to the format expected by perspectiveTransform
            # Convert to numpy float32 array with shape (N, 1, 2)
            points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
                
            # Apply the perspective transformation to points directly
            transformed_points = cv2.perspectiveTransform(points_array, transform_matrix)
                
            # Reshape back to a simple list of points
            return transformed_points.reshape(-1, 2).tolist()
                
        except Exception as e:
            print(f"Error transforming lane points: {e}")
            import traceback
            traceback.print_exc()
            return points  # Return original points on error

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
        lane_img, left_points, right_points = self.detector.process_segmentation_mask(lane_mask, original_frame)

        # Transform only the lane points to bird's eye view
            # Get all the lane points if available
        all_points = self.detector.lane_detector.all_lane_points if hasattr(self.detector.lane_detector, 'all_lane_points') else []
        
        # Create bird's eye view of lane points
        bev_img = self.create_lane_points_bev(all_points, left_points, right_points)
        
        # Convert to RGB for pygame
        bev_rgb = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
        self.bev_surface = pygame.surfarray.make_surface(bev_rgb.swapaxes(0, 1))

        # Convert BGR (OpenCV) to RGB (Pygame)
        lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
        
        # Create pygame surfaces
        self.lane_surface = pygame.surfarray.make_surface(lane_img.swapaxes(0, 1))
        
        process_time = time.time() - start_time
        if process_time > 0.01:  # Log only if processing takes more than 100ms
            print(f"Semantic processing took {process_time:.3f} seconds")

        return lane_img

    def create_lane_points_bev(self, all_points=None, left_points=None, right_points=None):
        """Create a bird's eye view visualization with 0.1s decision points"""
        # Create a blank image for bird's eye view
        bev_img = np.zeros((384, 384, 3), dtype=np.uint8)
        
        # Draw a grid for reference
        grid_size = 50
        for i in range(0, 384, grid_size):
            cv2.line(bev_img, (i, 0), (i, 384), (50, 50, 50), 1)  # Vertical lines
            cv2.line(bev_img, (0, i), (384, i), (50, 50, 50), 1)  # Horizontal lines
        
        # Draw vehicle indicator at the bottom center
        cv2.circle(bev_img, (192, 350), 8, (0, 200, 255), -1)  # Orange circle
        cv2.fillPoly(bev_img, [np.array([(177, 350), (207, 350), (192, 320)])], (0, 200, 255))  # Triangle
        
        # Transform lane points
        transformed_left = []
        transformed_right = []

        if left_points:
            transformed_left = self.transform_lane_points(left_points, (384, 192))
            # Draw all left lane points (thin red)
            for pt in transformed_left:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < 384 and 0 <= y < 384:
                    cv2.circle(bev_img, (x, y), 2, (0, 0, 100), -1)  # Dark red (thin)
        
        if right_points:
            transformed_right = self.transform_lane_points(right_points, (384, 192))
            # Draw all right lane points (thin green)
            for pt in transformed_right:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < 384 and 0 <= y < 384:
                    cv2.circle(bev_img, (x, y), 2, (0, 100, 0), -1)  # Dark green (thin)
        
        # Calculate and draw center line with 0.1s decision points
        if transformed_left and transformed_right and len(transformed_left) > 0 and len(transformed_right) > 0:
            # Get vehicle speed
            velocity = self.vehicle.get_velocity()
            speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s
            speed_kmh = speed_ms * 3.6  # km/h for display
            
            # Calculate distance traveled in 0.1 seconds (in meters)
            distance_per_step = speed_ms * 0.1  # meters per 0.1s
            
            # Convert meters to pixels based on IPM calibration
            # Lane width is 7m and spans ~384 pixels horizontally
            # The near-far distance is 13.5m (15-1.5) and spans ~384 pixels vertically
            pixels_per_meter = 384 / 13.5  # Vertical scale
            
            # Distance in pixels per 0.1s
            pixel_step = distance_per_step * pixels_per_meter
            
            # Center points between lanes
            center_points = []
            
            # Calculate center points from left and right lanes
            min_len = min(len(transformed_left), len(transformed_right))
            for i in range(min_len):
                center_x = (transformed_left[i][0] + transformed_right[i][0]) / 2
                center_y = (transformed_left[i][1] + transformed_right[i][1]) / 2
                center_points.append((center_x, center_y))
                
            # Sort center points by y (distance from car)
            center_points.sort(key=lambda pt: pt[1], reverse=True)
            
            # Create 0.1s decision points along centerline
            decision_points = []
            
            # Start from the bottom of the image (closest to car)
            if center_points:
                # Start from closest point to car (highest y value)
                current_y = center_points[0][1]
                last_idx = 0
                
                # Add first point
                decision_points.append((center_points[0][0], center_points[0][1]))
                
                # Add points at 0.1s intervals based on speed
                while pixel_step > 0 and current_y > 0 and last_idx < len(center_points) - 1:
                    # Move up by pixel_step (corresponds to 0.1s travel)
                    current_y -= pixel_step
                    
                    # Find the closest center point to this y position
                    for i in range(last_idx, len(center_points)):
                        if center_points[i][1] <= current_y:
                            decision_points.append((center_points[i][0], center_points[i][1]))
                            last_idx = i
                            break
                
                # Connect all centerline points with thin yellow line
                if len(center_points) >= 2:
                    pts = np.array(center_points, dtype=np.int32)
                    cv2.polylines(bev_img, [pts], False, (0, 180, 255), 1)
                
                # Draw decision points (larger yellow circles with time markers)
                for i, pt in enumerate(decision_points):
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < 384 and 0 <= y < 384:
                        # Draw larger yellow circle for each 0.1s point
                        cv2.circle(bev_img, (x, y), 6, (0, 255, 255), -1)
                        
                        # Label with time
                        time_sec = (i * 0.1)
                        if time_sec < 1.0:
                            time_label = f"{time_sec:.1f}s"
                        else:
                            time_label = f"{time_sec:.1f}s"
                            
                        cv2.putText(bev_img, time_label, (x + 10, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Display speed information
            # cv2.putText(bev_img, f"Speed: {speed_kmh:.1f} km/h ({speed_ms:.1f} m/s)", (10, 40), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.putText(bev_img, f"Decision interval: 0.1s = {distance_per_step:.2f}m", (10, 60), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add title
        # cv2.putText(bev_img, "Bird's Eye View with 0.1s Decision Points", (10, 20), 
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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

        pygame.display.flip()


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
                    milliseconds = clock.tick_busy_loop(10)  # 60 FPS
                    
                    # Process events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                print("Manual control deactivated")
                                self.vehicle.set_autopilot(True)
                            elif event.key == pygame.K_r:  # 'R' key toggles recording
                                self.toggle_recording()
                    
                    # Get pressed keys
                    keys = pygame.key.get_pressed()
                    
                    # Handle throttle - same as _parse_vehicle_keys
                    if keys[pygame.K_w]:
                        control.throttle = min(control.throttle + 0.01, 1.0)
                        control.brake = 0.0
                    else:
                        control.throttle = 0.0  # No throttle decay, immediate zero
                    
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