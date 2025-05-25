
import sys
import os
import numpy as np
import cv2
import keyboard
import time
import pygame

from adas_sil.perception.Detection import Detection
opencv_bin = "C:/Users/manue/opencv/build/x64/vc16/bin"
os.environ["PATH"] = opencv_bin + os.pathsep + os.environ["PATH"]

carla_egg = "C:/Users/manue/Documents/SEA_ME/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg"
sys.path.append(carla_egg)


import carla

class CameraManager:
    def __init__(self, vehicle, world, display, detector):
        self.world = world
        self.vehicle = vehicle
        self.rgb_cam = None
        self.sem_cam = None
        self.rgb_detcam = None
        self.display = display
        self.detector = detector

        self.setup_cameras()

        # self.rgb_cam.listen(self.process_rgb_image)
        # self.sem_cam.listen(self.process_semantic_image)
        self.detector.load_model(self.rgb_detcam)
        self.rgb_detcam.listen(self.process_rgb_with_detection)
        self.rgb_surface = None
        self.lane_surface = None
        self.seg_surface = None
        self.bev_surface = None
        self.polylines_surface = None

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
        self.rgb_detcam = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

    def visualize_ipm_region(self, image):
        """Draw the source region that are  being transformed to bird's eye view"""
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
        # if self.record_video:
        #     # Initialize VideoWriter on first frame
        #     if self.video_writer is None:
        #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        #         self.video_writer = cv2.VideoWriter(
        #             self.video_filename, 
        #             fourcc, 
        #             self.video_fps, 
        #             (image.width, image.height)
        #         )
        #         print(f"Recording video to {self.video_filename}")

            # # Write the frame
            # self.video_writer.write(array)

        array = array[:, :, ::-1]  # Convert BGR to RGB
        
        # Create pygame surface (swapaxes is needed as pygame uses a different coordinate system)
        # self.rgb_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        # array = array[:, :, :3]  # Remove alpha channel
        
        # if image.frame % 60 == 0:
        #     image.save_to_disk(f'carla_recordings/rgb/%08d' % image.frame)


    def process_rgb_with_detection(self, image):
        # Convert CARLA image to numpy array for OpenCV
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        
        # Process with detection model
        lane_img, mask_img = self.detector.processing(image, None)
        
        # Convert BGR (OpenCV) to RGB (Pygame)
        lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
        
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
            
        #     # Mark the start and end points
        #     if len(points) > 0:
        #         cv2.circle(bev_img, points[0], 5, (0, 0, 255), -1)  # Red dot at start
        #         cv2.circle(bev_img, points[-1], 5, (255, 0, 0), -1)  # Blue dot at end
        
        return bev_img

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