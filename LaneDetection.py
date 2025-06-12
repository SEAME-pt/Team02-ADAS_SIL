import sys
import random
import time
import numpy as np
import cv2
import math
import torch
import pygame
import onnxruntime as ort
import torch.nn.functional as F
from torchvision import transforms
import queue
import threading

# Create a queue to pass image data between threads
image_queue = queue.Queue(maxsize=1)

class LaneDetection:
    def __init__(self, display):
        self.display = display
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)
            
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.input_size = (1024, 512)
        self.ort_session = None

        # Define normalizer with ImageNet stats
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Update transform pipeline to match your test cases
        self.transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            self.normalizer
        ])

    def load_model(self):            
        model_path = "/home/luis_t2/SEAME/Team02-Course/MachineLearning/LaneDetection/Models/onnx/lane_Yolo_Carla3_epoch_3.onnx"
        # Set compute options based on device
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        print(f"ONNX model loaded from {model_path}")

    def camera_callback(self, image):
        """This function is called in the camera thread"""
        # Convert CARLA image to OpenCV format
        original_img = self.convert_Carla_image(image)
        
        # # Process image and run inference using the new preprocessing
        input_tensor, processed_img = self.preprocess_image(original_img, target_size=self.input_size)
        
        # ONNX inference
        input_name = self.ort_session.get_inputs()[0].name
        output_names = [output.name for output in self.ort_session.get_outputs()]
        input_data = input_tensor.cpu().numpy()
        outputs = self.ort_session.run(output_names, {input_name: input_data})
        
        # Convert outputs to torch tensor for consistency
        prediction = torch.from_numpy(outputs[0])

        # Create overlay with the new function
        overlayed_img = self.overlay_predictions(original_img, prediction)
        
        # Get vehicle speed if available
        speed_kmh = 0
        if hasattr(image, 'parent') and image.parent is not None:
            vehicle = image.parent
            velocity = vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Put the processed image and data in the queue
        try:
            image_queue.put_nowait({
                'image': original_img,
                'overlay': overlayed_img,
                'speed': speed_kmh,
                'timestamp': image.timestamp
            })
        except queue.Full:
            # Queue is full, remove old item and add new one
            try:
                image_queue.get_nowait()
                image_queue.put_nowait({
                    'image': original_img,
                    'overlay': overlayed_img,
                    'speed': speed_kmh,
                    'timestamp': image.timestamp
                })
            except (queue.Empty, queue.Full):
                pass

    def preprocess_image(self, image, target_size=(256, 128)):
        # Resize image
        img = cv2.resize(image, target_size)
        
        # 2. Enhance contrast within the ROI
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        img_tensor = self.transform_pipeline(img).unsqueeze(0).to(self.device)
        
        return img_tensor, img

    def post_process(self, lane_mask, kernel_size=10, min_area=50, max_lanes=6):
        """
        Color lanes based on their horizontal position: left=red, center=yellow, right=green
        """
        # Ensure input is binary uint8 image
        if lane_mask.dtype is not np.uint8:
            lane_mask = np.array(lane_mask, np.uint8)
        if len(lane_mask.shape) == 3:
            lane_mask = cv2.cvtColor(lane_mask, cv2.COLOR_BGR2GRAY)

        # Create a colored mask (3-channel)
        colored_lanes = np.zeros((lane_mask.shape[0], lane_mask.shape[1], 3), dtype=np.uint8)

        # Fill the pixel gap using Closing operator (dilation followed by erosion)
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(kernel_size, kernel_size))

        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            lane_mask, connectivity=8, ltype=cv2.CV_32S)
        
        # Define standard colors for lane positions
        lane_position_colors = [
            [0, 0, 255],      # Far left: Red
            [0, 128, 255],    # Left: Orange
            [0, 255, 255],    # Center left: Yellow
            [0, 255, 0],      # Center right: Green
            [255, 0, 0],      # Right: Blue
            [255, 0, 255],    # Far right: Purple
        ]
        
        # Create a list of valid components with their centroids
        valid_components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                # Get centroid x-position
                center_x = centroids[i][0]
                valid_components.append((i, center_x))
        
        # # With this Mean Shift implementation:
        # if valid_components:
        #     # Extract x-positions of all valid components
        #     positions = np.array([x_pos for _, x_pos in valid_components]).reshape(-1, 1)
            
        #     # Apply Mean Shift clustering using the cluster function
        #     num_clusters, labels, centers = cluster(positions)
            
        #     if num_clusters > 0:
        #         # Group components by their cluster
        #         clusters = {}
        #         for i, (comp_idx, x_pos) in enumerate(valid_components):
        #             cluster_id = labels[i]
        #             if cluster_id not in clusters:
        #                 clusters[cluster_id] = []
        #             clusters[cluster_id].append((comp_idx, x_pos))
                
        #         # For each cluster, take the component closest to center
        #         keep_components = []
        #         for cluster_id, components in clusters.items():
        #             center = centers[cluster_id][0]
        #             # Find component closest to cluster center
        #             closest = min(components, key=lambda x: abs(x[1] - center))
        #             keep_components.append(closest)
                
        #         # Sort final components by position
        #         keep_components.sort(key=lambda x: x[1])
        #     else:
        #         keep_components = []
        # else:
        #     keep_components = []

        # Sort components by area (largest first)
        area_sorted = sorted(valid_components, key=lambda x: x[1])
        # # Keep only the largest max_lanes components
        keep_components = area_sorted[:max_lanes]
        
        # For storing lane polylines
        lane_polylines = []
        
        # Process each lane
        for idx, (comp_idx, _) in enumerate(keep_components):
            # Get this lane's mask
            lane_mask = (labels == comp_idx).astype(np.uint8) * 255
            
            # Color fill the lane in the colored mask
            lane = (labels == comp_idx)
            # color = lane_position_colors[min(idx, len(lane_position_colors)-1)]
            color = [0, 255, 0]
            # colored_lanes[lane] = color
            
            # Extract lane coordinates for polyline
            # First, find contours to get a rough outline
            contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Extract main points to represent the lane
                lane_points = []
                h, w = lane_mask.shape
                
                # Sample points from top to bottom (every 5 pixels)
                for y in range(0, h, 5):
                    # Find all points at this y-level
                    x_points = np.where(lane_mask[y, :] > 0)[0]
                    if len(x_points) > 0:
                        # Use the middle point at this y-level
                        mid_x = (np.min(x_points) + np.max(x_points)) // 2
                        lane_points.append([mid_x, y])
                
                if lane_points:
                    lane_polyline = np.array(lane_points)
                    lane_polylines.append((lane_polyline, color))
                    
                    # Draw the polyline on the colored mask
                    cv2.polylines(
                        img=colored_lanes,
                        pts=[lane_polyline],
                        isClosed=False,
                        color=color,
                        thickness=5)
        
        return colored_lanes
    
    # Update your overlay_predictions function
    def overlay_predictions(self, image, prediction, threshold=0.5):
        # Convert prediction to binary mask
        prediction = prediction.squeeze().cpu().detach().numpy()
        lane_mask = (prediction > threshold).astype(np.uint8) * 255
        
        # Resize mask to match the original image size
        lane_mask_resized = cv2.resize(lane_mask, (image.shape[1], image.shape[0]))
        
        # Create a copy of the original image for our final result
        result = image.copy()
        
        # Create a colormap for the raw detections (blue with transparency)
        raw_overlay = np.zeros_like(image)
        raw_overlay[lane_mask_resized > 0] = [0, 255, 0]  # Blue tint for raw detections
        
        # Apply the raw detection overlay with transparency
        result = cv2.addWeighted(result, 1.0, raw_overlay, 0.5, 0)
        
        # # Apply lane connection post-processing to get polylines
        # colored_mask = self.post_process(lane_mask_resized)
        
        # # Apply the polyline overlay with transparency
        # # Using a higher alpha to make polylines more visible
        # result = cv2.addWeighted(result, 1.0, colored_mask, 0.5, 0)
        
        return result

    def convert_Carla_image(self, frame):
        # Convert CARLA raw data (BGRA) to a BGR image
        array = np.frombuffer(frame.raw_data, dtype=np.uint8)
        array = array.reshape((frame.height, frame.width, 4))
        frame = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        return frame

    def update_display(self):
        """Update the pygame display with the latest processed image.
        This function should be called from the main thread."""
        try:
            # Try to get the latest frame data without waiting
            data = image_queue.get_nowait()
            
            # Get the overlay image - already processed with lane prediction
            overlayed_img = data['overlay']
            
            # Convert to RGB for Pygame
            rgb_overlay = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
            
            # Convert to Pygame surface
            surface = pygame.surfarray.make_surface(np.transpose(rgb_overlay, (1, 0, 2)))
            
            # Clear display and show the image
            self.display.blit(surface, (0, 0))
            
            # Update the display
            pygame.display.flip()
            
            return True
        except queue.Empty:
            # No new frame available
            return False