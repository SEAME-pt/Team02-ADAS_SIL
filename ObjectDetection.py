import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import torch
import pygame
import onnxruntime as ort
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import queue
import threading

# Create a queue to pass image data between threads
image_queue = queue.Queue(maxsize=1)

class ObjectDetection:
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

        self.input_size = (258, 126)
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
        model_path = "/home/luis_t2/SEAME/Team02-Course/MachineLearning/ObjectDetection/Models/onnx/obj_YOLO_1_epoch_198.onnx"
        # Set compute options based on device
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        print(f"ONNX model loaded from {model_path}")

    def camera_callback(self, image):
        """This function is called in the camera thread"""
        # Convert CARLA image to OpenCV format
        original_img = self.convert_Carla_image(image)
        
        # # Process image and run inference using the new preprocessing
        input_tensor, processed_img = self.preprocess_image(original_img)
        
        # ONNX inference
        input_name = self.ort_session.get_inputs()[0].name
        output_names = [output.name for output in self.ort_session.get_outputs()]
        input_data = input_tensor.cpu().numpy()
        outputs = self.ort_session.run(output_names, {input_name: input_data})
        
        # Convert outputs to torch tensor for consistency
        prediction = torch.from_numpy(outputs[0])

        # Create overlay with the new function
        overlayed_img, detected_objects = self.overlay_predictions(original_img, prediction)
        
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
    
    # Update your overlay_predictions function
    def overlay_predictions(self, image, prediction, show_debug=True):
        # Create a color map for classes
        color_map = {
            0: [0, 0, 0],         # Background
            1: [128, 64, 128],    # Road
            2: [0, 0, 142],       # Car
            3: [250, 170, 30],    # Traffic Light
            4: [220, 220, 0],     # Traffic Sign
            5: [220, 20, 60],     # Person
            6: [244, 35, 232],    # Sidewalks
            7: [0, 0, 70],        # Truck
            8: [0, 60, 100],      # Bus
            9: [0, 0, 230],       # Motorcycle
        }
        
        # Convert prediction logits to class indices
        _, predicted_class = torch.max(prediction, dim=1)
        predicted_class = predicted_class.squeeze().cpu().numpy()
        
        # Resize mask to match original image size
        predicted_class = cv2.resize(predicted_class.astype(np.uint8), 
                                    (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # Save original road mask for comparison
        original_road_mask = (predicted_class == 1).astype(np.uint8) * 255
        
        # IMPROVEMENT: Clean up road segmentation with morphological operations
        road_mask = original_road_mask.copy()

        # Define kernel - rectangular shape works well for roads
        kernel_size = 15  # Increase for more noticeable effect
        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT, 
            ksize=(kernel_size, kernel_size)
        )

        # Apply morphological closing to connect nearby road segments
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

        # Find connected components
        ccs = cv2.connectedComponentsWithStats(
            road_mask, connectivity=8, ltype=cv2.CV_32S)
        labels = ccs[1]
        stats = ccs[2]

        # Keep only the largest component (main road)
        # Ignore label 0 which is background
        if len(stats) > 1:
            # Find the largest component by area, excluding background (index 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            # Create mask with only the largest road component
            cleaned_mask = np.zeros_like(road_mask)
            cleaned_mask[labels == largest_label] = 255
            road_mask = cleaned_mask

        # Update the predicted class with the cleaned road mask
        predicted_class_cleaned = predicted_class.copy()
        predicted_class_cleaned[road_mask == 255] = 1
        
        # Create colored overlays
        overlay = image.copy()
        
        # Apply colors based on class prediction
        for class_idx, color in color_map.items():
            overlay[predicted_class_cleaned == class_idx] = color
        
        # Create car mask for finding car objects
        car_mask = (predicted_class_cleaned == 2).astype(np.uint8) * 255
        
        # Find contours of cars
        contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Dictionary to store detected objects
        detected_objects = {'cars': 0}
        
        # Draw bounding boxes around cars
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter out small detections (noise)
            if area > 300:  # Adjust this threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Calculate approximate distance (using bottom of bounding box)
                y_bottom = y + h
                distance_factor = 1.0 - (y_bottom / image.shape[0])
                estimated_distance = int(50 * distance_factor)  # Simple approximation
                
                # Label with estimated distance
                cv2.putText(overlay, f"{estimated_distance}m", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                detected_objects['cars'] += 1
        
        # Blend with original image
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        return result, detected_objects

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