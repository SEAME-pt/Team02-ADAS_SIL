import sys
import os
from tkinter import N

opencv_bin = "C:/Users/manue/opencv/build/x64/vc16/bin"
os.environ["PATH"] = opencv_bin + os.pathsep + os.environ["PATH"]

carla_egg = "C:/Users/manue/Documents/SEA_ME/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg"
sys.path.append(carla_egg)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from cpp_postprocessing.build.lane_detector_py import LaneProcessor as LaneDetector
    print("Successfully imported LaneProcessor")
except ImportError as e:
    print(f"Failed to import LaneProcessor: {e}")
    
import carla
import math
import random
import time
import numpy as np
import torch
import cv2
from PIL import Image
import torch.nn.functional as F

import torch
from torchvision import transforms
from PIL import Image


class Detection:

    """
    Lane detection system using hybrid deep learning and computer vision techniques.
    
    The Detection class combines neural network segmentation with traditional CV algorithms 
    to detect and fit lane boundaries. It uses a C++ backend through Python bindings 
    for efficient image processing and lane polynomial fitting.
    
    Key components:
        - Neural network inference using ONNX for lane segmentation
        - C++ LaneProcessor backend for lane fitting and tracking
        - Bird's Eye View (BEV) transformation for better lane visualization
        - Polynomial lane representation for autonomous driving control
        
    Detection pipeline:
        1. Convert Carla simulator images to OpenCV format
        2. Preprocess images for neural network inference
        3. Run inference using ONNX Runtime
        4. Process segmentation outputs with C++ LaneProcessor
        5. Extract lane polynomial coefficients for visualization and control
    
    Integrated with C++ LaneDetectorIPM through:
        - preProcess() - Image preprocessing
        - postProcess() - Lane detection algorithms
        - Left/right polynomial coefficients extraction
        - BEV image transformation
        - Lane error calculation for lateral control
    
    Available C++ binding properties (accessed via self.lane_detector):
        - left_coeffs - Polynomial coefficients for left lane [a, b, c] where x = a*y² + b*y + c
        - right_coeffs - Polynomial coefficients for right lane [a, b, c]
        - midCoeffs - Mid-lane polynomial coefficients derived from left and right lanes
        - left_points - List of points [(x,y),...] along the left lane curve
        - right_points - List of points [(x,y),...] along the right lane curve
        - all_lane_points - All detected lane points before left/right classification
        - bev_image - Bird's-eye view perspective transform of the lane detection
        - polylines_viz - Visualization of detected lane polylines
        - lane_Error - Lateral position error relative to lane center (float)
    
    The hybrid approach leverages deep learning robustness with 
    traditional CV efficiency for real-time lane detection.
    """

    # Define preprocessing transform (using torchvision)
    # transform_pipeline = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Print more detailed GPU info if available
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("No CUDA device detected! Running on CPU.")
            
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.model = None
        self.ort_session = None
        self.is_onnx = False
        self.lane_detector = LaneDetector()
        print(f"Detection initialized using {self.device}")


        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Update transform pipeline to match your test cases
        self.transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            self.normalizer
        ])

    def preprocess_image(self, image, target_size=(256, 128)):
        # Resize image
        img = cv2.resize(image, target_size)
        print(f"Image resized to: {img.shape}")
        # 2. Enhance contrast within the ROI
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Image converted to RGB: {img.shape}")
        # Apply transforms
        img_tensor = self.transform_pipeline(img).unsqueeze(0).to(self.device)
        print(f"Image tensor shape after transforms: {img_tensor.shape}")
        return img_tensor, img

    def load_model(self, camera):
        import onnxruntime as ort
            
        model_path = "C:\\Users\\manue\\Documents\\SEA_ME\\ADAS-SIL-validation\\models\\lane_segmentation_model.onnx" 
            
        # Set compute options based on device
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        print(f"ONNX model loaded from {model_path}")

    def processing(self, img, camera):

        # Convert CARLA image to OpenCV format
        frame = self.convert_Carla_image(img)
        print(f"converted image")
        
        # Create a copy of the original image to draw on
        display_img = frame.copy()
        
        try:
            # Step 1: Preprocess the frame using LaneDetector
            # frame = self.lane_detector.preProcess(frame)
            # prep rocessed_data = self.transform_pipeline(frame).unsqueeze(0).to(self.device)
            input_tensor, processed_img = self.preprocess_image(frame)
            # Step 3: Run inference with ONNX
            input_name = self.ort_session.get_inputs()[0].name
            output_names = [output.name for output in self.ort_session.get_outputs()]
            
            input_data = input_tensor.cpu().numpy()
            print(f"Input data shape: {input_data.shape}")
            # Run inference
            outputs = self.ort_session.run(output_names, {input_name: input_data})
            print(f"ONNX inference completed, output shape: {outputs[0].shape}")
            seg_mask = torch.from_numpy(outputs[0])

            # # Handle different output formats
            # if len(seg_mask.shape) == 4:  # [batch, channels, height, width]
            #     if seg_mask.shape[1] == 1:  # Single channel output
            #         # Create two channels from single channel (the second is inverse of first)
            #         channel1 = seg_mask[0, 0]  # Shape: [128, 256]
            #         channel2 = - channel1  # Inverse provides good separation
                    
            #         # Stack to create [2, 128, 256] format needed by C++ code
            #         proper_format = np.stack([channel1, channel2], axis=0)
            #         # print(f"Created 2-channel format from 1-channel: {proper_format.shape}")
                
            #     elif seg_mask.shape[1] == 2:  # Already has two channels
            #         # Just remove batch dimension: [1, 2, 128, 256] -> [2, 128, 256]
            #         proper_format = seg_mask[0]
            #         # print(f"Using existing 2-channel format: {proper_format.shape}")
                
            #     else:
            #         print(f"Unexpected channel count: {seg_mask.shape[1]}")
            #         # Adapt as needed
            #         proper_format = seg_mask[0]

            # else:
            #     # Handle unexpected format
            #     proper_format = seg_mask
            #     print(f"Unexpected shape: {seg_mask.shape}")


            # channel1 = proper_format[0].flatten()  # Shape: [32768]
            # channel2 = proper_format[1].flatten()  # Shape: [32768]~
            output_data = seg_mask[0].flatten()  # Shape: [32768]

            # # Concatenate to create the expected memory layout:
            # # [all of channel1 first, then all of channel2]
            # concatenated_data = np.concatenate([channel1, channel2])

            # # Make sure it's contiguous and float32
            output_data = np.ascontiguousarray(output_data, dtype=np.float32)
            print(f"Output data set in LaneDetector, shape: {output_data.shape}")
            self.lane_detector.setOutputData(output_data)
            print(f"AQUI")
            
            # Step 5: Postprocess to detect lanes
            self.lane_detector.postProcess(display_img)
            
            # Get lane coefficients if needed
            left_coeffs = self.lane_detector.left_coeffs
            right_coeffs = self.lane_detector.right_coeffs

            # # if left_coeffs and right_coeffs:
            # #     print(f"Left lane coefficients: {left_coeffs}")
            # #     print(f"Right lane coefficients: {right_coeffs}")

            
            # height, width, _ = display_img.shape
            print("AQUIIII")
            if left_coeffs and right_coeffs:
                # Draw polynomial lanes on display image
                display_img = self.draw_lane_polynomials(display_img, left_coeffs, right_coeffs)
            
            display_img = self.draw_roi_area(display_img)
            print("AQUIIII")

            binary_mask = (seg_mask[0] > 0.5).detach().cpu().numpy().astype(np.uint8) * 255
            binary_mask = binary_mask.squeeze(0)
            # Get the dimensions of the original image
            
            # If needed, resize the mask to match original image dimensions
            # if binary_mask.shape != (height, width):
            #     binary_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
        except Exception as e:
            print(f"Error in lane detection: {e}")
            # # Fallback visualization
            binary_mask = np.zeros_like(display_img)

        
        return display_img, binary_mask

    def process_segmentation_mask(self, segmentation_mask, original_image):
        """
            Process a binary segmentation mask through the lane detector
            
            Args:
                segmentation_mask: Binary mask where lanes/curbs are white (255)
                original_image: Original image for visualization
            
            Returns:
                Visualization image with lanes drawn and binary mask
        """

        left_points = []
        right_points = []
        try:
            # Create a copy for visualization
            display_img = original_image.copy()
            height, width = display_img.shape[:2]
            
            # Create the two channels the C++ code expects:
            # Channel 1: lane probability (1.0 where lane, 0.0 elsewhere)
            # Channel 2: non-lane probability (1.0 where not lane, 0.0 elsewhere)
            channel1 = (segmentation_mask > 127).astype(np.float32)
            channel2 = 1 - channel1
            print(f"Channel 1 shape: {channel1.shape}, Channel 2 shape: {channel2.shape}")  
            # Stack channels for visualization
            proper_format = np.stack([channel1, channel2], axis=0)
            
            # Flatten and concatenate channels as required by C++ code
            # The C++ code expects: [all_channel1_values, all_channel2_values]
            channel1_flat = proper_format[0].flatten()
            channel2_flat = proper_format[1].flatten()
            concatenated_data = np.concatenate([channel1_flat, channel2_flat])
            
            # Make sure it's contiguous and float32
            output_data = np.ascontiguousarray(concatenated_data, dtype=np.float32)
            expected_size = width * height * 2
            if output_data.size != expected_size:
                print(f"Warning: Output data size {output_data.size} doesn't match expected {expected_size}")
                return display_img, left_points, right_points, bev_img
            # Set data and run lane detection
            self.lane_detector.setOutputData(output_data)
            self.lane_detector.postProcess(display_img)
            
            # Get lane coefficients if needed
            # left_coeffs = self.lane_detector.left_coeffs
            # right_coeffs = self.lane_detector.right_coeffs

            # left_points = self.lane_detector.left_points
            # right_points = self.lane_detector.right_points
            # all_points = self.lane_detector.all_lane_points

            # try:
            #     points_vis = self.lane_detector.lane_points_visualization
            #     if isinstance(points_vis, np.ndarray) and points_vis.size > 0:
            #         self.lane_points_vis = points_vis
            #         # Count the number of lane points
            #         lane_points = self.lane_detector.all_lane_points
            #         # print(f"Detected {len(lane_points)} lane points")
            # except Exception as e:
            #     print(f"Error accessing lane points visualization: {e}")


            # for pt in all_points:
            #     x, y = pt
            #     cv2.circle(display_img, (int(x), int(y)), 2, (255, 255, 255), -1)  # White for all points

            # for pt in left_points:
            #     x, y = pt
            #     cv2.circle(display_img, (int(x), int(y)), 2, (255, 100, 100), -1)  # Light blue for left
            
            # for pt in right_points:
            #     x, y = pt
            #     cv2.circle(display_img, (int(x), int(y)), 2, (100, 255, 100), -1)  # Light green for right
        

            # # print(f"Original image size: {height}x{width}")
            # # print(f"Binary mask size: {binary_mask.shape}")
          
            # # Draw lanes if detected
            # if left_coeffs and right_coeffs:
            #     display_img = self.draw_lane_polynomials(display_img, left_coeffs, right_coeffs)
                
            # # Optionally draw ROI area
            # display_img = self.draw_roi_area(display_img)
             # Get the BEV image directly after postProcess
            bev_img = None
            try:
                # This will access the bev_image property from your binding
                bev_img = self.lane_detector.bev_image
                # print(f"BEV image shape: {bev_img.shape if bev_img is not None else 'None'}")
            except Exception as e:
                print(f"Error accessing BEV image: {e}")

            self.polylines_viz = None
            if hasattr(self.lane_detector, 'polylines_viz'):
                self.polylines_viz = self.lane_detector.polylines_viz
            
            # Return both the display image and the BEV image
            return display_img, left_points, right_points, bev_img
            
        except Exception as e:
            print(f"Error processing segmentation mask: {e}")
            return original_image, left_points, right_points, None

 
    def convert_Carla_image(self, frame):
        
        # Convert CARLA raw data (BGRA) to a BGR image
        array = np.frombuffer(frame.raw_data, dtype=np.uint8)
        array = array.reshape((frame.height, frame.width, 4))
        frame = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    
        return frame


    def process_input(self, frame):

        # Convert BGR frame (OpenCV) to RGB.
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        half = device.type != 'cpu'
        # Preprocess the image.
        input_tensor = self.transform_pipeline(pil_img).to(device)  # Shape: [3,256,256]
        #input_tensor = input_tensor.half() if half else input_tensor.float()
        input_tensor = input_tensor.float()
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension: 1x3x256x256
        return input_tensor

    def draw_lane_lines(self, img, result, thickness=3, draw_points=True):
        """
        Draw lane lines on the image with customizable styling.
        
        Parameters:
        - img: Input image to draw on
        - result: LaneResult object from lane_processor
        - thickness: Line thickness
        - draw_points: Whether to draw individual points on the curves
        
        Returns:
        - Image with lane lines drawn
        """
        # Create a copy of the image to draw on
        output = img.copy()
        
        # Draw left lane in blue
        if result.left_points:
            points = result.left_points
            for i in range(len(points)-1):
                cv2.line(output, 
                        (points[i][0], points[i][1]), 
                        (points[i+1][0], points[i+1][1]), 
                        (255, 0, 0), thickness)
                
            if draw_points:
                for pt in points:
                    cv2.circle(output, (pt[0], pt[1]), 4, (255, 0, 255), -1)
        
        # Draw right lane in green
        if result.right_points:
            points = result.right_points
            for i in range(len(points)-1):
                cv2.line(output, 
                        (points[i][0], points[i][1]), 
                        (points[i+1][0], points[i+1][1]), 
                        (0, 255, 0), thickness)
                
            if draw_points:
                for pt in points:
                    cv2.circle(output, (pt[0], pt[1]), 4, (255, 255, 0), -1)
        
        # Draw mid lane in red
        if result.mid_points:
            points = result.mid_points
            for i in range(len(points)-1):
                cv2.line(output, 
                        (points[i][0], points[i][1]), 
                        (points[i+1][0], points[i+1][1]), 
                        (0, 0, 255), 2)  # Slightly thinner for mid lane
        
        # Draw lateral error indicator at bottom of frame
        height, width = img.shape[:2]
        center_x = width // 2
        
        # Draw vertical center line
        cv2.line(output, (center_x, height-50), (center_x, height-10), (255, 255, 255), 2)
        
        # Draw error indicator
        if hasattr(result, 'lateral_error'):
            error_pixels = int(result.lateral_error * width / 4)  # Scale error to pixels
            error_x = center_x + error_pixels
            error_x = max(10, min(width-10, error_x))  # Keep in frame bounds
            
            # Draw error position
            cv2.circle(output, (error_x, height-30), 10, (0, 165, 255), -1)
            
            # Add error text
            cv2.putText(output, f"Error: {result.lateral_error:.2f}", 
                    (10, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (255, 255, 255), 2)
        
        # Add lane coefficient information if available
        if result.left_coeffs and len(result.left_coeffs) >= 3:
            # Format coefficients to 3 decimal places
            a, b, c = result.left_coeffs
            coeff_text = f"Left: {a:.3f}x² + {b:.3f}x + {c:.3f}"
            cv2.putText(output, coeff_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if result.right_coeffs and len(result.right_coeffs) >= 3:
            a, b, c = result.right_coeffs
            coeff_text = f"Right: {a:.3f}x² + {b:.3f}x + {c:.3f}"
            cv2.putText(output, coeff_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    def draw_lane_polynomials(self, display_img, left_coeffs=None, right_coeffs=None, steps=50, thickness=3):
        """
        Draw lane lines based on polynomial coefficients without scaling.
        Uses the coefficients directly as they come from the lane detector.
        
        Args:
            display_img: The image to draw on
            left_coeffs: List of coefficients [a, b, c, d] for x = a*y^3 + b*y^2 + c*y + d
            right_coeffs: List of coefficients [a, b, c, d] for x = a*y^3 + b*y^2 + c*y + d
            steps: Number of points to generate along each curve
            thickness: Line thickness
        
        Returns:
            Image with lane lines drawn
        """
        h, w = display_img.shape[:2]
        
        # Make a copy to avoid modifying original
        output = display_img.copy()
        
        # Define y coordinates (from bottom of image to middle)
        y_points = np.linspace(h-1, h//2, steps, dtype=np.int32)
        
        # Draw curves if coefficients available
        for lane_type, coeffs, color in [
            ("Left", left_coeffs, (255, 0, 0)),
            ("Right", right_coeffs, (0, 255, 0))
        ]:
            if coeffs and len(coeffs) >= 3:
                # Handle either cubic (4 coeffs) or quadratic (3 coeffs) polynomials
                if len(coeffs) >= 4:
                    a, b, c, d = coeffs[:4]
                    equation_text = f"{lane_type}: {a:.3f}y³ + {b:.3f}y² + {c:.3f}y + {d:.3f}"
                else:
                    # Fall back to quadratic if only 3 coefficients
                    a, b, c = coeffs[:3]
                    d = 0  # No cubic term
                    equation_text = f"{lane_type}: {a:.3f}y² + {b:.3f}y + {c:.3f}"
                
                # Draw points along the curve - use coefficients directly
                points = []
                for y in y_points:
                    # Calculate x using the polynomial equation with optional cubic term
                    if len(coeffs) >= 4:
                        x = int(a * (y**3) + b * (y**2) + c * y + d)
                    else:
                        x = int(a * (y**2) + b * y + c)
                    
                    # Only add points within image boundaries
                    if 0 <= x < w:
                        points.append((x, y))
                
                # Draw the curve if we have enough points
                if len(points) >= 2:
                    for i in range(len(points)-1):
                        cv2.line(output, points[i], points[i+1], color, thickness)
                    
                    # Add text showing equation
                    cv2.putText(output, equation_text, 
                            (10, 30 if lane_type == "Left" else 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # If both lanes are detected, draw the middle lane
        if left_coeffs and right_coeffs and len(left_coeffs) >= 3 and len(right_coeffs) >= 3:
            # Average coefficients for center line
            # Handle either cubic or quadratic case
            if len(left_coeffs) >= 4 and len(right_coeffs) >= 4:
                center_a = (left_coeffs[0] + right_coeffs[0]) / 2  # Cubic term
                center_b = (left_coeffs[1] + right_coeffs[1]) / 2  # Quadratic term
                center_c = (left_coeffs[2] + right_coeffs[2]) / 2  # Linear term
                center_d = (left_coeffs[3] + right_coeffs[3]) / 2  # Constant term
            else:
                # Fall back to quadratic if needed
                center_a = (left_coeffs[0] + right_coeffs[0]) / 2  # Quadratic term
                center_b = (left_coeffs[1] + right_coeffs[1]) / 2  # Linear term
                center_c = (left_coeffs[2] + right_coeffs[2]) / 2  # Constant term
                center_d = 0  # No cubic term
            
            # Draw center lane in red using same technique
            center_points = []
            for y in y_points:
                # Calculate x using the average coefficients
                if len(left_coeffs) >= 4 and len(right_coeffs) >= 4:
                    x = int(center_a * (y**3) + center_b * (y**2) + center_c * y + center_d)
                else:
                    x = int(center_a * (y**2) + center_b * y + center_c)
                
                if 0 <= x < w:
                    center_points.append((x, y))
            
            # Draw the center line
            if len(center_points) >= 2:
                for i in range(len(center_points)-1):
                    cv2.line(output, center_points[i], center_points[i+1], (0, 0, 255), 2)
        
        return output
    
    def draw_roi_area(self, display_img):
        """
        Draws the same ROI area that is used in the C++ LaneDetector code
        """
        h, w = display_img.shape[:2]
        
        # Define trapezoidal ROI (same as in C++ code)
        bottomY = h
        topY = int(h * 0.40)
        
        # The C++ code uses these extreme values for bottom corners
        # But let's make them visible in our display
        bottomLeftX = max(0, int(w * -1.5))  # Limit to visible area
        bottomRightX = min(w-1, int(w * 2.5))  # Limit to visible area
        
        topLeftX = int(w * 0.3)
        topRightX = int(w * 0.7)
        
        # Create trapezoid points list
        trapezoid = np.array([
            [bottomLeftX, bottomY],
            [bottomRightX, bottomY],
            [topRightX, topY],
            [topLeftX, topY]
        ], np.int32)
        
        # Draw filled semi-transparent ROI
        overlay = display_img.copy()
        cv2.fillPoly(overlay, [trapezoid], (0, 255, 255, 128))
        
        # Draw the outline in yellow
        cv2.polylines(display_img, [trapezoid], True, (0, 255, 255), 2)
        
        # Blend the overlay with semi-transparency
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)
        
        # Add text label
        cv2.putText(display_img, "ROI", (topLeftX + 10, topY + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return display_img