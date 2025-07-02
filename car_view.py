import zenoh
import time
import sys
import base64
import numpy as np
import cv2
import queue
import signal

frame_queue = queue.Queue(maxsize=5)
ipm_queue = queue.Queue(maxsize=5)
lane_queue = queue.Queue(maxsize=5)
obj_queue = queue.Queue(maxsize=5)
traffic_queue = queue.Queue(maxsize=5)
trafficLight_queue = queue.Queue(maxsize=5)

display_active = True
display_window_name = "CARLA Camera Feed"

def frame_subscriber_handler(sample):
    """Process incoming Zenoh messages with camera frames"""
    try:
        frame = sample.payload.to_bytes()
        
        print(f"Received data length: {len(frame)}")
        
        img_data = np.frombuffer(frame, dtype=np.uint8)

        try:
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print("cv2.imdecode returned None")
                return
                
            print(f"Successfully decoded image: {img.shape}")
            try:
                frame_queue.put(img, block=False)
            except queue.Full:
                try:
                    frame_queue.get_nowait()  # Remove oldest
                    frame_queue.put(img, block=False)  # Add new
                except:
                    pass
                
        except Exception as e:
            print(f"OpenCV error during imdecode: {e}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

def ipm_subscriber_handler(sample):
    """Process incoming Zenoh messages with camera frames"""
    try:
        debug = sample.payload.to_bytes()
        
        print(f"Received data length: {len(debug)}")
        
        img_data = np.frombuffer(debug, dtype=np.uint8)
        
        try:
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print("cv2.imdecode returned None")
                return
                
            print(f"Successfully decoded image: {img.shape}")
            try:
                ipm_queue.put(img, block=False)
            except queue.Full:
                try:
                    ipm_queue.get_nowait()  # Remove oldest
                    ipm_queue.put(img, block=False)  # Add new
                except:
                    pass
                
        except Exception as e:
            print(f"OpenCV error during imdecode: {e}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

def lane_mask_subscriber_handler(sample):
    """Process incoming Zenoh messages with camera frames"""
    try:
        debug = sample.payload.to_bytes()
        
        print(f"Received data length: {len(debug)}")
        
        img_data = np.frombuffer(debug, dtype=np.uint8)
        
        try:
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print("cv2.imdecode returned None")
                return
                
            print(f"Successfully decoded image: {img.shape}")
            try:
                lane_queue.put(img, block=False)
            except queue.Full:
                try:
                    lane_queue.get_nowait()  # Remove oldest
                    lane_queue.put(img, block=False)  # Add new
                except:
                    pass
                
        except Exception as e:
            print(f"OpenCV error during imdecode: {e}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

def obj_mask_subscriber_handler(sample):
    """Process incoming Zenoh messages with camera frames"""
    try:
        debug = sample.payload.to_bytes()
        
        print(f"Received data length: {len(debug)}")
        
        img_data = np.frombuffer(debug, dtype=np.uint8)
        
        try:
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print("cv2.imdecode returned None")
                return
                
            print(f"Successfully decoded image: {img.shape}")
            try:
                obj_queue.put(img, block=False)
            except queue.Full:
                try:
                    obj_queue.get_nowait()  # Remove oldest
                    obj_queue.put(img, block=False)  # Add new
                except:
                    pass
                
        except Exception as e:
            print(f"OpenCV error during imdecode: {e}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

def traffic_mask_subscriber_handler(sample):
    """Process incoming Zenoh messages with camera frames"""
    try:
        debug = sample.payload.to_bytes()
        
        print(f"Received data length: {len(debug)}")
        
        img_data = np.frombuffer(debug, dtype=np.uint8)
        
        try:
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print("cv2.imdecode returned None")
                return
                
            print(f"Successfully decoded image: {img.shape}")
            try:
                traffic_queue.put(img, block=False)
            except queue.Full:
                try:
                    traffic_queue.get_nowait()  # Remove oldest
                    traffic_queue.put(img, block=False)  # Add new
                except:
                    pass
                
        except Exception as e:
            print(f"OpenCV error during imdecode: {e}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

def trafficLight_mask_subscriber_handler(sample):
    """Process incoming Zenoh messages with camera frames"""
    try:
        debug = sample.payload.to_bytes()
        
        print(f"Received data length: {len(debug)}")
        
        img_data = np.frombuffer(debug, dtype=np.uint8)
        
        try:
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print("cv2.imdecode returned None")
                return
                
            print(f"Successfully decoded image: {img.shape}")
            try:
                trafficLight_queue.put(img, block=False)
            except queue.Full:
                try:
                    trafficLight_queue.get_nowait()  # Remove oldest
                    trafficLight_queue.put(img, block=False)  # Add new
                except:
                    pass
                
        except Exception as e:
            print(f"OpenCV error during imdecode: {e}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

def signal_handler(sig, frame):
    """Handle clean shutdown on CTRL+C"""
    global display_active
    print("\nShutting down...")
    display_active = False
    cv2.destroyAllWindows()
    sys.exit(0)

# def draw_class_bboxes(mask, class_indices, color_map, margin=10):
#     """Draw bounding boxes for each class index in class_indices on the mask, with a margin."""
#     mask_out = mask.copy()
#     h, w = mask.shape[:2]
#     for class_idx in class_indices:
#         # Create binary mask for this class
#         binary = np.all(mask == color_map[class_idx], axis=-1).astype(np.uint8) * 255
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in contours:
#             x, y, box_w, box_h = cv2.boundingRect(cnt)
#             if box_w > 5 and box_h > 5:  # Filter out tiny boxes
#                 # Expand the bounding box by margin, but keep within image bounds
#                 x1 = max(x - margin, 0)
#                 y1 = max(y - margin, 0)
#                 x2 = min(x + box_w + margin, w - 1)
#                 y2 = min(y + box_h + margin, h - 1)
#                 cv2.rectangle(mask_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     return mask_out

def main():
    global display_active
    
    signal.signal(signal.SIGINT, signal_handler)
    
    test_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Waiting for camera feed...", (50, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    test_ipm = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(test_ipm, "Waiting for IPM view...", (50, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    test_lane_mask = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(test_lane_mask, "Waiting for lane mask feed...", (50, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    test_obj_mask = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(test_obj_mask, "Waiting for obj mask view...", (50, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    test_traffic_mask = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(test_traffic_mask, "Waiting for traffic sign mask view...", (50, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    test_trafficLight_mask = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(test_traffic_mask, "Waiting for traffic light mask view...", (50, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Initialize latest images with placeholders
    latest_frame = test_frame.copy()
    latest_ipm = test_ipm.copy()
    latest_lane_mask = test_lane_mask.copy()
    latest_obj_mask = test_obj_mask.copy()
    latest_traffic_mask = test_traffic_mask.copy()
    latest_trafficLight_mask = test_trafficLight_mask.copy()
    
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)

    create_and_show_grid(latest_frame, latest_ipm, latest_lane_mask, latest_obj_mask, latest_traffic_mask, latest_trafficLight_mask)
    
    # Configuration for subscriber
    config = zenoh.Config()
    config.insert_json5("connect/endpoints", '["udp/100.119.72.83:7447", "udp/100.119.72.83:7450"]')
    # config.insert_json5("connect/endpoints", '["udp/100.117.122.95:7447", "udp/100.117.122.95:7450"]')

    session = zenoh.open(config)
    
    ipm_frame_key = "Vehicle/1/Ipm"
    original_frame_key = "Vehicle/1/Frame"
    lane_mask_key = "Vehicle/1/LaneMask"
    obj_mask_key = "Vehicle/1/ObjMask"
    traffic_mask_key = "Vehicle/1/TrafficMask"
    trafficLight_mask_key = "Vehicle/1/TrafficLightMask"
    subscriber_frame = session.declare_subscriber(original_frame_key, frame_subscriber_handler)
    subscriber_ipm = session.declare_subscriber(ipm_frame_key, ipm_subscriber_handler)
    subscriber_lane_mask = session.declare_subscriber(lane_mask_key, lane_mask_subscriber_handler)
    subscriber_obj_mask = session.declare_subscriber(obj_mask_key, obj_mask_subscriber_handler)
    subscriber_traffic_mask = session.declare_subscriber(traffic_mask_key, traffic_mask_subscriber_handler)
    subscriber_trafficLight_mask = session.declare_subscriber(trafficLight_mask_key, trafficLight_mask_subscriber_handler)
    
    print("Waiting for images... (Press CTRL+C to exit)")
    
    try:
        print("Entering main display loop")
        while display_active:
            update_needed = False
            
            try:
                if not frame_queue.empty():
                    latest_frame = frame_queue.get(block=False)
                    frame_queue.task_done()
                    update_needed = True
            except:
                pass
            
            try:
                if not ipm_queue.empty():
                    latest_ipm = ipm_queue.get(block=False)
                    ipm_queue.task_done()
                    update_needed = True
            except:
                pass

            try:
                if not lane_queue.empty():
                    latest_lane_mask = lane_queue.get(block=False)
                    lane_queue.task_done()
                    update_needed = True
            except:
                pass

            try:
                if not obj_queue.empty():
                    latest_obj_mask = obj_queue.get(block=False)
                    obj_queue.task_done()
                    update_needed = True
            except:
                pass

            try:
                if not traffic_queue.empty():
                    latest_traffic_mask = traffic_queue.get(block=False)
                    traffic_queue.task_done()
                    update_needed = True
            except:
                pass

            try:
                if not trafficLight_queue.empty():
                    latest_trafficLight_mask = trafficLight_queue.get(block=False)
                    trafficLight_queue.task_done()
                    update_needed = True
            except:
                pass
            
            if update_needed:
                create_and_show_grid(latest_frame, latest_ipm, latest_lane_mask, latest_obj_mask, latest_traffic_mask, latest_trafficLight_mask)
            
            # Process key events
            key = cv2.waitKey(1)
            if key == ord('q'):
                display_active = False
                break
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping by user request")
    finally:
        display_active = False
        cv2.destroyAllWindows()
        session.close()
        print("Clean shutdown complete")
        
def create_and_show_grid(frame, ipm, lane_mask, obj_mask, traffic_mask, trafficLight_mask):
    """Create a 2x2 grid with images sized to a fixed output resolution"""
    
    # Define a fixed output size
    FIXED_WIDTH = 1920  # 3 columns
    FIXED_HEIGHT = 960  # 2 rows
    
    # Calculate cell dimensions
    cell_width = FIXED_WIDTH // 3
    cell_height = FIXED_HEIGHT // 2
    
    # Resize frame and masks to fit their respective cells
    frame_resized = cv2.resize(frame, (cell_width, cell_height))
    lane_mask_resized = cv2.resize(lane_mask, (cell_width, cell_height))
    obj_mask_resized = cv2.resize(obj_mask, (cell_width, cell_height))
    traffic_mask_resized = cv2.resize(traffic_mask, (cell_width, cell_height))
    trafficLight_mask_resized = cv2.resize(trafficLight_mask, (cell_width, cell_height))
    
    # For IPM, preserve aspect ratio and pad with black
    ipm_h, ipm_w = ipm.shape[:2]
    ipm_aspect = ipm_w / ipm_h
    
    if ipm_aspect > (cell_width / cell_height):  # wider than tall
        resize_width = cell_width
        resize_height = int(resize_width / ipm_aspect)
    else:  # taller than wide
        resize_height = cell_height
        resize_width = int(resize_height * ipm_aspect)
    
    # Resize IPM while maintaining aspect ratio
    ipm_resized = cv2.resize(ipm, (resize_width, resize_height))
    
    # Create a black cell for IPM placement
    ipm_cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    
    # Calculate position to center the IPM in its cell
    y_offset = (cell_height - resize_height) // 2
    x_offset = (cell_width - resize_width) // 2
    
    # Place IPM in the black cell
    ipm_cell[y_offset:y_offset+resize_height, x_offset:x_offset+resize_width] = ipm_resized
    
    # Create overlays with consistent sizing
    lane_mask_overlayed = cv2.addWeighted(frame_resized, 0.7, lane_mask_resized, 0.3, 0)

    color_map = {
        2: [255, 0, 0],       # Car
        3: [250, 170, 30],    # Traffic Light
        4: [220, 220, 0],     # Traffic Sign
        5: [220, 20, 60],     # Person
        6: [244, 35, 232],    # Sidewalks
        7: [0, 0, 70],        # Motorcycle
    }

    # obj_mask_with_boxes = draw_class_bboxes(obj_mask_resized, [4], color_map, margin=15)

    obj_mask_overlayed = cv2.addWeighted(frame_resized, 0.7, obj_mask_resized, 0.3, 0)
    
    # Create a fixed-size canvas
    combined_img = np.zeros((FIXED_HEIGHT, FIXED_WIDTH, 3), dtype=np.uint8)
    
    # Top row
    combined_img[0:cell_height, 0:cell_width] = frame_resized
    combined_img[0:cell_height, cell_width:2*cell_width] = ipm_cell
    combined_img[0:cell_height, 2*cell_width:3*cell_width] = lane_mask_overlayed
    # Bottom row
    combined_img[cell_height:2*cell_height, 0:cell_width] = obj_mask_overlayed
    combined_img[cell_height:2*cell_height, cell_width:2*cell_width] = traffic_mask_resized
    combined_img[cell_height:2*cell_height, 2*cell_width:3*cell_width] = trafficLight_mask_resized
    
    # Add labels to each quadrant
    cv2.putText(combined_img, "Camera Feed", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined_img, "IPM View", (cell_width + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(combined_img, "Lane Mask", (2 * cell_width + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(combined_img, "Object Mask", (10, cell_height + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(combined_img, "Traffic Sign Mask", (cell_width + 10, cell_height + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(combined_img, "Traffic Light Mask", (2 * cell_width + 10, cell_height + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Set the window to fixed size
    cv2.resizeWindow(display_window_name, FIXED_WIDTH, FIXED_HEIGHT)
    
    # Display the combined image
    cv2.imshow(display_window_name, combined_img)
    cv2.waitKey(1)

if __name__ == "__main__":
    main()
