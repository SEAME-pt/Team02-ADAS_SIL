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

def signal_handler(sig, frame):
    """Handle clean shutdown on CTRL+C"""
    global display_active
    print("\nShutting down...")
    display_active = False
    cv2.destroyAllWindows()
    sys.exit(0)

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
    
    # Initialize latest images with placeholders
    latest_frame = test_frame.copy()
    latest_ipm = test_ipm.copy()
    latest_lane_mask = test_lane_mask.copy()
    latest_obj_mask = test_obj_mask.copy()
    
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)

    create_and_show_grid(latest_frame, latest_ipm, latest_lane_mask, latest_obj_mask)
    
    # Configuration for subscriber
    config = zenoh.Config()
    # config.insert_json5("connect/endpoints", '["udp/100.119.72.83:7447", "udp/100.119.72.83:7450"]')
    config.insert_json5("connect/endpoints", '["udp/100.117.122.95:7447", "udp/100.117.122.95:7450"]')
    config.insert_json5("scouting/multicast/enabled", "true")

    session = zenoh.open(config)
    
    ipm_frame_key = "Vehicle/1/Ipm"
    original_frame_key = "Vehicle/1/Frame"
    lane_mask_key = "Vehicle/1/LaneMask"
    obj_mask_key = "Vehicle/1/ObjMask"
    subscriber_frame = session.declare_subscriber(original_frame_key, frame_subscriber_handler)
    subscriber_ipm = session.declare_subscriber(ipm_frame_key, ipm_subscriber_handler)
    subscriber_lane_mask = session.declare_subscriber(lane_mask_key, lane_mask_subscriber_handler)
    subscriber_obj_mask = session.declare_subscriber(obj_mask_key, obj_mask_subscriber_handler)
    
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
            
            if update_needed:
                create_and_show_grid(latest_frame, latest_ipm, latest_lane_mask, latest_obj_mask)
            
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
def create_and_show_grid(frame, ipm, lane_mask, obj_mask):
    """Create a 2x2 grid with images sized relative to IPM height"""
    
    # Use IPM height as reference
    reference_height = ipm.shape[0]
    
    # Resize other images to match IPM height while preserving aspect ratio
    frame_ratio = frame.shape[1] / frame.shape[0]
    frame_resized = cv2.resize(frame, (int(reference_height * frame_ratio), reference_height))
    
    reference_width = frame_resized.shape[1]

    lane_mask_resized = cv2.resize(lane_mask, (reference_width, reference_height))
    
    obj_mask_resized = cv2.resize(obj_mask, (reference_width, reference_height))

    lane_mask_overlayed = cv2.addWeighted(frame,0.4,lane_mask_resized,0.1,0)

    obj_mask_overlayed = cv2.addWeighted(frame,0.4,obj_mask_resized,0.1,0)
    
    row1_height = reference_height
    row2_height = reference_height
    col1_width = max(frame_resized.shape[1], lane_mask_overlayed.shape[1])
    col2_width = max(ipm.shape[1], obj_mask_overlayed.shape[1])
    
    # Create a blank canvas large enough to hold all images
    total_height = row1_height + row2_height
    total_width = col1_width + col2_width
    combined_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    # Place images at their respective positions
    # Top-left: frame
    combined_img[0:frame_resized.shape[0], 0:frame_resized.shape[1]] = frame_resized
    
    # Top-right: ipm (original size - our reference)
    combined_img[0:ipm.shape[0], col1_width:col1_width+ipm.shape[1]] = ipm
    
    # Bottom-left: lane_mask
    combined_img[row1_height:row1_height+lane_mask_overlayed.shape[0], 
                0:lane_mask_overlayed.shape[1]] = lane_mask_overlayed
    
    # Bottom-right: obj_mask
    combined_img[row1_height:row1_height+obj_mask_overlayed.shape[0], 
                col1_width:col1_width+obj_mask_overlayed.shape[1]] = obj_mask_overlayed
    
    # Add labels to each quadrant
    cv2.putText(combined_img, "Camera Feed", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined_img, "IPM View", (col1_width + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(combined_img, "Lane Mask", (10, row1_height + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(combined_img, "Object Mask", (col1_width + 10, row1_height + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    # Resize the window to fit the combined image
    cv2.resizeWindow(display_window_name, total_width, total_height)
    
    # Display the combined image
    cv2.imshow(display_window_name, combined_img)
    cv2.waitKey(1)

if __name__ == "__main__":
    main()
