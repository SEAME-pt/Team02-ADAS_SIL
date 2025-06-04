import zenoh
import time
import sys
import base64
import numpy as np
import cv2
import queue
import signal

frame_queue = queue.Queue(maxsize=15)
debug_queue = queue.Queue(maxsize=15)
display_active = True
display_window_name = "CARLA Camera Feed"

def frame_subscriber_handler(sample):
    """Process incoming Zenoh messages with camera frames"""
    try:
        base64_str = sample.payload.to_string()
        
        print(f"Received data length: {len(base64_str)}")
        
        # Decode from base64
        try:
            img_bytes = base64.b64decode(base64_str)
        except Exception as e:
            print(f"Base64 decoding failed: {e}")
            return
        
        img_data = np.frombuffer(img_bytes, dtype=np.uint8)
        
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

def debug_subscriber_handler(sample):
    """Process incoming Zenoh messages with camera frames"""
    try:
        base64_str = sample.payload.to_string()
        
        print(f"Received data length: {len(base64_str)}")
        
        # Decode from base64
        try:
            img_bytes = base64.b64decode(base64_str)
        except Exception as e:
            print(f"Base64 decoding failed: {e}")
            return
        
        img_data = np.frombuffer(img_bytes, dtype=np.uint8)
        
        try:
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print("cv2.imdecode returned None")
                return
                
            print(f"Successfully decoded image: {img.shape}")
            try:
                debug_queue.put(img, block=False)
            except queue.Full:
                try:
                    debug_queue.get_nowait()  # Remove oldest
                    debug_queue.put(img, block=False)  # Add new
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
    
    test_debug = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(test_debug, "Waiting for debug view...", (50, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Initialize latest images with placeholders
    latest_frame = test_frame.copy()
    latest_debug = test_debug.copy()
    
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(display_window_name, 800, 600)

    max_height = max(latest_frame.shape[0], latest_debug.shape[0])
    total_width = latest_frame.shape[1] + latest_debug.shape[1]
    combined_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    
    combined_img[0:latest_frame.shape[0], 0:latest_frame.shape[1]] = latest_frame
    combined_img[0:latest_debug.shape[0], latest_frame.shape[1]:latest_frame.shape[1]+latest_debug.shape[1]] = latest_debug
    
    cv2.putText(combined_img, "Camera Feed", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined_img, "Debug View", (latest_frame.shape[1] + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow(display_window_name, combined_img)
    cv2.waitKey(1)
    
    # Configuration for subscriber
    config = zenoh.Config()
    config.insert_json5("connect/endpoints", '["tcp/100.117.122.95:7447", "tcp/100.117.122.95:7450"]')
    config.insert_json5("scouting/multicast/enabled", "true")

    session = zenoh.open(config)
    
    frame_key = "carla/frame"
    debug_key = "carla/debug"
    subscriber_frame = session.declare_subscriber(frame_key, frame_subscriber_handler)
    subscriber_debug = session.declare_subscriber(debug_key, debug_subscriber_handler)
    
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
                if not debug_queue.empty():
                    latest_debug = debug_queue.get(block=False)
                    debug_queue.task_done()
                    update_needed = True
            except:
                pass
            
            if update_needed:
                max_height = max(latest_frame.shape[0], latest_debug.shape[0])
                total_width = latest_frame.shape[1] + latest_debug.shape[1]
                combined_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)
                
                combined_img[0:latest_frame.shape[0], 0:latest_frame.shape[1]] = latest_frame
                combined_img[0:latest_debug.shape[0], latest_frame.shape[1]:latest_frame.shape[1]+latest_debug.shape[1]] = latest_debug
                
                # Add labels
                cv2.putText(combined_img, "Camera Feed", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(combined_img, "Debug View", (latest_frame.shape[1] + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Update the window
                cv2.imshow(display_window_name, combined_img)
            
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

if __name__ == "__main__":
    main()