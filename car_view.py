import zenoh
import time
import sys
import base64
import numpy as np
import cv2
import queue
import signal

# Create a global queue for images
image_queue = queue.Queue(maxsize=15)  # Limit queue size to avoid memory issues
display_active = True
display_window_name = "CARLA Camera Feed"

def message_handler(sample):
    """Process incoming Zenoh messages with camera frames"""
    try:
        # # Get payload as string or bytes
        # if hasattr(sample.payload, 'as_string'):
        base64_str = sample.payload.as_string()
        # else:
        #     base64_str = sample.payload.decode('utf-8')
        
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
                image_queue.put(img, block=False)
            except queue.Full:
                try:
                    image_queue.get_nowait()  # Remove oldest
                    image_queue.put(img, block=False)  # Add new
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
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create window in main thread
    print("Creating window in main thread...")
    test_img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(test_img, "Waiting for images...", (50, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Initialize OpenCV window system
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(display_window_name, 800, 600)
    cv2.imshow(display_window_name, test_img)
    cv2.waitKey(1)
    
    # Configuration for subscriber
    config = zenoh.Config()
    config.insert_json5(zenoh.config.CONNECT_KEY, '["tcp/100.117.122.95:7447"]')
    config.insert_json5("scouting/multicast/enabled", "true")

    session = zenoh.open(config)
    
    key = "carla/frame"
    subscriber = session.declare_subscriber(key, message_handler)
    
    print(f"Subscribed to '{key}'")
    print("Waiting for images... (Press CTRL+C to exit)")
    
    try:
        print("Entering main display loop")
        while display_active:
            # Check for new image in queue
            try:
                img = image_queue.get(block=True, timeout=0.1)
                
                cv2.imshow(display_window_name, img)
                
                image_queue.task_done()
            except queue.Empty:
                pass
            
            # Process any pending UI events - CRITICAL for displaying images
            key = cv2.waitKey(1)
            
            # Allow exiting with 'q' key
            if key == ord('q'):
                display_active = False
                break
            
            # Short sleep to prevent high CPU usage
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