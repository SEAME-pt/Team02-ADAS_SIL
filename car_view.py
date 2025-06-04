#!/usr/bin/env python3
# filepath: /home/luis_t2/SEAME/Team02-CarlaSimulation/car_view.py

import zenoh
import time
import sys
import base64
import numpy as np
import cv2
import threading
import queue
import signal

# Create a global queue for images
image_queue = queue.Queue(maxsize=10)  # Limit queue size to avoid memory issues
display_active = True
display_window_name = "CARLA Camera Feed"
def message_handler(sample):
    """Process incoming Zenoh messages with camera frames"""
    print("Message received, processing...")
    try:
        # Get payload as string or bytes
        if hasattr(sample.payload, 'as_string'):
            base64_str = sample.payload.as_string()
        else:
            base64_str = sample.payload.decode('utf-8')
        
        print(f"Received data length: {len(base64_str)}")
        
        # Decode from base64
        try:
            img_bytes = base64.b64decode(base64_str)
            print(f"Decoded to {len(img_bytes)} bytes of image data")
        except Exception as e:
            print(f"Base64 decoding failed: {e}")
            return
        
        # Convert to numpy and decode image
        img_data = np.frombuffer(img_bytes, dtype=np.uint8)
        
        # Decode with explicit error handling
        try:
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print("cv2.imdecode returned None")
                return
                
            print(f"Successfully decoded image: {img.shape}")
            
            # Add to queue instead of displaying directly
            try:
                image_queue.put(img, timeout=0.1)
                print("Added image to display queue")
            except queue.Full:
                print("Queue full, skipping frame")
                # Queue is full, skip this frame
                pass
                
        except Exception as e:
            print(f"OpenCV error during imdecode: {e}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

def display_thread_function():
    """Thread to handle displaying images from the queue"""
    global display_active
    
    print("Display thread started")
    frame_counter = 0
    
    while display_active:
        try:
            # Get image from queue with timeout
            img = image_queue.get(timeout=0.1)
            frame_counter += 1
            
            print(f"Got frame #{frame_counter} from queue, size: {img.shape}")
            
            # Display the image - window already exists
            if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                # Make a copy to avoid potential memory issues
                display_img = img.copy()
                
                # Add frame counter to image
                cv2.putText(display_img, f"Frame: {frame_counter}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Use this to force window update - critical fix!
                cv2.imshow(display_window_name, display_img)
                print(f"Displayed frame #{frame_counter}")
                
                # Save occasional frames to verify content
                if frame_counter % 30 == 0:  # Save every 30th frame
                    cv2.imwrite(f"frame_{frame_counter}.jpg", display_img)
                    print(f"Saved frame_{frame_counter}.jpg")
            else:
                print("Received invalid image")
                
            # Mark task as done
            image_queue.task_done()
        except queue.Empty:
            # No image in queue, just continue
            pass
        except Exception as e:
            print(f"Display error in loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.01)

def signal_handler(sig, frame):
    """Handle clean shutdown on CTRL+C"""
    global display_active
    print("\nShutting down...")
    display_active = False
    time.sleep(0.5)  # Give threads time to notice
    sys.exit(0)

def main():
    global display_active
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize OpenCV window system with explicit thread
    print("Initializing OpenCV window system...")
    cv2.startWindowThread()
    
    # Create window in main thread
    print("Creating window in main thread...")
    test_img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(test_img, "Waiting for images...", (50, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(display_window_name, 800, 600)
    cv2.imshow(display_window_name, test_img)
    cv2.waitKey(1000)  # Give it more time to render initially
    
    # Start display thread - which now only updates the existing window
    display_thread = threading.Thread(target=display_thread_function)
    display_thread.daemon = True
    display_thread.start()
    
    # Configuration for subscriber
    config = zenoh.Config()
    
    # Connect to the publisher's IP address
    config.insert_json5(zenoh.config.CONNECT_KEY, '["tcp/100.117.122.95:7447"]')
    
    # Enable peer discovery
    config.insert_json5("scouting/multicast/enabled", "true")

    print("Initializing Zenoh subscriber...")
    session = zenoh.open(config)
    
    # Subscribe to the camera frame topic
    key = "carla/frame"
    subscriber = session.declare_subscriber(key, message_handler)
    
    print(f"Subscribed to '{key}'")
    print("Waiting for images... (Press CTRL+C to exit)")
    
    # Main loop - also handles updating the window periodically from main thread
    try:
        while display_active:
            # Process any pending UI events and keep window responsive
            key = cv2.waitKey(1)  # Short timeout keeps UI responsive
            
            # Allow exiting with 'q' key
            if key == ord('q'):
                display_active = False
                break
                
            time.sleep(0.01)  # Prevent high CPU usage
    except KeyboardInterrupt:
        print("\nStopping by user request")
    finally:
        # Clean shutdown
        display_active = False
        display_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        session.close()
        print("Clean shutdown complete")

if __name__ == "__main__":
    main()