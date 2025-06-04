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

def display_thread_function():
    """Thread to handle displaying images from the queue"""
    global display_active
    
    # Create window once
    cv2.namedWindow("CARLA Camera Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CARLA Camera Feed", 800, 600)
    
    print("Display thread started")
    
    while display_active:
        try:
            # Get image from queue with timeout
            img = image_queue.get(timeout=0.1)
            
            # Display the image
            cv2.imshow("CARLA Camera Feed", img)
            key = cv2.waitKey(1)
            
            # Allow exiting with 'q' key
            if key == ord('q'):
                display_active = False
                break
                
            # Mark task as done
            image_queue.task_done()
        except queue.Empty:
            # No image in queue, just update the window
            cv2.waitKey(1)
        except Exception as e:
            print(f"Display error: {e}")
            time.sleep(0.1)  # Prevent tight loop on errors

    # Clean up
    cv2.destroyAllWindows()
    print("Display thread stopped")

def main():
    global display_active
    
    # Start display thread
    display_thread = threading.Thread(target=display_thread_function)
    display_thread.daemon = True
    display_thread.start()
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        global display_active
        print("\nShutting down...")
        display_active = False
        display_thread.join(timeout=1.0)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create a message handler function for Zenoh
    def message_handler(sample):
        try:
            # Get payload as string or bytes
            if hasattr(sample.payload, 'as_string'):
                base64_str = sample.payload.as_string()
            else:
                base64_str = sample.payload.decode('utf-8')
            
            # Decode from base64
            try:
                img_bytes = base64.b64decode(base64_str)
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
                
                # Add to queue instead of displaying directly
                # Use non-blocking put with timeout
                try:
                    image_queue.put(img, timeout=0.1)
                except queue.Full:
                    # Queue is full, skip this frame
                    pass
                    
            except Exception as e:
                print(f"OpenCV error during imdecode: {e}")
                
        except Exception as e:
            print(f"Error processing image: {e}")
    
    # Configuration for subscriber
    config = zenoh.Config()
    
    # Connect to the publisher's IP address
    config.insert_json5(zenoh.config.CONNECT_KEY, '["tcp/100.117.122.95:7447"]')
    
    config.insert_json5("scouting/multicast/enabled", "true")

    print("Initializing Zenoh subscriber...")
    session = zenoh.open(config)
    
    # Subscribe to the camera frame topic
    key = "carla/frame"
    subscriber = session.declare_subscriber(key, message_handler)
    
    print(f"Subscribed to '{key}'")
    print("Waiting for images... (Press CTRL+C to exit)")
    
    try:
        # Keep the program running to receive messages
        while display_active:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nSubscriber stopped by user")
    finally:
        # Clean shutdown
        display_active = False
        display_thread.join(timeout=1.0)
        session.close()
        print("Zenoh session closed")

if __name__ == "__main__":
    main()