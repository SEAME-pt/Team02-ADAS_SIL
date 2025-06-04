import zenoh
import time
import sys
import base64
import numpy as np
import cv2
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
            
            # Add to queue for main thread to display
            try:
                # Use put_nowait to avoid blocking the Zenoh callback
                image_queue.put(img, block=False)
                print("Added image to display queue")
            except queue.Full:
                print("Queue full, dropping oldest frame")
                # If queue is full, remove oldest item and add new one
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
    cv2.waitKey(100)  # Give it time to render initially
    
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
    
    # Main display loop - everything happens in the main thread
    frame_counter = 0
    try:
        print("Entering main display loop")
        while display_active:
            # Check for new image in queue
            try:
                # Non-blocking get with short timeout
                img = image_queue.get(block=True, timeout=0.1)
                
                # Got a new image, display it
                frame_counter += 1
                print(f"Displaying frame #{frame_counter}")
                
                # Add frame counter to image
                display_img = img.copy()  # Make a copy to avoid modifying original
                cv2.putText(display_img, f"Frame: {frame_counter}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the image
                cv2.imshow(display_window_name, display_img)
                
                # Save occasional frames to verify content
                if frame_counter % 30 == 0:
                    cv2.imwrite(f"frame_{frame_counter}.jpg", display_img)
                    print(f"Saved frame_{frame_counter}.jpg")
                
                # Mark as done
                image_queue.task_done()
            except queue.Empty:
                # No new image, just update the window
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
        # Clean shutdown
        display_active = False
        cv2.destroyAllWindows()
        session.close()
        print("Clean shutdown complete")

if __name__ == "__main__":
    main()