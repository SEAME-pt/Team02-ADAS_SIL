import glob
import os
import sys
import random
import time
import zenoh
import numpy as np
import cv2

def main():
    # This function will actually decode and display the received image
    def listener(sample):
        try:
            # Now it's safe to convert to string
            base64_str = sample.payload.to_string()
            
            # Decode from base64
            import base64
            img_bytes = base64.b64decode(base64_str)
            
            # Convert to numpy and decode image
            img_data = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            
            if img is not None:
                cv2.imshow("CARLA Camera Feed", img)
                cv2.waitKey(1)
            else:
                print("Failed to decode image")
        except Exception as e:
            print(f"Error processing image: {e}")

    # Open Zenoh session
    session = zenoh.open(zenoh.Config())
    key = 'carla/frame'
    sub = session.declare_subscriber(key, listener)

    print("Subscriber running. Press Ctrl+C to exit.")
    
    try:
        # Better wait mechanism that doesn't flood the console
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nSubscriber stopped by user")
    finally:
        # Clean up
        cv2.destroyAllWindows()
        session.close()
        print("Session closed")

if __name__ == "__main__":
    main()