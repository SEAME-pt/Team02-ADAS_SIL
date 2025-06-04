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
            if isinstance(sample.payload, bytes):
                base64_str = sample.payload.decode('utf-8')  # Convert bytes to string
                print("bytes")
            elif hasattr(sample.payload, 'to_string'):
                base64_str = sample.payload.to_string()
                print("string")
            else:
                base64_str = str(sample.payload)
                print("else")
            
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

    config = zenoh.Config()
    
    # Connect to the publisher's IP address
    config.insert_json5(zenoh.config.CONNECT_KEY, '["tcp/100.117.122.95:7447"]')
    
    # Enable peer discovery
    config.insert_json5("scouting/multicast/enabled", "true")
    
    session = zenoh.open(config)
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