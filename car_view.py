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
            # Get payload as string or bytes
            if hasattr(sample.payload, 'as_string'):
                base64_str = sample.payload.as_string()
            else:
                base64_str = sample.payload.decode('utf-8')
            
            print(f"Received data of length: {len(base64_str)}")
            print(f"First 20 chars: {base64_str[:20]}")
            
            # Decode from base64
            import base64
            try:
                img_bytes = base64.b64decode(base64_str)
                print(f"Decoded to {len(img_bytes)} bytes of image data")
            except Exception as e:
                print(f"Base64 decoding failed: {e}")
                return
            
            # Convert to numpy and decode image
            import numpy as np
            img_data = np.frombuffer(img_bytes, dtype=np.uint8)
            
            # Try decoding with explicit error handling
            import cv2
            try:
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None:
                    print("cv2.imdecode returned None")
                    return
                
                print(f"Successfully decoded image: {img.shape}")
                cv2.imshow("Received Image", img)
                cv2.waitKey(1)
            except Exception as e:
                print(f"OpenCV error during imdecode: {e}")
                
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()

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