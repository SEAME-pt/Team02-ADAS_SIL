import cv2
import numpy as np
import time
import sys

def test_opencv_gui():
    print("Starting OpenCV GUI test...")
    
    try:
        # Create a simple black image
        print("Creating test image...")
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Add some text and shapes to verify rendering
        print("Adding text and shapes...")
        cv2.putText(img, "OpenCV GUI Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), 2)
        cv2.circle(img, (400, 200), 50, (255, 0, 0), -1)
        
        print("Attempting to create window...")
        # Try to create a window - this is where it typically fails on macOS
        cv2.namedWindow("OpenCV Test", cv2.WINDOW_NORMAL)
        print("Window created successfully!")
        
        print("Attempting to resize window...")
        cv2.resizeWindow("OpenCV Test", 800, 600)
        print("Window resized successfully!")
        
        print("Attempting to display image...")
        cv2.imshow("OpenCV Test", img)
        print("Image displayed successfully!")
        
        print("Entering waitKey loop...")
        print("Press 'q' to exit")
        while True:
            key = cv2.waitKey(100)
            if key == ord('q') or key == 27:  # q or ESC
                break
        
        print("Destroying windows...")
        cv2.destroyAllWindows()
        print("Windows destroyed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Print OpenCV version for debugging
    print(f"OpenCV version: {cv2.__version__}")
    
    # Check if we have GUI support
    print(f"GUI backend: {cv2.getBuildInformation()}")
    
    # Run the test
    test_opencv_gui()