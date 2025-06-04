import zenoh
import time
import sys

def main():
    # Create a message handler function
    def message_handler(sample):
        
        message = sample.payload.to_string()        
        print(f"Received: {message}")

    
    # Configuration for subscriber
    config = zenoh.Config()
    
    # # Connect to the publisher's IP address
    config.insert_json5("connect/endpoints", '["tcp/100.117.122.95:7447"]')
    
    # # Enable peer discovery
    # config.insert_json5("scouting/multicast/enabled", "true")
    
    print("Initializing Zenoh subscriber...")
    session = zenoh.open(config)
    
    # Subscribe to the test/message topic
    key = "carla/frame"
    subscriber = session.declare_subscriber(key, message_handler)
    
    print(f"Subscribed to '{key}'")
    print("Waiting for messages... (Press CTRL+C to exit)")
    
    try:
        # Keep the program running to receive messages
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nSubscriber stopped by user")
    finally:
        # Clean shutdown
        session.close()
        print("Zenoh session closed")

if __name__ == "__main__":
    main()