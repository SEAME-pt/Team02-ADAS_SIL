import zenoh
import time
import sys

def main():
    # Create a message handler function
    def message_handler(sample):
        try:
            # Try to get message payload as a string
            if hasattr(sample.payload, 'as_string'):
                message = sample.payload.as_string()
            else:
                # Fall back to decoding bytes
                message = sample.payload.decode('utf-8')
                
            print(f"Received: {message}")
        except Exception as e:
            print(f"Error processing message: {e}")
            print(f"Payload type: {type(sample.payload)}")
    
    # Configuration for subscriber
    config = zenoh.Config()
    
    # Connect to the publisher's IP address
    config.insert_json5(zenoh.config.CONNECT_KEY, '["tcp/100.117.122.95:7447"]')
    
    # Enable peer discovery
    config.insert_json5("scouting/multicast/enabled", "true")
    
    # # Optional: Enable debug logging
    # config.insert_json5("logging/level", '"debug"')
    
    print("Initializing Zenoh subscriber...")
    session = zenoh.open(config)
    
    # Subscribe to the test/message topic
    key = "Carla/frame/test"
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