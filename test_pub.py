import zenoh
import time
import sys

def main():
    # Configuration for publisher
    config = zenoh.Config()
    
    # # Configure to listen on all interfaces
    config.insert_json5("listen/endpoints", '["tcp/100.117.122.95:7447"]')
    
    # # Enable peer discovery
    # config.insert_json5("scouting/multicast/enabled", "true")
    
    print("Initializing Zenoh publisher...")
    session = zenoh.open(config)
    
    # Create a publisher on "test/message" topic
    key = "carla/frame"
    publisher = session.declare_publisher(key)
    
    # Counter for messages
    count = 0
    
    print(f"Publishing messages on '{key}'...")
    print("Press CTRL+C to exit")
    
    try:
        while True:
            # Create message with incrementing counter
            message = f"Test message #{count} from publisher at {time.strftime('%H:%M:%S')}"
            
            # Publish the message
            publisher.put(message)
            print(f"Published: {message}")
            
            # Increment counter
            count += 1
            
            # Wait a second before next message
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nPublisher stopped by user")
    finally:
        # Clean shutdown
        session.close()
        print("Zenoh session closed")

if __name__ == "__main__":
    main()