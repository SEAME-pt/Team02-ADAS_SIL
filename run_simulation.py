import glob
import os
import sys
import random
import time
import zenoh
import numpy as np
import cv2
import math

try:
    sys.path.append(glob.glob('/home/luis_t2/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def camera_setup(ego_vehicle, bp_library, world):
    camera_bp = bp_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '105')

    camera_init_trans = carla.Transform(
        carla.Location(2, 0.0, 1.5),
        carla.Rotation(-15, 0, 0)
    )

    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
    return camera

def setup_carla_environment(num_traffic_vehicles = 150):
    client = carla.Client('127.0.0.1', 2000)
    client.load_world("Town04")
    client.set_timeout(60.0)

    world = client.get_world()
    world.unload_map_layer(carla.MapLayer.Buildings)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    bp_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    traffic_manager = client.get_trafficmanager()
    traffic_manager.global_percentage_speed_difference(20)
    traffic_manager.set_synchronous_mode(True)

    main_spawn_index = 100
    main_spawn_point = spawn_points[main_spawn_index]
    
    vehicle_bp = bp_library.find('vehicle.tesla.model3')
    
    if vehicle_bp.has_attribute('color'):
        vehicle_bp.set_attribute('color', '255,0,0')
    
    ego_vehicle = world.try_spawn_actor(vehicle_bp, main_spawn_point)
    if ego_vehicle is not None:
        print(f"Spawned main vehicle at fixed location: {main_spawn_point.location}")
    else:
        raise RuntimeError("Failed to spawn main vehicle at fixed location")
    
    remaining_spawn_points = [sp for i, sp in enumerate(spawn_points) if i != main_spawn_index]

    vehicle_bps = bp_library.filter('vehicle.*')
    
    traffic_vehicles = []
    
    for i in range(min(num_traffic_vehicles, len(remaining_spawn_points))):
        traffic_bp = random.choice(vehicle_bps)
        
        traffic_vehicle = world.try_spawn_actor(traffic_bp, remaining_spawn_points[i])
        if traffic_vehicle:
            traffic_vehicle.set_autopilot(True)
            traffic_manager.set_desired_speed(traffic_vehicle, 15)
            traffic_vehicles.append(traffic_vehicle)
    
    print(f"Successfully spawned {len(traffic_vehicles)} traffic vehicles")

    spectator = world.get_spectator()
    transform = spectator.get_transform()
    location = transform.location
    rotation = transform.rotation

    camera = camera_setup(ego_vehicle, bp_library, world)
    return client, world, ego_vehicle, camera

def main():
    # Run your simulation
    client, world, vehicle, camera = setup_carla_environment(num_traffic_vehicles=150)

    config = zenoh.Config()
    
    # Configure to listen on all interfaces
    config.insert_json5("listen/endpoints", '["udp/0.0.0.0:7447"]')
    
    # Enable peer discovery
    config.insert_json5("scouting/multicast/enabled", "true")
    
    # Create session with network configuration
    session = zenoh.open(config)
    key_frame = 'carla/frame'
    key_speed = 'carla/speed'
    pub_frame = session.declare_publisher(key_frame)
    pub_speed = session.declare_publisher(key_speed)

    # Vehicle control subscriber
    control_steering = 'Vehicle/1/Chassis/SteeringWheel/Angle'
    control_throttle = 'Vehicle/1/Powertrain/ElectricMotor/Speed'

    current_control = carla.VehicleControl()
    current_control.throttle = 0.0
    current_control.steer = 0.0
    current_control.brake = 0.0
    current_control.reverse = False
    current_control.hand_brake = False
    
    # Define control callback
    def control_steering_callback(sample):
        try:
            control_steering_str = sample.payload.to_string()
            control_steering_str = control_steering_str.strip('\x00')
                
            angle_degrees = float(control_steering_str)

            normalized_steering = (angle_degrees - 90) / 90.0
            
            current_control.steer = max(-1.0, min(1.0, normalized_steering))
                
        except Exception as e:
            print(f"Error processing control: {e}")

    def control_throttle_callback(sample):
        try:
            control_throttle_str = sample.payload.to_string()
            control_throttle_str = control_throttle_str.strip('\x00')

            # Convert string to float
            speed_value = float(control_throttle_str)
            
            min_throttle = 0.2
            if speed_value > 0:
                # Forward
                current_control.throttle = min_throttle + (speed_value / 100.0) * (1.0 - min_throttle)
                current_control.brake = 0.0
                current_control.reverse = False
            elif speed_value < 0:
                # Reverse
                current_control.throttle = min_throttle + (abs(speed_value) / 100.0) * (1.0 - min_throttle)
                current_control.brake = 0.0
                current_control.reverse = True
            else:
                # Stop
                current_control.throttle = 0.0
                current_control.brake = 10.0
                
        except Exception as e:
            print(f"Error processing control: {e}")
    
    # Register control subscriber
    control_sub_s = session.declare_subscriber(control_steering, control_steering_callback)
    control_sub_t = session.declare_subscriber(control_throttle, control_throttle_callback)
    
    def get_speed():
        """
        Returns the speed of the vehicle in km/h
        """
        velocity = vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        return speed_kmh

    def camera_callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded_image = cv2.imencode('.jpg', array, encode_param)
        
        image_bytes = encoded_image.tobytes()

        speed_str = str(get_speed())
        pub_frame.put(image_bytes)
        pub_speed.put(speed_str)
        message = f"Message send from publisher at {time.strftime('%H:%M:%S')}"
        print(message)
    
    camera.listen(camera_callback)

    vehicle.set_autopilot(False)

    try:
        print("Simulation running. Press Ctrl+C to exit.")
        
        while True:
            vehicle.apply_control(current_control)
            world.tick()
            
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        if 'camera' in locals() and camera:
            camera.stop()
            camera.destroy()
        if 'vehicle' in locals() and vehicle:
            vehicle.set_autopilot(False)
            vehicle.destroy()
        if 'world' in locals() and world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        
        session.close()
        print("Simulation ended.")

if __name__ == "__main__":
    main()