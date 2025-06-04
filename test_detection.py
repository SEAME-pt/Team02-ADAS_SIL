import glob
import os
import sys
import random
import time
import pygame
from LaneDetection import LaneDetection
from ObjectDetection import ObjectDetection

try:
    sys.path.append(glob.glob('/home/luis_t2/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def camera_setup(ego_vehicle, bp_library, world):
    # We create the camera through a blueprint that defines its properties
    camera_bp = bp_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '105')

    # Create a transform to place the camera on top of the vehicle
    camera_init_trans = carla.Transform(
        carla.Location(2, 0.0, 1.5),  # Position inside car at driver's head position
        carla.Rotation(-15, 0, 0)  # Look slightly downward
    )

    # We spawn the camera and attach it to our ego vehicle
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
    return camera

def setup_carla_environment():
    client = carla.Client('127.0.0.1', 2000)
    client.load_world("Town04")
    client.set_timeout(60.0)

    traffic_manager = client.get_trafficmanager()
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(True)
    
    bp_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    vehicle_bp = bp_library.filter('vehicle.*')[0]
    ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if ego_vehicle is not None:
        print(f"Spawned {vehicle_bp.id}")
    else:
        raise RuntimeError("Failed to spawn vehicle")

    spectator = world.get_spectator()
    transform = spectator.get_transform()
    location = transform.location
    rotation = transform.rotation

    camera = camera_setup(ego_vehicle, bp_library, world)
    return client, world, ego_vehicle, camera

def main():
    # Initialize pygame
    pygame.init()
    pygame.font.init()
    
    # Set up display
    display = pygame.display.set_mode(
            (800, 600),  # Match camera resolution
            pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Camera View")

    # Run your simulation
    client, world, vehicle, camera = setup_carla_environment()

    # Enable autopilot
    vehicle.set_autopilot(True)
    
    # Set a more aggressive driving behavior
    tm = client.get_trafficmanager()
    tm.global_percentage_speed_difference(-20)  # Drive faster

    tm.ignore_lights_percentage(vehicle, 100)

    # Initialize detector with display reference
    # detector = LaneDetection(display)
    detector = ObjectDetection(display)
    detector.load_model()
    
    # Set up camera listener with the callback method
    camera.listen(detector.camera_callback)

    try:
        print("Simulation running. Press Ctrl+C to exit.")
        clock = pygame.time.Clock()
        
        while True:
            # Tick the world simulation
            world.tick()
            
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
            
            # Update display in the main thread
            detector.update_display()
            
            clock.tick(20)  # Limit to 20 FPS to match simulation
            
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        # Clean up resources
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
        
        pygame.quit()
        print("Simulation ended.")

if __name__ == "__main__":
    main()