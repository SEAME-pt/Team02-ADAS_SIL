import sys
import random

carla_egg = "C:/Users/manue/Documents/SEA_ME/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg"
sys.path.append(carla_egg)

import carla


def setup_carla_environment(town='Town04'):

    client = carla.Client('127.0.0.1', 2000)
    world = client.get_world()
    if world.get_map().name != town:
        print(f"Loading {town}...")
        world = client.load_world(town)
    

    bp_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    vehicle_bp = bp_library.filter('vehicle.*')[0]
    ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if ego_vehicle is not None:
        print(f"Spawned {vehicle_bp.id}")
    else:
        raise RuntimeError("Failed to spawn vehicle")

    spectator = world.get_spectator()
    # Get the location and rotation of the spectator through its transform
    transform = spectator.get_transform()
    location = transform.location
    rotation = transform.rotation

    # camera = camera_setup(ego_vehicle, bp_library, world)
    return client, world, ego_vehicle #camera


def main():
    _, _, _, camera = setup_carla_environment()
    return camera

if __name__ == "__main__":
    main()