#!/usr/bin/env python
"""Main simulation entry point."""

import time
import argparse
from adas_sil.simulation.carla_setup import setup_carla_environment
from adas_sil.perception.Detection import Detection
from adas_sil.control.controller2 import Controller

def main():
    parser = argparse.ArgumentParser(description="Run ADAS simulation")
    parser.add_argument("--scenario", default="Town04", help="Scenario to run")
    args = parser.parse_args()
    
    # Run your simulation
    client, world, vehicle = setup_carla_environment(args.scenario)
    
    controller = Controller(vehicle, world)
    # Keep the simulation running until interrupted
    try:
        print("Simulation running. Press Ctrl+C to exit.")
        controller.control_car()
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        # Clean up resources
        if vehicle:
            vehicle.destroy()
        if controller:
            controller.cleanup()
        if world is not None:
            world.destroy_sensors() 
        print("I am still standing")


if __name__ == "__main__":
    main()