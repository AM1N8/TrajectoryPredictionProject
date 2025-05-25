#!/usr/bin/env python

import carla
import argparse
import time
import os

def get_all_available_towns(client):
    """
    Get a list of all available town/map names from the CARLA server.
    
    Args:
        client: CARLA client object
        
    Returns:
        List of available town names
    """
    # Get all available maps
    available_maps = client.get_available_maps()
    
    # Extract just the town names from the full paths
    town_names = []
    for map_path in available_maps:
        # Maps are usually named like '/Game/Carla/Maps/Town01'
        # Extract just the town name (e.g., 'Town01')
        town_name = os.path.basename(map_path)
        town_names.append(town_name)
    
    return town_names

def load_town(client, town_name):
    """
    Load a specific town in the CARLA server.
    
    Args:
        client: CARLA client object
        town_name: Name of the town to load
        
    Returns:
        World object of the loaded town
    """
    print(f"Loading town: {town_name}")
    
    # Find the full map path that matches the town name
    available_maps = client.get_available_maps()
    map_path = None
    
    for available_map in available_maps:
        if town_name in available_map:
            map_path = available_map
            break
    
    if map_path is None:
        raise ValueError(f"Town '{town_name}' not found")
    
    # Load the world with the chosen map
    world = client.load_world(map_path)
    
    # Wait for the world to be ready
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    print(f"Successfully loaded {town_name}")
    return world

def main():
    parser = argparse.ArgumentParser(description='CARLA Town Selector')
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', default=2000, type=int, help='CARLA server port')
    parser.add_argument('--town', help='Specific town to load (optional)', default=None)
    parser.add_argument('--timeout', default=20.0, type=float, help='CARLA client connection timeout')
    
    args = parser.parse_args()
    
    # Connect to the CARLA server
    try:
        print(f"Connecting to CARLA server at {args.host}:{args.port}")
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        
        # Get the current world
        world = client.get_world()
        print(f"Connected to CARLA. Current map: {world.get_map().name}")
        
        # Get all available towns
        towns = get_all_available_towns(client)
        print("\nAvailable towns:")
        for i, town in enumerate(towns):
            print(f"{i+1}. {town}")
        
        if args.town:
            # Load the specified town from command line
            world = load_town(client, args.town)
        else:
            # Let the user select a town
            while True:
                try:
                    choice = input("\nEnter town number to load (or 'q' to quit): ")
                    
                    if choice.lower() == 'q':
                        print("Exiting town selector")
                        break
                    
                    town_index = int(choice) - 1
                    if 0 <= town_index < len(towns):
                        world = load_town(client, towns[town_index])
                        
                        # Ask if user wants to load another town
                        another = input("\nDo you want to load another town? (y/n): ")
                        if another.lower() != 'y':
                            break
                    else:
                        print(f"Invalid selection. Please enter a number between 1 and {len(towns)}")
                except ValueError:
                    print("Please enter a valid number")
                except Exception as e:
                    print(f"Error: {e}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    print("Town selector closed")

if __name__ == '__main__':
    main()