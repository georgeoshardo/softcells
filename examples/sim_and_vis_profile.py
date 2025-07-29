#!/usr/bin/env python3
"""
SoftCells - Soft Body Physics Simulation (Decoupled Version)

Main entry point for the soft body physics simulation using the new decoupled architecture.
Run this script to start the interactive simulation.

Usage:
    python main_decoupled.py

Controls:
    TAB - Pause/Unpause simulation
    PERIOD - Step one frame (when paused)
    R - Reset simulation
    SPACE - Add point at mouse
    C - Add circle at mouse
    P - Toggle pressure physics
    +/- - Increase/Decrease pressure
    S - Toggle spring physics
    Q/A - Increase/Decrease spring stiffness
    W/Z - Increase/Decrease spring damping
    D - Toggle drag physics
    E/X - Increase/Decrease drag coefficient
    T - Toggle drag type (linear/quadratic)
    H - Toggle instructions display
    I - Toggle physics info display
    L - Toggle trails display
    ESC - Exit
"""

from softcells import SimulationVisualizer

import numpy as np
np.random.seed(40)

from softcells.config import GLOBAL_PRESSURE_AMOUNT

def main():
    import cProfile 
    import pstats

    with cProfile.Profile() as pr:
        """Main entry point for the decoupled simulation."""
        try:
            # Create and run the visualization (which contains the physics engine)
            visualizer = SimulationVisualizer()
            visualizer.physics_engine.enable_collision_multiprocessing(True, num_processes=14)

            # Add circles every 150 units in x and y
            for x in range(168, visualizer.width, 200):
                for y in range(120, visualizer.height, 170):
                    visualizer.physics_engine.add_circle_shape(
                        x+np.random.randint(-20,20), y+np.random.randint(-20,20), 50,
                        num_points=30,
                        point_mass=1.0,
                        pressure=GLOBAL_PRESSURE_AMOUNT*2,
                        spring_stiffness=1150.0,
                        spring_damping=10.0
                    )
            visualizer.run()
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.dump_stats(filename="profile.prof")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
    



if __name__ == "__main__":
    main()
