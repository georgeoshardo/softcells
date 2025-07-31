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
import pickle

import numpy as np
def main():
    """Main entry point for the decoupled simulation."""
    try:
        # Create and run the visualization (which contains the physics engine)
        visualizer = SimulationVisualizer()
        # for x in range(168, visualizer.width, 400):
        #     for y in range(120, visualizer.height, 400  ):
        #         circle1, circle2 = visualizer.physics_engine.add_cell_shape(
        #                         x+np.random.randint(-10,20), y+np.random.randint(-10,20), 20, 
        #                         num_points=50, 
        #                         point_mass=1.0, 
        #                         pressure=3500,
        #                         spring_stiffness=2150.0, 
        #                         spring_damping=20.0
        #                     )
        #         circle1.set_color((255, 255, 100))  # Yellow
        #         circle2.set_color((255, 100, 100))  # Red

        visualizer.run()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        with open("simulation_log.pkl", "wb") as f:
            pickle.dump(visualizer.physics_engine.simulation_log, f)
            print("Log successfully saved to simulation_log.pkl")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
