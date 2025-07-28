#!/usr/bin/env python3
"""
SoftCells - Soft Body Physics Simulation

Main entry point for the soft body physics simulation.
Run this script to start the interactive simulation.

Usage:
    python main.py

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
    ESC - Exit
"""

from softcells import PhysicsSimulation


def main():
    """Main entry point for the simulation."""
    try:
        # Create and run the simulation
        simulation = PhysicsSimulation()
        simulation.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main() 