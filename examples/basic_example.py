#!/usr/bin/env python3
"""
Basic Example - SoftCells Physics Simulation

This example demonstrates how to create a simple simulation with custom shapes
and physics parameters.
"""

from softcells import PhysicsSimulation, CircleShape, PointMass
from softcells.shapes import Shape


def create_custom_simulation():
    """Create a custom simulation with different soft body configurations."""
    
    # Create simulation with custom dimensions
    sim = PhysicsSimulation(width=1200, height=1200)
    

    
    return sim


def main():
    """Run the custom simulation example."""
    print("Starting SoftCells Basic Example...")
    print("Use the same controls as the main simulation.")
    print("Press ESC to exit.")
    
    try:
        simulation = create_custom_simulation()
        simulation.run()
    except KeyboardInterrupt:
        print("\nExample interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main() 