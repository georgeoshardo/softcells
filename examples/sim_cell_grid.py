#!/usr/bin/env python3
"""
Example: Grid of Cells Simulation

This example creates a simulation with 20 cells arranged on a regular grid
with some noise added to their centers. It demonstrates the use of add_cell_shape
function to create biological cell-like structures.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path so we can import softcells
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from softcells.simulation import PhysicsEngine
from softcells.visualization import SimulationVisualizer
from softcells.config import (
    DEFAULT_CELL_RADIUS, 
    DEFAULT_CELL_NUM_POINTS, 
    INTERFACE_POINT_MASS,
    INTERFACE_CELL_SPRING_STIFFNESS,
    INTERFACE_CELL_SPRING_DAMPING,
    GLOBAL_PRESSURE_AMOUNT,
    INTERFACE_YELLOW_COLOR,
    INTERFACE_RED_COLOR,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT
)

def create_cell_grid(physics_engine, num_cells=20, noise_factor=0.3, margin=100):
    """
    Create a grid of cells with noise on their positions.
    Grid spacing is automatically calculated based on window dimensions.
    
    Args:
        physics_engine: The physics engine to add cells to
        num_cells: Number of cells to create (default 20)
        noise_factor: Amount of noise as fraction of grid_spacing (0-1)
        margin: Margin from window edges (pixels)
    """
    # Calculate grid dimensions (trying to make it roughly square)
    grid_cols = int(np.ceil(np.sqrt(num_cells)))
    grid_rows = int(np.ceil(num_cells / grid_cols))
    
    # Calculate available space and grid spacing
    available_width = DEFAULT_WIDTH - 2 * margin
    available_height = DEFAULT_HEIGHT - 2 * margin
    
    grid_spacing_x = available_width / max(1, grid_cols - 1) if grid_cols > 1 else 0
    grid_spacing_y = available_height / max(1, grid_rows - 1) if grid_rows > 1 else 0
    
    # Use the smaller spacing to maintain aspect ratio
    grid_spacing = min(grid_spacing_x, grid_spacing_y)
    
    # Calculate starting position to center the grid
    total_grid_width = (grid_cols - 1) * grid_spacing if grid_cols > 1 else 0
    total_grid_height = (grid_rows - 1) * grid_spacing if grid_rows > 1 else 0
    start_x = (DEFAULT_WIDTH - total_grid_width) / 2
    start_y = (DEFAULT_HEIGHT - total_grid_height) / 2
    
    print(f"Creating {num_cells} cells in a {grid_rows}x{grid_cols} grid")
    print(f"Window size: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
    print(f"Grid spacing: {grid_spacing:.1f}, Noise factor: {noise_factor}")
    print(f"Grid starts at ({start_x:.1f}, {start_y:.1f})")
    
    cell_count = 0
    cells_created = []
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            if cell_count >= num_cells:
                break
                
            # Calculate base position
            base_x = start_x + col * grid_spacing
            base_y = start_y + row * grid_spacing
            
            # Add noise to position
            noise_x = (np.random.random() - 0.5) * grid_spacing * noise_factor
            noise_y = (np.random.random() - 0.5) * grid_spacing * noise_factor
            
            cell_x = base_x + noise_x
            cell_y = base_y + noise_y
            
            print(f"Creating cell {cell_count + 1} at ({cell_x:.1f}, {cell_y:.1f})")
            
            # Create the cell using add_cell_shape
            membrane, nucleus = physics_engine.add_cell_shape(
                cell_x, cell_y, 
                radius=DEFAULT_CELL_RADIUS,
                num_points=DEFAULT_CELL_NUM_POINTS,
                point_mass=INTERFACE_POINT_MASS,
                pressure=GLOBAL_PRESSURE_AMOUNT,
                spring_stiffness=INTERFACE_CELL_SPRING_STIFFNESS,
                spring_damping=INTERFACE_CELL_SPRING_DAMPING
            )
            
            # Set colors for membrane and nucleus
            membrane.set_color(INTERFACE_YELLOW_COLOR)  # Yellow membrane
            nucleus.set_color(INTERFACE_RED_COLOR)      # Red nucleus
            
            cells_created.append((membrane, nucleus))
            cell_count += 1
            
        if cell_count >= num_cells:
            break
    
    print(f"Successfully created {len(cells_created)} cells")
    return cells_created

def main():
    """Main simulation function."""
    # Create physics engine
    
    # Create visualization
    visualizer = SimulationVisualizer()

    create_cell_grid(visualizer.physics_engine, num_cells=20, noise_factor=0.6)
    
    print("\nStarting simulation...")
    print("Controls:")
    print("  TAB - Pause/Resume")
    print("  PERIOD - Step one frame when paused")
    print("  R - Reset simulation")
    print("  ESC - Exit")
    print("  V - Add new cell at mouse position")
    print("  C - Add new circle at mouse position")
    print("  SPACE - Add point at mouse position")
    print("  P - Toggle pressure physics")
    print("  S - Toggle spring physics")
    print("  +/- - Adjust pressure")
    print("  Q/A - Adjust spring stiffness")
    print("  W/Z - Adjust spring damping")
    
    # Run the simulation
    visualizer.run()

if __name__ == "__main__":
    main()
