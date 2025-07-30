"""
Configuration settings and constants for the soft body simulation.
"""

# Global physics constants
GLOBAL_PRESSURE_AMOUNT = 3500

# Simulation settings
DEFAULT_WIDTH = 1500
DEFAULT_HEIGHT = 1000
DEFAULT_FPS = 120
DEFAULT_DT = 1.0 / 100.0  # Time step for physics calculations

# Physics defaults
DEFAULT_GRAVITY = 0.0
DEFAULT_GLOBAL_DRAG_COEFFICIENT = 5.5
DEFAULT_DRAG_TYPE = "linear"

# Spring physics defaults
DEFAULT_SPRING_STIFFNESS = 100.0
DEFAULT_SPRING_DAMPING = 10.0

# Shape defaults
DEFAULT_CIRCLE_POINTS = 8
DEFAULT_POINT_MASS = 1.0
DEFAULT_CIRCLE_SPRING_STIFFNESS = 150.0
DEFAULT_CIRCLE_SPRING_DAMPING = 10.0

# Collision resolution settings
COLLISION_SLOP = 0.0
COLLISION_CORRECTION_PERCENT = 0.4
COLLISION_RESTITUTION = 0.3

# Rendering settings
DEFAULT_BACKGROUND_COLOR = (20, 20, 30)
DEFAULT_POINT_COLOR = (100, 150, 255)
DEFAULT_TRAIL_COLOR = (50, 75, 125)
DEFAULT_LINE_WIDTH = 2
MAX_TRAIL_LENGTH = 30

# Shape colors
CIRCLE_COLOR = (150, 255, 150)  # Green
DEFAULT_SHAPE_COLOR = (100, 150, 255)  # Blue 

PERIODIC = True

# Ornstein Uhlenbeck process parameters
OU_TAU = 10  # Time constant for OU process

# Cell creation parameters (preserving current behavior)
DEFAULT_CELL_SPRING_STIFFNESS = 1150.0  # Default stiffness for add_cell_shape function
DEFAULT_CELL_SPRING_DAMPING = 10.0      # Default damping for add_cell_shape function
DEFAULT_CELL_NUM_POINTS = 50            # Number of points for cell shapes
DEFAULT_CELL_RADIUS = 20                # Default radius for cell creation

# Cell geometry scaling factors
CELL_MEMBRANE_RADIUS_FACTOR = 2.2       # Membrane size relative to input radius
CELL_NUCLEUS_RADIUS_FACTOR = 1.8        # Nucleus size relative to input radius
CELL_PRESSURE_FACTOR = 1.2              # Pressure multiplier for both membrane and nucleus
CELL_NUCLEUS_STIFFNESS_FACTOR = 1.2     # Nucleus spring stiffness multiplier

# Cell spring connection parameters
CELL_SPRING_CONNECTION_PROBABILITY = 0.3  # Probability of creating nucleus-membrane springs
CELL_SPRING_STIFFNESS_FACTOR = 0.3       # Stiffness factor for nucleus-membrane springs
CELL_SPRING_DAMPING_FACTOR = 0.2         # Damping factor for nucleus-membrane springs
CELL_SPRING_LENGTH_MIN_FACTOR = 0.1      # Minimum length factor for random spring lengths
CELL_SPRING_LENGTH_MAX_FACTOR = 0.3      # Maximum length factor for random spring lengths

# Interface parameters (current hardcoded values in visualization)
INTERFACE_CELL_SPRING_STIFFNESS = 2150.0  # Spring stiffness used in visualization interface
INTERFACE_CELL_SPRING_DAMPING = 20.0      # Spring damping used in visualization interface
INTERFACE_CIRCLE_SPRING_STIFFNESS = 1150.0  # Spring stiffness for regular circles in interface

# Adjustment factors for UI controls
PRESSURE_ADJUSTMENT_FACTOR = 1.2         # Factor for pressure increase/decrease
STIFFNESS_ADJUSTMENT_FACTOR = 1.2        # Factor for stiffness increase/decrease
DAMPING_ADJUSTMENT_FACTOR = 1.2          # Factor for damping increase/decrease
DRAG_ADJUSTMENT_FACTOR = 1.2             # Factor for drag coefficient increase/decrease

# Physics boundary parameters  
BOUNDARY_ENERGY_LOSS_FACTOR = 0.8        # Energy loss factor on boundary bounce
SPRING_STRETCH_THRESHOLD = 1.1           # Threshold for spring stretch visualization

# Default world dimensions
DEFAULT_WORLD_WIDTH = 1000
DEFAULT_WORLD_HEIGHT = 800

# Interface creation parameters
INTERFACE_CIRCLE_RADIUS = 50             # Radius for regular circles created via interface
INTERFACE_LARGE_CIRCLE_RADIUS = 70       # Radius for large circles created via interface
INTERFACE_CIRCLE_NUM_POINTS = 30         # Number of points for interface-created circles
INTERFACE_CELL_NUM_POINTS = 50           # Number of points for interface-created cells
INTERFACE_POINT_MASS = 1.0               # Mass for interface-created points
INTERFACE_CIRCLE_DAMPING = 10.0          # Damping for interface-created circles

# Interface colors (RGB tuples)
INTERFACE_YELLOW_COLOR = (255, 255, 100)  # Yellow color for membranes
INTERFACE_RED_COLOR = (255, 100, 100)     # Red color for nuclei and large circles
INTERFACE_BRIGHT_YELLOW_COLOR = (255, 255, 100)  # Bright yellow for various UI elements
INTERFACE_DRAG_ENABLED_COLOR = (255, 255, 0)     # Color when drag is enabled
INTERFACE_DRAG_DISABLED_COLOR = (100, 100, 100)  # Color when drag is disabled

# Spring visualization parameters
SPRING_STRETCH_INTENSITY_FACTOR = 3.0    # Factor for calculating stretch visualization intensity
SPRING_INTENSITY_MIN = 1.0               # Minimum intensity value for spring visualization