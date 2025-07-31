"""
Configuration settings and constants for the soft body simulation.
"""

# Global physics constants
GLOBAL_PRESSURE_AMOUNT = 3500

# Simulation settings
DEFAULT_WIDTH = 1536 
DEFAULT_HEIGHT = 1024  
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