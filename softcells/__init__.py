"""
SoftCells - A soft body physics simulation library.

This package provides a comprehensive soft body physics simulation system
with features including:
- Verlet integration for stable physics
- Spring-mass systems
- Pressure-based volume preservation
- Collision detection and resolution
- Real-time visualization with Pygame

Main Components:
- core: Basic physics components (PointMass, Spring)
- shapes: Shape implementations (Shape, CircleShape)
- simulation: Simulation engine and collision handling
- utils: Utility functions for geometry and math
- config: Configuration constants and settings
"""

from .core import PointMass, Spring
from .shapes import Shape, CircleShape
from .simulation import PhysicsSimulation, CollisionHandler
from .config import *

__version__ = "1.0.0"
__author__ = "SoftCells Development Team"

__all__ = [
    'PointMass', 'Spring', 'Shape', 'CircleShape', 
    'PhysicsSimulation', 'CollisionHandler'
] 