"""
Circle shape implementation for soft body physics.
"""

import math

from ..core import PointMass
from ..config import (
    GLOBAL_PRESSURE_AMOUNT, DEFAULT_CIRCLE_POINTS, DEFAULT_POINT_MASS,
    DEFAULT_CIRCLE_SPRING_STIFFNESS, DEFAULT_CIRCLE_SPRING_DAMPING,
    DEFAULT_GLOBAL_DRAG_COEFFICIENT, DEFAULT_DRAG_TYPE, CIRCLE_COLOR,
    DEFAULT_WIDTH, DEFAULT_HEIGHT
)
from .base_shape import Shape


class CircleShape(Shape):
    """
    A shape that arranges point masses in a circle pattern.
    Uses pressure physics to maintain circular form and springs for structural integrity.
    """
    
    def __init__(self, center_x, center_y, radius, num_points=DEFAULT_CIRCLE_POINTS, 
                 point_mass=DEFAULT_POINT_MASS, pressure=GLOBAL_PRESSURE_AMOUNT, 
                 spring_stiffness=DEFAULT_CIRCLE_SPRING_STIFFNESS, 
                 spring_damping=DEFAULT_CIRCLE_SPRING_DAMPING, 
                 drag_coefficient=DEFAULT_GLOBAL_DRAG_COEFFICIENT, 
                 drag_type=DEFAULT_DRAG_TYPE):
        """
        Create a circular shape with evenly distributed point masses.
        
        Args:
            center_x (float): X coordinate of circle center
            center_y (float): Y coordinate of circle center
            radius (float): Radius of the circle
            num_points (int): Number of point masses around the circle
            point_mass (float): Mass of each point
            pressure (float): Internal gas pressure (auto-calculated if None)
            spring_stiffness (float): Stiffness of springs connecting points
            spring_damping (float): Damping of springs connecting points
            drag_coefficient (float): Viscous drag coefficient for all points
            drag_type (str): Type of drag - "linear" or "quadratic"
        """
        super().__init__()
        
        # Generate points around the circle
        for i in range(num_points):
            angle = (2 * math.pi * i) / num_points
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            point = PointMass(x, y, point_mass, drag_coefficient, 
                              world_height=DEFAULT_HEIGHT, world_width=DEFAULT_WIDTH)
            self.add_point(point)
        
        # Set spring properties and create springs
        self.set_spring_properties(spring_stiffness, spring_damping)
        self.create_springs()
        
        # Set drag properties
        self.set_drag_properties(drag_coefficient, drag_type)
        
        # Set pressure
        self.set_pressure(pressure)
        
        # Set a distinct color for circle shapes
        self.set_color(CIRCLE_COLOR) 