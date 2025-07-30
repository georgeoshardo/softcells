"""
Spring physics implementation for connecting point masses.
"""

import math
from ..config import DEFAULT_SPRING_STIFFNESS, DEFAULT_SPRING_DAMPING


class Spring:
    """
    A spring connects two point masses with elastic and damping forces.
    Implements Hooke's law with velocity damping.
    """
    
    def __init__(self, point1, point2, stiffness=DEFAULT_SPRING_STIFFNESS, damping=DEFAULT_SPRING_DAMPING, rest_length=None):
        """
        Create a spring between two point masses.
        
        Args:
            point1 (PointMass): First point mass
            point2 (PointMass): Second point mass
            stiffness (float): Spring constant (ks)
            damping (float): Damping coefficient (kd)
            rest_length (float): Natural length of spring (calculated if None)
        """
        self.point1 = point1
        self.point2 = point2
        self.stiffness = stiffness
        self.damping = damping
        
        # Calculate rest length if not provided
        if rest_length is None:
            dx = point1.x - point2.x
            dy = point1.y - point2.y
            self.rest_length = math.sqrt(dx * dx + dy * dy)
        else:
            self.rest_length = rest_length
    
    def apply_force(self):
        """
        Apply spring force to both connected point masses.
        Formula from soft body paper: F = (|r1-r2| - rest_length) * ks + (v1-v2)·(r1-r2)/|r1-r2| * kd
        """
        # Get positions and velocities
        x1, y1 = self.point1.x, self.point1.y
        x2, y2 = self.point2.x, self.point2.y
        vx1, vy1 = self.point1.vx, self.point1.vy
        vx2, vy2 = self.point2.vx, self.point2.vy
        
        # Calculate distance and direction
        dx = x1 - x2
        dy = y1 - y2
        current_length = math.sqrt(dx * dx + dy * dy)
        
        # Avoid division by zero
        if current_length < 0.001:
            return
        
        # Normalize direction vector
        nx = dx / current_length
        ny = dy / current_length
        
        # Spring force, no direction (Hooke's law)
        spring_force = (current_length - self.rest_length) * self.stiffness # k(L - L₀)
        
        # Damping force (velocity difference projected onto spring direction)
        vx_rel = vx1 - vx2 # relative velocity x
        vy_rel = vy1 - vy2 # relative velocity y
        velocity_projection = (vx_rel * nx + vy_rel * ny) # v_rel · n̂
        damping_force = velocity_projection * self.damping # c * v_parallel
        
        # Total force magnitude
        total_force = spring_force + damping_force
        
        # Apply forces (Newton's third law: equal and opposite)
        fx = nx * total_force
        fy = ny * total_force
        
        self.point1.apply_force(-fx, -fy)  # Force on mass 1, Pull point1 toward point2
        self.point2.apply_force(fx, fy)    # Force on mass 2 Pull point2 toward point1 (opposite)
    
    def get_current_length(self):
        """Get the current length of the spring."""
        dx = self.point1.x - self.point2.x
        dy = self.point1.y - self.point2.y
        return math.sqrt(dx * dx + dy * dy)
    
    def get_stretch_ratio(self):
        """Get how much the spring is stretched relative to rest length."""
        return self.get_current_length() / self.rest_length 