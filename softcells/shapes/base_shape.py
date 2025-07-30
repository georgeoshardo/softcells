"""
Base shape implementation for soft body physics.
This module contains no rendering dependencies.
"""

import math
import numpy as np
from numba import njit, prange

from ..core import Spring, PointMass
from ..utils.geometry import vectorized_orientations, on_segment
from ..config import (
    DEFAULT_SHAPE_COLOR, DEFAULT_LINE_WIDTH, COLLISION_SLOP, 
    COLLISION_CORRECTION_PERCENT, COLLISION_RESTITUTION, DEFAULT_DT, OU_TAU
)
fastmath = False
parallel = False

@njit(cache=True, fastmath=fastmath, nogil=True, parallel=False)
def _compute_windings_batch(x_coords, y_coords, world_width, world_height):
    """Batch compute winding numbers for all points."""
    n = len(x_coords)

    winding_x_out = np.empty(n, dtype=np.int32)
    winding_y_out = np.empty(n, dtype=np.int32)
    for i in range(n):
        winding_x_out[i] = int(x_coords[i] // world_width)
        winding_y_out[i] = int(y_coords[i] // world_height)
    return winding_x_out, winding_y_out
    


@njit(cache=True, parallel=parallel, fastmath=fastmath, nogil=True)
def _ray_cast_intersections_stable(test_point_x, test_point_y, outside_point_x, outside_point_y, 
                                  shape_points_x, shape_points_y):
    """
    Numba-compiled ray casting algorithm for point-in-polygon testing.
    Uses numpy arrays for stable type inference.
    
    Args:
        test_point_x, test_point_y: Coordinates of the test point (float64)
        outside_point_x, outside_point_y: Coordinates of a point outside the shape (float64)
        shape_points_x, shape_points_y: Numpy arrays of shape vertices coordinates (float64[:])
    
    Returns:
        int: Number of intersections between ray and shape edges
    """
    intersections = np.int32(0)
    num_points = shape_points_x.shape[0]
    
    for i in range(num_points):
        edge_start_x = shape_points_x[i]
        edge_start_y = shape_points_y[i] 
        edge_end_x = shape_points_x[(i + 1) % num_points]
        edge_end_y = shape_points_y[(i + 1) % num_points]
        
        # Inline orientation calculations for better performance
        # o1 = orientation((test_point_x, test_point_y), (outside_point_x, outside_point_y), (edge_start_x, edge_start_y))
        val1 = (outside_point_y - test_point_y) * (edge_start_x - outside_point_x) - (outside_point_x - test_point_x) * (edge_start_y - outside_point_y)
        o1 = 0 if val1 == 0.0 else (1 if val1 > 0.0 else 2)
        
        # o2 = orientation((test_point_x, test_point_y), (outside_point_x, outside_point_y), (edge_end_x, edge_end_y))  
        val2 = (outside_point_y - test_point_y) * (edge_end_x - outside_point_x) - (outside_point_x - test_point_x) * (edge_end_y - outside_point_y)
        o2 = 0 if val2 == 0.0 else (1 if val2 > 0.0 else 2)
        
        # o3 = orientation((edge_start_x, edge_start_y), (edge_end_x, edge_end_y), (test_point_x, test_point_y))
        val3 = (edge_end_y - edge_start_y) * (test_point_x - edge_end_x) - (edge_end_x - edge_start_x) * (test_point_y - edge_end_y)
        o3 = 0 if val3 == 0.0 else (1 if val3 > 0.0 else 2)
        
        # o4 = orientation((edge_start_x, edge_start_y), (edge_end_x, edge_end_y), (outside_point_x, outside_point_y))
        val4 = (edge_end_y - edge_start_y) * (outside_point_x - edge_end_x) - (edge_end_x - edge_start_x) * (outside_point_y - edge_end_y)
        o4 = 0 if val4 == 0.0 else (1 if val4 > 0.0 else 2)
        
        # General case of intersection
        if o1 != o2 and o3 != o4:
            intersections += 1
            continue
            
        # Special case for collinear points
        if o3 == 0:
            # Check if test_point lies on segment (edge_start, edge_end)
            if (test_point_x <= max(edge_start_x, edge_end_x) and test_point_x >= min(edge_start_x, edge_end_x) and
                test_point_y <= max(edge_start_y, edge_end_y) and test_point_y >= min(edge_start_y, edge_end_y)):
                intersections += 1
    
    return intersections

#TODO how much faster is this?
@njit(cache=True, fastmath=fastmath, nogil=True)
def _ray_cast_optimized(test_point_x, test_point_y, outside_point_x, outside_point_y, 
                       shape_points_x, shape_points_y):
    """
    Optimized ray casting with early termination and vectorized operations.
    """
    intersections = 0
    num_points = shape_points_x.shape[0]
    
    # Pre-compute ray direction
    ray_dx = outside_point_x - test_point_x
    ray_dy = outside_point_y - test_point_y
    
    for i in range(num_points):
        edge_start_x = shape_points_x[i]
        edge_start_y = shape_points_y[i]
        edge_end_x = shape_points_x[(i + 1) % num_points]
        edge_end_y = shape_points_y[(i + 1) % num_points]
        
        # Quick bounding box check for edge
        edge_min_y = min(edge_start_y, edge_end_y)
        edge_max_y = max(edge_start_y, edge_end_y)
        
        # Early termination: if ray doesn't cross edge's Y range
        if test_point_y < edge_min_y or test_point_y > edge_max_y:
            continue
            
        # Quick check: if edge is completely to the left of test point
        edge_max_x = max(edge_start_x, edge_end_x)
        if edge_max_x < test_point_x:
            continue
            
        # Use parametric line intersection (faster than orientation tests)
        edge_dx = edge_end_x - edge_start_x
        edge_dy = edge_end_y - edge_start_y
        
        # Avoid division by zero
        if abs(edge_dy) < 1e-10:
            continue
            
        # Calculate intersection parameter
        t_edge = (test_point_y - edge_start_y) / edge_dy
        
        if t_edge >= 0.0 and t_edge <= 1.0:
            # Calculate x coordinate of intersection
            intersect_x = edge_start_x + t_edge * edge_dx
            
            # Check if intersection is to the right of test point
            if intersect_x > test_point_x:
                intersections += 1
    
    return intersections

@njit(cache=True, parallel=parallel, fastmath=fastmath, nogil=True)
def _get_bounding_box_coords_and_windings(x_coords, y_coords, winding_x_coords, winding_y_coords):
    """
    Numba-compiled function to calculate bounding box coordinates and unique windings.
    
    Args:
        x_coords: Numpy array of x coordinates (float64[:])
        y_coords: Numpy array of y coordinates (float64[:]) 
        winding_x_coords: Numpy array of winding x coordinates (int32[:])
        winding_y_coords: Numpy array of winding y coordinates (int32[:])
    
    Returns:
        tuple: (min_x, max_x, min_y, max_y, unique_windings_x, unique_windings_y)
    """
    if len(x_coords) == 0:
        return 0.0, 0.0, 0.0, 0.0, np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)
    
    # Calculate bounding box
    min_x = np.float64(x_coords[0])
    max_x = np.float64(x_coords[0])
    min_y = np.float64(y_coords[0])
    max_y = np.float64(y_coords[0])
    
    for i in range(1, len(x_coords)):
        if x_coords[i] < min_x:
            min_x = x_coords[i]
        if x_coords[i] > max_x:
            max_x = x_coords[i]
        if y_coords[i] < min_y:
            min_y = y_coords[i]
        if y_coords[i] > max_y:
            max_y = y_coords[i]
    
    # Calculate unique windings for bounding box corners
    # We need windings for: (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)
    bb_windings_x = np.array([0, 0, 0, 0], dtype=np.int32)  # Will be overwritten
    bb_windings_y = np.array([0, 0, 0, 0], dtype=np.int32)  # Will be overwritten
    
    # Find windings for each corner by looking at the closest original point
    # This is a simplified approach - in practice you might want more sophisticated logic
    corner_count = 0
    
    # For each bounding box corner, find the winding from the closest original point
    corners_x = np.array([min_x, max_x, max_x, min_x], dtype=np.float64)
    corners_y = np.array([min_y, min_y, max_y, max_y], dtype=np.float64)
    
    for corner_idx in range(4):
        corner_x = corners_x[corner_idx]
        corner_y = corners_y[corner_idx]
        
        # Find closest original point to this corner
        min_dist_sq = float('inf')
        closest_idx = 0
        
        for i in range(len(x_coords)):
            dist_sq = (x_coords[i] - corner_x)**2 + (y_coords[i] - corner_y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_idx = i
        
        bb_windings_x[corner_idx] = winding_x_coords[closest_idx]
        bb_windings_y[corner_idx] = winding_y_coords[closest_idx]
    
    # Remove duplicates from windings (simple O(n²) approach for small arrays)
    unique_windings_x = np.empty(4, dtype=np.int32)
    unique_windings_y = np.empty(4, dtype=np.int32)
    unique_count = 0
    
    for i in range(4):
        is_duplicate = False
        for j in range(unique_count):
            if bb_windings_x[i] == unique_windings_x[j] and bb_windings_y[i] == unique_windings_y[j]:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_windings_x[unique_count] = bb_windings_x[i]
            unique_windings_y[unique_count] = bb_windings_y[i]
            unique_count += 1
    
    # Return only the unique portion of the arrays
    return (min_x, max_x, min_y, max_y, 
            unique_windings_x[:unique_count], unique_windings_y[:unique_count])


def _warm_up_ray_cast_cache():
    """
    Warm up the Numba cache for the ray casting function to avoid compilation delays.
    Call this once at startup for better performance.
    """
    # Create simple test data
    test_x = np.array([0.0, 1.0, 0.5], dtype=np.float64)
    test_y = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    test_winding_x = np.array([0, 0, 0], dtype=np.int32)
    test_winding_y = np.array([0, 0, 0], dtype=np.int32)
    
    # Call the functions once to trigger compilation
    _ray_cast_intersections_stable(0.5, 0.3, 2.0, 0.3, test_x, test_y)
    _get_bounding_box_coords_and_windings(test_x, test_y, test_winding_x, test_winding_y)


class Shape:
    """
    A shape is a collection of point masses connected in order.
    When you connect the points in order with line segments, you create a shape.
    Supports pressure-based physics to maintain shape integrity.
    """
    
    def __init__(self, points=None, identity=0):
        """
        Initialize a shape with a list of point masses.
        
        Args:
            points (list): List of PointMass objects
        """
        self.points = points if points is not None else []
        self.color = DEFAULT_SHAPE_COLOR
        self.line_width = DEFAULT_LINE_WIDTH
        
        # Pressure physics parameters
        self.pressure_enabled = True
        self.pressure_amount = 0
        self.normal_vectors = []  # Normal vectors for each edge
        self.edge_lengths = []    # Length of each edge
        self.damping_factor = 0.98  # Velocity damping to reduce oscillations
        self.initial_volume = 0.0   # Target volume for pressure regulation
        self.current_volume = 0.0   # Current volume
        self.scalar_pressure = 0.0  # Calculated pressure value
        
        # Spring physics parameters
        self.springs = []  # List of Spring objects connecting points
        self.spring_stiffness = 100.0  # Default spring stiffness
        self.spring_damping = 10.0     # Default spring damping
        self.springs_enabled = True    # Enable/disable spring forces
        
        # Drag physics parameters
        self.drag_coefficient = 0.0    # Default drag coefficient
        self.drag_type = "linear"      # Default drag type ("linear" or "quadratic")
        self.drag_enabled = True       # Enable/disable drag forces

        self.random_force = np.zeros(2)

    def random_noise(self, sigma):
        return sigma * np.random.normal(0,1,size=2)

    def set_identity(self, identity):
        """Set a unique identifier for this shape."""
        self.identity = identity

    def add_point(self, point):
        """Add a point mass to this shape."""
        self.points.append(point)
    
    def get_points(self):
        """Get all point masses in this shape."""
        return self.points
    
    def get_positions(self):
        """Get positions of all points as a list of (x, y) tuples."""
        return [(point.x, point.y) for point in self.points]
    
    def apply_force_to_all(self, fx, fy):
        """Apply a force to all points in this shape."""
        for point in self.points:
            point.apply_force(fx, fy)
    
    def update_all(self, dt):
        """Update physics for all points in this shape."""
        for point in self.points:
            point.update(dt)
            # Apply damping to reduce oscillations
            point.vx *= self.damping_factor
            point.vy *= self.damping_factor
    
    def set_cell_unique_id(self, cell_unique_id):
        self.cell_unique_id = cell_unique_id

    def set_color(self, color):
        """Set the rendering color for this shape."""
        self.color = color
    
    def set_pressure(self, pressure_amount):
        """Set the internal gas pressure amount."""
        self.pressure_amount = pressure_amount
    
    def enable_pressure(self, enabled=True):
        """Enable or disable pressure physics."""
        self.pressure_enabled = enabled
    
    def set_spring_properties(self, stiffness, damping):
        """Set spring stiffness and damping for all springs in this shape."""
        self.spring_stiffness = stiffness
        self.spring_damping = damping
        # Update existing springs
        for spring in self.springs:
            spring.stiffness = stiffness
            spring.damping = damping
    
    def enable_springs(self, enabled=True):
        """Enable or disable spring physics."""
        self.springs_enabled = enabled
    
    def set_drag_properties(self, drag_coefficient, drag_type="linear"):
        """Set drag coefficient and type for all points in this shape."""
        self.drag_coefficient = drag_coefficient
        self.drag_type = drag_type
        # Update all points in the shape
        for point in self.points:
            point.set_drag_coefficient(drag_coefficient)
    
    def enable_drag(self, enabled=True):
        """Enable or disable drag physics."""
        self.drag_enabled = enabled
    
    def apply_drag_forces(self):
        """Apply drag forces to all points in this shape."""
        if not self.drag_enabled:
            return
        
        for point in self.points:
            point.apply_drag_force(self.drag_type)
    
    def create_springs(self):
        """Create springs between adjacent points in the shape."""
        if len(self.points) < 2:
            return
        
        self.springs.clear()
        num_points = len(self.points)
        
        # Create springs between adjacent points (including wrapping around)
        for i in range(num_points):
            point1 = self.points[i]
            point2 = self.points[(i + 1) % num_points]  # Wrap around to close the shape
            
            spring = Spring(point1, point2, self.spring_stiffness, self.spring_damping)
            self.springs.append(spring)
    
    def apply_spring_forces(self):
        """Apply spring forces between connected point masses."""
        if not self.springs_enabled:
            return
        
        for spring in self.springs:
            spring.apply_force()
    
    def calculate_volume(self):
        """
        Calculate the volume (area in 2D) of the shape using Gauss theorem.
        Formula from paper: V = Σ 0.5 * |x1-x2| * |nx| * dl
        
        Returns:
            float: Volume/area of the shape
        """
        if len(self.points) < 3:
            return 0.0
        
        volume = 0
        num_points = len(self.points)
        
        # Calculate normal vectors and edge lengths first
        self.normal_vectors = []
        self.edge_lengths = []
        
        for i in range(num_points):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % num_points]  # Wrap around to close the shape
            
            # Calculate edge vector and length
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            edge_length = math.sqrt(dx * dx + dy * dy)
            
            # Avoid division by zero
            if edge_length > 0.001:
                # Calculate normal vector (perpendicular to edge, pointing outward)
                # From paper: nx = (y1-y2)/r12d, ny = -(x1-x2)/r12d
                nx = dy / edge_length
                ny = -dx / edge_length
            else:
                nx = 0.0
                ny = 0.0
            
            self.normal_vectors.append((nx, ny))
            self.edge_lengths.append(edge_length)
            
            # Add to volume calculation using Gauss theorem
            # V += 0.5 * |x1-x2| * |nx| * dl
            mid_x = (p1.x + p2.x) / 2.0

            volume += mid_x * nx * edge_length
        
        return max(abs(volume), 1.0)  # Avoid zero volume
    
    def apply_pressure_forces(self):
        """
        Apply pressure forces to maintain the shape's volume.
        Based on the pressure model from the soft body paper, but stabilized.
        """
        if not self.pressure_enabled or len(self.points) < 3:
            return
        
        if not self.initial_volume:
            self.initial_volume = self.calculate_volume()

        self.current_volume = self.calculate_volume()

        self.scalar_pressure = self.pressure_amount * self.initial_volume /  self.current_volume 
        # Limit pressure force to prevent instability
        max_pressure = self.scalar_pressure
        self.scalar_pressure = max(-max_pressure, min(max_pressure, self.scalar_pressure))

        num_points = len(self.points)
        
        # Apply pressure force to each edge
        for i in range(num_points):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % num_points]
            
            if i < len(self.normal_vectors) and i < len(self.edge_lengths):
                nx, ny = self.normal_vectors[i]
                edge_length = self.edge_lengths[i]
                
                # Calculate pressure force magnitude (scaled by edge length)
                pressure_magnitude = self.scalar_pressure * edge_length   # |F| = P*A
                
                # Calculate force components
                fx = nx * pressure_magnitude
                fy = ny * pressure_magnitude
                
                # Apply force to both endpoints of the edge
                # This distributes the pressure force evenly
                p1.apply_force(fx * 0.5, fy * 0.5)
                p2.apply_force(fx * 0.5, fy * 0.5)
    
    def apply_ou_forces(self):
        """
        Apply Ornstein-Uhlenbeck forces to all points in the shape.
        This simulates random thermal motion.
        """

        ### Compute random forces
        random_noise = self.random_noise(
            sigma=np.sqrt(2*10000000)
        )

        # see Gillespie, PRE 95
        # "Exact numerical simulation of the Ornstein-Uhlenbeck process and its integral"

        deterministic_ou_term = np.exp(-DEFAULT_DT/100)
        random_ou_term = random_noise * np.sqrt(1-np.exp(-DEFAULT_DT/100))

        self.random_force = self.random_force * deterministic_ou_term + random_ou_term
    
        
        for point in self.points:
            point.apply_force(self.random_force[0], self.random_force[1])


    def _get_bounding_box(self):
        """Get the min and max coordinates of the shape."""
        if not self.points:
            return 0, 0, 0, 0
        x_coords = [p.x for p in self.points]
        y_coords = [p.y for p in self.points]
        return min(x_coords), max(x_coords), min(y_coords), max(y_coords)
    
    def _get_bounding_box_with_windings(self):
        """Get the min and max coordinates of the shape and a list of unique winding pairs."""
        if not self.points:
            return (0, 0, 0, 0), []  # Return an empty list for windings

        x_coords = [p.x for p in self.points]
        y_coords = [p.y for p in self.points]

        # Create a list of (winding_x, winding_y) tuples for all points
        all_windings = zip([p.winding_x for p in self.points], [p.winding_y for p in self.points])

        # Find the unique winding pairs by converting the list to a set and back
        unique_windings = list(set(all_windings))

        return (min(x_coords), max(x_coords), min(y_coords), max(y_coords)), unique_windings

    def is_point_inside(self, test_point, outside=False):
        """
        Check if a point is inside this shape using the ray-casting algorithm.
        Optimized with Numba for performance.
        
        Args:
            test_point (PointMass): The point to check.
            outside (bool): If True, checks if the point is outside the shape.
        
        Returns:
            bool: True if the point is inside, False otherwise.
        """
        if len(self.points) < 3:
            return False

        # Extract coordinates into numpy arrays for Numba processing
        x_coords = np.array([p.x for p in self.points], dtype=np.float64)
        y_coords = np.array([p.y for p in self.points], dtype=np.float64)

        world_width = self.points[0].world_width
        world_height = self.points[0].world_height

        winding_x_coords, winding_y_coords = _compute_windings_batch(
            x_coords, y_coords, world_width, world_height,
        )
        
        
        # Use Numba-compiled function for fast bounding box and windings computation
        min_x, max_x, min_y, max_y, unique_windings_x, unique_windings_y = _get_bounding_box_coords_and_windings(
            x_coords, y_coords, winding_x_coords, winding_y_coords
        )



        # Pre-compute shape coordinates as numpy arrays for stable type inference
        shape_points_x = np.array([p.x for p in self.points], dtype=np.float64)
        shape_points_y = np.array([p.y for p in self.points], dtype=np.float64)

        for i in range(len(unique_windings_x)):
            winding_x = int(unique_windings_x[i])
            winding_y = int(unique_windings_y[i])
            
            test_point_in_shape_referential_x = (
                test_point.x + (winding_x - test_point.winding_x) * test_point.world_width
            )
            test_point_in_shape_referential_y = (
                test_point.y + (winding_y - test_point.winding_y) * test_point.world_height
            )

            outside_point_x = max_x + 10.0  # Use max_x directly instead of bb3.x
            outside_point_y = test_point_in_shape_referential_y
            
            # Use the Numba-compiled function for the expensive ray-casting computation
            intersections = _ray_cast_intersections_stable(
                float(test_point_in_shape_referential_x), float(test_point_in_shape_referential_y),
                float(outside_point_x), float(outside_point_y),
                shape_points_x, shape_points_y
            )

            if outside:
                if intersections % 2 == 0:
                    return True, (winding_x, winding_y)
            else:
                if intersections % 2 == 1:
                    return True, (winding_x, winding_y)  # Point is inside the shape

        return False, (None, None)  # Point is outside the shape
    
    def is_point_outside(self, test_point):
        """
        Check if a point is outside this shape using the ray-casting algorithm.
        This method finds the closest winding representation where the point is outside
        and returns that winding for collision resolution.
        
        Args:
            test_point (PointMass): The point to check.
        
        Returns:
            tuple: (is_outside, at_windings) where is_outside is True if point is outside,
                   and at_windings contains the winding numbers for collision resolution
        """
        if len(self.points) < 3:
            return False, (None, None)

        # Extract coordinates into numpy arrays for Numba processing
        x_coords = np.array([p.x for p in self.points], dtype=np.float64)
        y_coords = np.array([p.y for p in self.points], dtype=np.float64)

        world_width = self.points[0].world_width
        world_height = self.points[0].world_height

        winding_x_coords, winding_y_coords = _compute_windings_batch(
            x_coords, y_coords, world_width, world_height,
        )
        
        # Use Numba-compiled function for fast bounding box and windings computation
        min_x, max_x, min_y, max_y, unique_windings_x, unique_windings_y = _get_bounding_box_coords_and_windings(
            x_coords, y_coords, winding_x_coords, winding_y_coords
        )

        # Pre-compute shape coordinates as numpy arrays for stable type inference
        shape_points_x = np.array([p.x for p in self.points], dtype=np.float64)
        shape_points_y = np.array([p.y for p in self.points], dtype=np.float64)

        min_dist = float('inf')
        best_winding = (None, None)
        found_outside = False

        # Check each possible winding to find where the point is outside with minimum distance
        for i in range(len(unique_windings_x)):
            winding_x = int(unique_windings_x[i])
            winding_y = int(unique_windings_y[i])
            
            test_point_in_shape_referential_x = (
                test_point.x + (winding_x - test_point.winding_x) * test_point.world_width
            )
            test_point_in_shape_referential_y = (
                test_point.y + (winding_y - test_point.winding_y) * test_point.world_height
            )

            outside_point_x = max_x + 10.0
            outside_point_y = test_point_in_shape_referential_y
            
            # Use the Numba-compiled function for ray-casting
            intersections = _ray_cast_intersections_stable(
                float(test_point_in_shape_referential_x), float(test_point_in_shape_referential_y),
                float(outside_point_x), float(outside_point_y),
                shape_points_x, shape_points_y
            )

            # If intersections is even, point is outside in this winding
            if intersections % 2 == 0:
                found_outside = True
                
                # Find distance to closest edge in this winding
                edge_result = self.find_closest_edge(test_point, (winding_x, winding_y), return_distance=True)
                if edge_result[0] is not None:  # Check if edge was found
                    p1, p2, t, dist_sq = edge_result
                    dist = math.sqrt(dist_sq)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_winding = (winding_x, winding_y)

        if found_outside and best_winding[0] is not None:
            return True, best_winding

        return False, (None, None)
    

    def find_closest_edge(self, test_point, at_windings, return_distance=False):
        """
        Finds the closest edge of the shape to a given point.

        Args:
            test_point (PointMass): The point to test against.

        Returns:
            tuple: A tuple containing (edge_point1, edge_point2, closest_point_on_segment)
                Returns (None, None, None) if no edge is found.
        """
        min_dist_sq = float('inf')
        closest_edge = None
        closest_point_on_edge = None
        best_t = 0

        winding_x, winding_y = at_windings

        # Safety check: limit winding differences to prevent explosive calculations
        winding_diff_x = winding_x - test_point.winding_x
        winding_diff_y = winding_y - test_point.winding_y
        

        test_point_in_shape_referential = (
            test_point.x + winding_diff_x * test_point.world_width,
            test_point.y + winding_diff_y * test_point.world_height
        )

        if len(self.points) < 2:
            return None, None, None

        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]

            # Vector from p1 to p2
            line_vec_x = p2.x - p1.x
            line_vec_y = p2.y - p1.y

            # Squared length of the edge
            line_len_sq = line_vec_x**2 + line_vec_y**2
            if line_len_sq < 0.00001:
                continue

            # Projection of vector (test_point - p1) onto the edge vector
            t = ((test_point_in_shape_referential[0] - p1.x) * line_vec_x + (test_point_in_shape_referential[1] - p1.y) * line_vec_y) / line_len_sq
            t = max(0, min(1, t))  # Clamp t to the segment [0, 1]

            # Closest point on the line segment
            closest_x = p1.x + t * line_vec_x
            closest_y = p1.y + t * line_vec_y

            # Squared distance from the test point to the closest point on the segment
            dist_sq = (test_point_in_shape_referential[0] - closest_x)**2 + (test_point_in_shape_referential[1] - closest_y)**2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_edge = (p1, p2)
                best_t = t

        if return_distance:
            return closest_edge[0], closest_edge[1], best_t, min_dist_sq
        return closest_edge[0], closest_edge[1], best_t