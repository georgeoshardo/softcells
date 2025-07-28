import math
import pygame
import sys
from numba import njit

global_pressure_amount = 300


@njit(cache=True)
def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
            (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise


@njit(cache=True)
def on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

# Vectorized orientation calculation for performance with numba
# (p1, q1, p2, q2 are all tuples (x, y))
@njit(cache=True)
def vectorized_orientations(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    return o1, o2, o3, o4

class PointMass:
    """
    A point mass represents a single point in space with mass and velocity.
    Forces can act upon it to affect its motion, but it has no collision of its own.
    Uses Verlet integration for better stability and energy conservation.
    """
    
    def __init__(self, x, y, mass=1.0, drag_coefficient=0.0):
        """
        Initialize a point mass.
        
        Args:
            x (float): Initial x position
            y (float): Initial y position
            mass (float): Mass of the point (default 1.0)
            drag_coefficient (float): Viscous drag coefficient (default 0.0)
        """
        # Current position
        self.x = x
        self.y = y
        
        # Previous position (for Verlet integration)
        self.prev_x = x
        self.prev_y = y
        
        # Velocity (derived from position difference in Verlet)
        self.vx = 0.0
        self.vy = 0.0
        
        # Mass
        self.mass = mass
        
        # Drag coefficient for viscous drag
        self.drag_coefficient = drag_coefficient
        
        # Accumulated forces for this frame
        self.force_x = 0.0
        self.force_y = 0.0
        
        # Flag to handle first integration step
        self.first_step = True
    
    def apply_force(self, fx, fy):
        """
        Apply a force to this point mass.
        
        Args:
            fx (float): Force in x direction
            fy (float): Force in y direction
        """
        self.force_x += fx
        self.force_y += fy
    
    def apply_drag_force(self, drag_type="linear"):
        """
        Apply viscous drag force based on current velocity.
        
        Args:
            drag_type (str): Type of drag - "linear" or "quadratic"
        """
        if self.drag_coefficient <= 0:
            return
        
        if drag_type == "linear":
            # Linear drag: F_drag = -k * v
            drag_fx = -self.drag_coefficient * self.vx
            drag_fy = -self.drag_coefficient * self.vy
        elif drag_type == "quadratic":
            # Quadratic drag: F_drag = -k * v * |v|
            speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
            if speed > 0.001:  # Avoid division by zero
                drag_fx = -self.drag_coefficient * self.vx * speed
                drag_fy = -self.drag_coefficient * self.vy * speed
            else:
                drag_fx = drag_fy = 0.0
        else:
            drag_fx = drag_fy = 0.0
        
        self.apply_force(drag_fx, drag_fy)
    
    def set_drag_coefficient(self, drag_coefficient):
        """Set the drag coefficient."""
        self.drag_coefficient = max(0.0, drag_coefficient)
    
    def update(self, dt):
        """
        Update the point mass position using Verlet integration.
        Verlet integration: x(t+dt) = 2*x(t) - x(t-dt) + a(t)*dt^2
        
        Args:
            dt (float): Time step
        """
        # Apply Newton's second law: F = ma, so a = F/m
        ax = self.force_x / self.mass
        ay = self.force_y / self.mass

        
        if self.first_step:
            # For the first step, use Euler integration to establish previous position
            # Store current position as previous
            self.prev_x = self.x
            self.prev_y = self.y
            
            # Update position using Euler: p = p + v*dt + 0.5*a*dt^2 for the first time step
            self.x += self.vx * dt + 0.5 * ax * dt * dt
            self.y += self.vy * dt + 0.5 * ay * dt * dt
            
            # Update velocity: v = v + a*dt
            self.vx += ax * dt
            self.vy += ay * dt
            
            self.first_step = False
        else:
            # Verlet integration
            # Store current position
            temp_x = self.x
            temp_y = self.y
            
            # Calculate new position using standard Verlet
            #x(t+Δt) = 2x(t) - x(t-Δt) + a(t)Δt²
            self.x = 2.0 * self.x - self.prev_x + ax * dt * dt
            self.y = 2.0 * self.y - self.prev_y + ay * dt * dt
            
            # Update previous position
            self.prev_x = temp_x
            self.prev_y = temp_y
            
            # Derive velocity from position difference: v(t) = [x(t) - x(t-Δt)] / Δt
            self.vx = (self.x - self.prev_x) / dt
            self.vy = (self.y - self.prev_y) / dt
        
        # Clear forces for next frame
        self.force_x = 0.0
        self.force_y = 0.0
    
    def get_position(self):
        """Get the current position as a tuple."""
        return (self.x, self.y)
    
    def get_velocity(self):
        """Get the current velocity as a tuple."""
        return (self.vx, self.vy)
    
    def set_position(self, x, y):
        """Set the position directly."""
        # When setting position directly, update previous position to maintain velocity
        self.prev_x = self.x
        self.prev_y = self.y
        self.x = x
        self.y = y
    
    def set_velocity(self, vx, vy):
        """Set the velocity directly by adjusting previous position."""
        self.vx = vx
        self.vy = vy
        # Adjust previous position to achieve desired velocity: prev = current - v*dt
        # Note: This assumes a standard dt, might need adjustment in actual use
        if not self.first_step:
            dt = 1.0/60.0  # Assume standard frame rate for velocity setting
            self.prev_x = self.x - vx * dt
            self.prev_y = self.y - vy * dt


class Spring:
    """
    A spring connects two point masses with elastic and damping forces.
    Implements Hooke's law with velocity damping.
    """
    
    def __init__(self, point1, point2, stiffness=50.0, damping=5.0, rest_length=None):
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


class Shape:
    """
    A shape is a collection of point masses connected in order.
    When you connect the points in order with line segments, you create a shape.
    Supports pressure-based physics to maintain shape integrity.
    """
    
    def __init__(self, points=None):
        """
        Initialize a shape with a list of point masses.
        
        Args:
            points (list): List of PointMass objects
        """
        self.points = points if points is not None else []
        self.color = (100, 150, 255)  # Default blue color
        self.line_width = 2
        
        # Pressure physics parameters
        self.pressure_enabled = True
        self.normal_vectors = []  # Normal vectors for each edge
        self.edge_lengths = []    # Length of each edge
        self.damping_factor = 0.98  # Velocity damping to reduce oscillations
        self.initial_volume = 0.0   # Target volume for pressure regulation
        
        # Spring physics parameters
        self.springs = []  # List of Spring objects connecting points
        self.spring_stiffness = 100.0  # Default spring stiffness
        self.spring_damping = 10.0     # Default spring damping
        self.springs_enabled = True    # Enable/disable spring forces
        
        # Drag physics parameters
        self.drag_coefficient = 0.0    # Default drag coefficient
        self.drag_type = "linear"      # Default drag type ("linear" or "quadratic")
        self.drag_enabled = True       # Enable/disable drag forces
    
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
    
    def render(self, screen):
        """
        Render this shape by drawing springs and points.
        Springs are colored based on their stretch ratio.
        
        Args:
            screen: Pygame screen surface
        """
        if len(self.points) < 2:
            return
        
        # Draw springs with color based on stretch
        if self.springs_enabled and len(self.springs) > 0:
            for spring in self.springs:
                # Calculate spring color based on stretch
                stretch_ratio = spring.get_stretch_ratio()
                
                if stretch_ratio > 1.1:  # Stretched (red)
                    intensity = min(1.0, (stretch_ratio - 1.0) * 3.0)
                    color = (int(255 * intensity), 0, int(255 * (1 - intensity)))
                elif stretch_ratio < 0.9:  # Compressed (blue)
                    intensity = min(1.0, (1.0 - stretch_ratio) * 3.0)
                    color = (int(255 * (1 - intensity)), 0, int(255 * intensity))
                else:  # Normal length (green)
                    color = (0, 255, 0)
                
                # Draw the spring
                pos1 = (int(spring.point1.x), int(spring.point1.y))
                pos2 = (int(spring.point2.x), int(spring.point2.y))
                pygame.draw.line(screen, color, pos1, pos2, self.line_width)
        else:
            # Draw basic lines if springs are disabled
            positions = [(int(point.x), int(point.y)) for point in self.points]
            for i in range(len(positions)):
                start_pos = positions[i]
                end_pos = positions[(i + 1) % len(positions)]
                pygame.draw.line(screen, self.color, start_pos, end_pos, self.line_width)
        
        # Draw the individual points
        for point in self.points:
            radius = max(2, int(3 + point.mass))
            pos = (int(point.x), int(point.y))
            pygame.draw.circle(screen, self.color, pos, radius)
            pygame.draw.circle(screen, (255, 255, 255), pos, radius, 1)

        # Add these new methods to the Shape class:
    def _get_bounding_box(self):
        """Get the min and max coordinates of the shape."""
        if not self.points:
            return 0, 0, 0, 0
        x_coords = [p.x for p in self.points]
        y_coords = [p.y for p in self.points]
        return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

    def is_point_inside(self, test_point):
        """
        Check if a point is inside this shape using the ray-casting algorithm.
        
        Args:
            test_point (PointMass): The point to check.
        
        Returns:
            bool: True if the point is inside, False otherwise.
        """
        if len(self.points) < 3:
            return False

        # Get a point guaranteed to be outside the shape's bounding box.
        _, max_x, _, _ = self._get_bounding_box()
        outside_point = (max_x + 10, test_point.y)
        
        intersections = 0
        num_points = len(self.points)

        # The ray is from test_point to outside_point.
        p1 = (test_point.x, test_point.y)
        q1 = outside_point

        # Iterate over each edge of the shape.
        for i in range(num_points):
            edge_start = self.points[i]
            edge_end = self.points[(i + 1) % num_points]
            
            p2 = (edge_start.x, edge_start.y)
            q2 = (edge_end.x, edge_end.y)

            # Check for intersection between the ray and the current edge.
            # This is a standard line segment intersection algorithm.

            o1, o2, o3, o4 = vectorized_orientations(p1, q1, p2, q2)

            # General case of intersection
            if o1 != o2 and o3 != o4:
                intersections += 1
                continue

            # Special Cases for collinear points
            if o3 == 0 and on_segment(p2, p1, q2):
                intersections += 1

        return intersections % 2 == 1
    # Add this new method to the Shape class in soft_body_simulation.py

    def find_closest_edge(self, test_point):
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
            t = ((test_point.x - p1.x) * line_vec_x + (test_point.y - p1.y) * line_vec_y) / line_len_sq
            t = max(0, min(1, t))  # Clamp t to the segment [0, 1]

            # Closest point on the line segment
            closest_x = p1.x + t * line_vec_x
            closest_y = p1.y + t * line_vec_y

            # Squared distance from the test point to the closest point on the segment
            dist_sq = (test_point.x - closest_x)**2 + (test_point.y - closest_y)**2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_edge = (p1, p2)
                closest_point_on_edge = (closest_x, closest_y, t) # Also return t for weighting

        return closest_edge[0], closest_edge[1], closest_point_on_edge


class CircleShape(Shape):
    """
    A shape that arranges point masses in a circle pattern.
    Uses pressure physics to maintain circular form and springs for structural integrity.
    """
    
    def __init__(self, center_x, center_y, radius, num_points=8, point_mass=1.0, pressure=global_pressure_amount, 
                 spring_stiffness=150.0, spring_damping=10.0, drag_coefficient=0.0, drag_type="linear"):
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
            
            point = PointMass(x, y, point_mass, drag_coefficient)
            self.add_point(point)
        
        # Set spring properties and create springs
        self.set_spring_properties(spring_stiffness, spring_damping)
        self.create_springs()
        
        # Set drag properties
        self.set_drag_properties(drag_coefficient, drag_type)
        
        ## Set pressure based on circle size (much more conservative)
            # Scale pressure with radius for stability (much smaller values)
        ##if pressure is None:
        #    auto_pressure = radius * 0.5  # Much smaller pressure
        #else:
        #    auto_pressure = pressure
        #self.initial_volume = self.calculate_volume()
        self.set_pressure(pressure)
        # Set a distinct color for circle shapes
        self.set_color((150, 255, 150))  # Green


class PhysicsSimulation:
    """
    A generic Pygame-based visualization of physics simulation.
    Supports both individual point masses and shapes composed of point masses.
    """
    
    def __init__(self, width=1500, height=1000):
        """Initialize the simulation window and physics world."""
        pygame.init()
        
        # Window setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Soft Body Physics Simulation")
        
        # Physics setup with improved stability from Verlet integration
        self.clock = pygame.time.Clock()
        self.dt = 1.0 / 100.0  # Larger time step thanks to Verlet stability (was 1/200)
        self.gravity = 0.0  # Reduced gravity for more stable simulation
        
        # Drag physics parameters
        self.global_drag_coefficient = 5.5  # Global drag coefficient
        self.drag_type = "linear"           # Global drag type
        self.drag_enabled = True            # Global drag enable/disable
        
        # Simulation control
        self.paused = False
        self.step_next_frame = False
        
        # Create physics objects
        self.points = []  # Individual point masses
        self.shapes = []  # Shape objects (which contain point masses)
        self.create_initial_scene()
        
        # Colors
        self.background_color = (20, 20, 30)
        self.point_color = (100, 150, 255)
        self.trail_color = (50, 75, 125)
        
        # Trail system for visual effect (for individual points only)
        self.point_trails = [[] for _ in self.points]
        self.max_trail_length = 30
    
    def create_initial_scene(self):
        """Create initial physics objects for the simulation."""
        # Create some circle shapes with different spring and pressure settings
        # Stiff circle - high spring stiffness and pressure
        circle1 = CircleShape(500, 100, 50, num_points=20, point_mass=1.0, pressure=global_pressure_amount,
                             spring_stiffness=1150.0, spring_damping=10.0, 
                             drag_coefficient=self.global_drag_coefficient, drag_type=self.drag_type)
        circle1.set_color((150, 255, 150))  # Green - Stiff
        self.shapes.append(circle1)
        
        # Update trail system to match current points
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_TAB:
                    # Toggle pause
                    self.paused = not self.paused
                elif event.key == pygame.K_PERIOD:
                    # Step one frame when paused
                    if self.paused:
                        self.step_next_frame = True
                elif event.key == pygame.K_r:
                    # Reset simulation
                    self.points.clear()
                    self.shapes.clear()
                    self.point_trails.clear()
                    self.create_initial_scene()
                elif event.key == pygame.K_SPACE:
                    # Add a new point at mouse position
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    new_point = PointMass(mouse_x, mouse_y, 1.0, self.global_drag_coefficient)
                    self.points.append(new_point)
                    self.point_trails.append([])
                elif event.key == pygame.K_c:
                    # Add a new circle shape at mouse position
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    new_circle = CircleShape(mouse_x, mouse_y, 50, num_points=50, point_mass=1.0, pressure=global_pressure_amount,
                             spring_stiffness=1150.0, spring_damping=10.0,
                             drag_coefficient=self.global_drag_coefficient, drag_type=self.drag_type)
                    new_circle.set_color((255, 255, 100))  # Yellow
                    self.shapes.append(new_circle)
                elif event.key == pygame.K_p:
                    # Toggle pressure physics for all shapes
                    for shape in self.shapes:
                        shape.enable_pressure(not shape.pressure_enabled)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Increase pressure for all shapes
                    for shape in self.shapes:
                        shape.set_pressure(shape.pressure_amount * 1.2)
                elif event.key == pygame.K_MINUS:
                    # Decrease pressure for all shapes
                    for shape in self.shapes:
                        shape.set_pressure(shape.pressure_amount * 0.8)
                elif event.key == pygame.K_s:
                    # Toggle spring physics for all shapes
                    for shape in self.shapes:
                        shape.enable_springs(not shape.springs_enabled)
                elif event.key == pygame.K_q:
                    # Increase spring stiffness
                    for shape in self.shapes:
                        new_stiffness = shape.spring_stiffness * 1.2
                        shape.set_spring_properties(new_stiffness, shape.spring_damping)
                elif event.key == pygame.K_a:
                    # Decrease spring stiffness
                    for shape in self.shapes:
                        new_stiffness = shape.spring_stiffness * 0.8
                        shape.set_spring_properties(new_stiffness, shape.spring_damping)
                elif event.key == pygame.K_w:
                    # Increase spring damping
                    for shape in self.shapes:
                        new_damping = shape.spring_damping * 1.2
                        shape.set_spring_properties(shape.spring_stiffness, new_damping)
                elif event.key == pygame.K_z:
                    # Decrease spring damping
                    for shape in self.shapes:
                        new_damping = shape.spring_damping * 0.8
                        shape.set_spring_properties(shape.spring_stiffness, new_damping)
                elif event.key == pygame.K_d:
                    # Toggle drag physics for all objects
                    self.drag_enabled = not self.drag_enabled
                    for shape in self.shapes:
                        shape.enable_drag(self.drag_enabled)
                elif event.key == pygame.K_e:
                    # Increase drag coefficient
                    self.global_drag_coefficient *= 1.2
                    for shape in self.shapes:
                        shape.set_drag_properties(self.global_drag_coefficient, shape.drag_type)
                    for point in self.points:
                        point.set_drag_coefficient(self.global_drag_coefficient)
                elif event.key == pygame.K_x:
                    # Decrease drag coefficient
                    self.global_drag_coefficient *= 0.8
                    for shape in self.shapes:
                        shape.set_drag_properties(self.global_drag_coefficient, shape.drag_type)
                    for point in self.points:
                        point.set_drag_coefficient(self.global_drag_coefficient)
                elif event.key == pygame.K_t:
                    # Toggle drag type between linear and quadratic
                    self.drag_type = "quadratic" if self.drag_type == "linear" else "linear"
                    for shape in self.shapes:
                        shape.set_drag_properties(shape.drag_coefficient, self.drag_type)
        return True
    
    def update_physics(self):
        """Update all physics objects with forces and motion."""
        # Update individual point masses
        for i, point in enumerate(self.points):
            # Apply gravity
            point.apply_force(0, self.gravity * point.mass)
            
            # Apply drag force if enabled
            if self.drag_enabled:
                point.apply_drag_force(self.drag_type)
            
            # Update physics
            point.update(self.dt)
            
            # Handle boundary conditions (bounce off walls)
            self._handle_boundary_collision(point)
            
            # Update trail
            if i < len(self.point_trails):
                self.point_trails[i].append((int(point.x), int(point.y)))
                if len(self.point_trails[i]) > self.max_trail_length:
                    self.point_trails[i].pop(0)
        
        # Update shapes (which contain point masses)
        for shape in self.shapes:
            # Apply gravity to all points in the shape
            shape.apply_force_to_all(0, self.gravity)
            
            # Apply drag forces to all points in the shape
            shape.apply_drag_forces()
            
            # Apply spring forces between connected points
            shape.apply_spring_forces()
            
            # Apply pressure forces to maintain shape integrity
            shape.apply_pressure_forces()
            
            # Update physics for all points in the shape
            shape.update_all(self.dt)
            
            # Handle boundary collisions for all points in the shape
            for point in shape.get_points():
                self._handle_boundary_collision(point)

        self.handle_collisions()
    
    def _handle_boundary_collision(self, point):
        """Handle collision with window boundaries for a single point."""
        if point.x < 0:
            point.x = 0
            point.vx = -point.vx * 0.8  # Some energy loss on bounce
        elif point.x > self.width:
            point.x = self.width
            point.vx = -point.vx * 0.8
        
        if point.y < 0:
            point.y = 0
            point.vy = -point.vy * 0.8
        elif point.y > self.height:
            point.y = self.height
            point.vy = -point.vy * 0.8


            # Add this new method to the PhysicsSimulation class
    def handle_collisions(self):
        """Detect and resolve collisions between shapes."""
        num_shapes = len(self.shapes)
        if num_shapes < 2:
            return

        for i in range(num_shapes):
            for j in range(i + 1, num_shapes):
                shape_a = self.shapes[i]
                shape_b = self.shapes[j]


                # --- BROAD PHASE: Bounding Box Check ---
                min_ax, max_ax, min_ay, max_ay = shape_a._get_bounding_box()
                min_bx, max_bx, min_by, max_by = shape_b._get_bounding_box()

                # If the bounding boxes do not overlap, skip to the next pair
                if max_ax < min_bx or min_ax > max_bx or max_ay < min_by or min_ay > max_by:
                    continue  # Not colliding, so no need for the expensive check


                # Test points of A inside B
                for point in shape_a.get_points():
                    if shape_b.is_point_inside(point):
                        self.resolve_collision(point, shape_b)

                # Test points of B inside A
                for point in shape_b.get_points():
                    if shape_a.is_point_inside(point):
                        self.resolve_collision(point, shape_a)


    def resolve_collision(self, colliding_point, shape):
            """
            Resolves a collision between a point and a shape.
            Moves points to resolve overlap and updates velocities for bounce.
            """
            edge_p1, edge_p2, closest_point_data = shape.find_closest_edge(colliding_point)

            if not edge_p1:
                return

            closest_x, closest_y, t = closest_point_data

            # --- 1. POSITION RESOLUTION ---
            # The penetration vector points from the edge to the colliding point (inward)
            penetration_vec_x = colliding_point.x - closest_x
            penetration_vec_y = colliding_point.y - closest_y
            penetration_depth = math.sqrt(penetration_vec_x**2 + penetration_vec_y**2)
            
            if penetration_depth < 0.001:
                return

            # The normal points INWARD into the shape.
            normal_x = penetration_vec_x / penetration_depth
            normal_y = penetration_vec_y / penetration_depth
            
            slop = 0.1
            correction_percent = 0.4
            
            correction_depth = max(penetration_depth - slop, 0.0)
            if correction_depth == 0.0:
                return

            inv_mass_p = 1.0 / colliding_point.mass if colliding_point.mass > 0 else 0
            inv_mass_e1 = 1.0 / edge_p1.mass if edge_p1.mass > 0 else 0
            inv_mass_e2 = 1.0 / edge_p2.mass if edge_p2.mass > 0 else 0
            
            total_inv_mass = inv_mass_p + inv_mass_e1 * (1 - t) + inv_mass_e2 * t
            if total_inv_mass < 0.001:
                return

            move_dist = (correction_depth / total_inv_mass) * correction_percent
            
            # Move the colliding point OUTWARD (in the opposite direction of the inward normal)
            colliding_point.x -= normal_x * move_dist * inv_mass_p
            colliding_point.y -= normal_y * move_dist * inv_mass_p
            
            # Move the edge points of the shape OUTWARD to push back
            edge_p1.x += normal_x * move_dist * inv_mass_e1 * (1 - t)
            edge_p1.y += normal_y * move_dist * inv_mass_e1 * (1 - t)
            edge_p2.x += normal_x * move_dist * inv_mass_e2 * t
            edge_p2.y += normal_y * move_dist * inv_mass_e2 * t

            # --- 2. VELOCITY RESOLUTION ---
            edge_vx = edge_p1.vx * (1 - t) + edge_p2.vx * t
            edge_vy = edge_p1.vy * (1 - t) + edge_p2.vy * t

            rel_vx = colliding_point.vx - edge_vx
            rel_vy = colliding_point.vy - edge_vy

            # Velocity along the inward normal. If positive, they are already separating.
            vel_along_normal = rel_vx * normal_x + rel_vy * normal_y
            if vel_along_normal > 0:
                return

            e = 0.3
            # Impulse magnitude (will be positive since vel_along_normal is negative)
            impulse_j = -(1 + e) * vel_along_normal / total_inv_mass
            
            # Apply impulse to push the colliding point OUTWARD (-n)
            colliding_point.vx -= (impulse_j * inv_mass_p) * normal_x
            colliding_point.vy -= (impulse_j * inv_mass_p) * normal_y

            # Apply impulse to push the edge points INWARD (+n)
            edge_p1.vx += (impulse_j * inv_mass_e1 * (1 - t)) * normal_x
            edge_p1.vy += (impulse_j * inv_mass_e1 * (1 - t)) * normal_y
            edge_p2.vx += (impulse_j * inv_mass_e2 * t) * normal_x
            edge_p2.vy += (impulse_j * inv_mass_e2 * t) * normal_y
        
    def render(self):
        """Render the simulation."""
        # Clear screen
        self.screen.fill(self.background_color)
        
        # Draw trails for individual points
        for trail in self.point_trails:
            if len(trail) > 1:
                for i in range(len(trail) - 1):
                    start_pos = trail[i]
                    end_pos = trail[i + 1]
                    pygame.draw.line(self.screen, self.trail_color, start_pos, end_pos, 1)
        
        # Draw shapes first (so they appear behind individual points)
        for shape in self.shapes:
            shape.render(self.screen)
            
            # Draw physics indicators
            if len(shape.points) > 0:
                # Calculate center of shape for text
                center_x = sum(p.x for p in shape.points) / len(shape.points)
                center_y = sum(p.y for p in shape.points) / len(shape.points)
                
                # Create multi-line text display
                font_small = pygame.font.Font(None, 14)
                info_lines = []
                
                if hasattr(shape, 'pressure_enabled') and shape.pressure_enabled:
                    info_lines.append(f"P:{shape.pressure_amount:.1f}")
                    info_lines.append(f"Pr:{shape.scalar_pressure:.1f}")

                info_lines.append(f"V:{shape.current_volume:.1f}")

                if hasattr(shape, 'springs_enabled') and shape.springs_enabled:
                    info_lines.append(f"K:{shape.spring_stiffness:.0f}")
                    info_lines.append(f"D:{shape.spring_damping:.1f}")
                
                if hasattr(shape, 'drag_enabled') and shape.drag_enabled:
                    info_lines.append(f"Drag:{shape.drag_coefficient:.2f}")
                    info_lines.append(f"Type:{shape.drag_type}")
                
                # Draw each line of info
                for i, line in enumerate(info_lines):
                    text_surface = font_small.render(line, True, (255, 255, 255))
                    text_rect = text_surface.get_rect(center=(int(center_x), int(center_y) + i * 12 - 18))
                    self.screen.blit(text_surface, text_rect)
        
        # Draw individual point masses
        for point in self.points:
            # Size based on mass (but keep it visible)
            radius = max(3, int(5 + point.mass * 2))
            
            # Color intensity based on speed for visual effect
            speed = math.sqrt(point.vx**2 + point.vy**2)
            intensity = min(255, int(100 + speed * 0.5))
            color = (intensity, intensity // 2, 255)
            
            # Draw the point
            pos = (int(point.x), int(point.y))
            pygame.draw.circle(self.screen, color, pos, radius)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, radius, 1)
        
        # Draw instructions
        font = pygame.font.Font(None, 18)
        instructions = [
            "Soft Body Physics with Verlet Integration:",
            "TAB - Pause/Unpause simulation",
            "PERIOD - Step one frame (when paused)",
            "R - Reset simulation",
            "SPACE - Add point at mouse",
            "C - Add circle at mouse",
            "P - Toggle pressure physics",
            "+/- - Increase/Decrease pressure",
            "S - Toggle spring physics",
            "Q/A - Increase/Decrease spring stiffness",
            "W/Z - Increase/Decrease spring damping",
            "D - Toggle drag physics",
            "E/X - Increase/Decrease drag coefficient",
            "T - Toggle drag type (linear/quadratic)",
            f"Current drag: {self.global_drag_coefficient:.3f} ({self.drag_type})",
            "ESC - Exit"
        ]
        
        for i, text in enumerate(instructions):
            color = (200, 200, 200) if i == 0 else (150, 150, 150)
            if "Current drag" in text:
                color = (255, 255, 0) if self.drag_enabled else (100, 100, 100)
            surface = font.render(text, True, color)
            self.screen.blit(surface, (10, 10 + i * 25))
        
        # Display pause indicator
        if self.paused:
            pause_font = pygame.font.Font(None, 48)
            pause_surface = pause_font.render("PAUSED", True, (255, 255, 0))
            pause_rect = pause_surface.get_rect(center=(self.width // 2, 50))
            # Add background for better visibility
            background_rect = pause_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, (0, 0, 0), background_rect)
            pygame.draw.rect(self.screen, (255, 255, 0), background_rect, 2)
            self.screen.blit(pause_surface, pause_rect)
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main simulation loop."""
        running = True
        
        while running:
            running = self.handle_events()
            
            # Only update physics if not paused, or if stepping one frame
            should_update_physics = not self.paused or self.step_next_frame
            
            if should_update_physics:
                # Run multiple small physics steps for stability with Verlet integration
                for _ in range(1):  # Reduced from 10 substeps due to Verlet stability
                    self.update_physics()
                
                # Reset step flag after processing the frame
                if self.step_next_frame:
                    self.step_next_frame = False
            
            self.render()
            self.clock.tick(120) #fps
        
        pygame.quit()
        sys.exit()


# Example usage and basic test
if __name__ == "__main__":
    # Create and run the visualization
    simulation = PhysicsSimulation()
    simulation.run()

