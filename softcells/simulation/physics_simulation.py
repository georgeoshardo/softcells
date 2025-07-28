"""
Main physics simulation engine with Pygame rendering.
"""

import math
import pygame
import sys

from ..core import PointMass
from ..shapes import CircleShape
from ..config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS, DEFAULT_DT, DEFAULT_GRAVITY,
    DEFAULT_GLOBAL_DRAG_COEFFICIENT, DEFAULT_DRAG_TYPE,
    DEFAULT_BACKGROUND_COLOR, DEFAULT_POINT_COLOR, DEFAULT_TRAIL_COLOR,
    MAX_TRAIL_LENGTH, GLOBAL_PRESSURE_AMOUNT
)
from .collision_handler import CollisionHandler


class PhysicsSimulation:
    """
    A generic Pygame-based visualization of physics simulation.
    Supports both individual point masses and shapes composed of point masses.
    """
    
    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        """Initialize the simulation window and physics world."""
        pygame.init()
        
        # Window setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Soft Body Physics Simulation")
        
        # Physics setup with improved stability from Verlet integration
        self.clock = pygame.time.Clock()
        self.dt = DEFAULT_DT
        self.gravity = DEFAULT_GRAVITY
        
        # Drag physics parameters
        self.global_drag_coefficient = DEFAULT_GLOBAL_DRAG_COEFFICIENT
        self.drag_type = DEFAULT_DRAG_TYPE
        self.drag_enabled = True
        
        # Simulation control
        self.paused = False
        self.step_next_frame = False
        
        # Create physics objects
        self.points = []  # Individual point masses
        self.shapes = []  # Shape objects (which contain point masses)
        
        # Collision handling
        self.collision_handler = CollisionHandler()
        
        # Initialize scene
        self.create_initial_scene()
        
        # Colors
        self.background_color = DEFAULT_BACKGROUND_COLOR
        self.point_color = DEFAULT_POINT_COLOR
        self.trail_color = DEFAULT_TRAIL_COLOR
        
        # Trail system for visual effect (for individual points only)
        self.point_trails = [[] for _ in self.points]
        self.max_trail_length = MAX_TRAIL_LENGTH
    
    def create_initial_scene(self):
        """Create initial physics objects for the simulation."""
        # Create some circle shapes with different spring and pressure settings
        # Stiff circle - high spring stiffness and pressure
        circle1 = CircleShape(500, 100, 50, num_points=20, point_mass=1.0, pressure=GLOBAL_PRESSURE_AMOUNT,
                             spring_stiffness=1150.0, spring_damping=10.0, 
                             drag_coefficient=self.global_drag_coefficient, drag_type=self.drag_type)
        circle1.set_color((150, 255, 150))  # Green - Stiff
        self.shapes.append(circle1)
    
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
                    new_circle = CircleShape(mouse_x, mouse_y, 50, num_points=50, point_mass=1.0, pressure=GLOBAL_PRESSURE_AMOUNT,
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

        # Handle shape-to-shape collisions
        self.collision_handler.handle_collisions(self.shapes)
    
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
            self.clock.tick(DEFAULT_FPS)
        
        pygame.quit()
        sys.exit() 