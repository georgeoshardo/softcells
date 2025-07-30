"""
Pygame-based visualization layer for the physics simulation.
This module handles all rendering and user interface elements.
"""

import math
import pygame
import sys

from ..config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS,
    DEFAULT_BACKGROUND_COLOR, DEFAULT_POINT_COLOR, DEFAULT_TRAIL_COLOR,
    MAX_TRAIL_LENGTH, GLOBAL_PRESSURE_AMOUNT
)
from ..simulation.physics_engine import PhysicsEngine


class SimulationVisualizer:
    """
    Pygame-based visualizer for the physics simulation.
    Handles rendering, user input, and visual effects.
    """
    
    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        """
        Initialize the visualization system.
        
        Args:
            width (int): Window width
            height (int): Window height
        """
        pygame.init()
        
        # Window setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Soft Body Physics Simulation")
        
        # Create the physics engine
        self.physics_engine = PhysicsEngine(world_width=width, world_height=height)
        
        # Rendering control
        self.clock = pygame.time.Clock()
        
        # FPS counter
        self.fps_font = pygame.font.SysFont("Arial", 18, bold=True)
        
        # Colors
        self.background_color = DEFAULT_BACKGROUND_COLOR
        self.point_color = DEFAULT_POINT_COLOR
        self.trail_color = DEFAULT_TRAIL_COLOR
        
        # Trail system for visual effect (for individual points only)
        self.point_trails = []
        self.max_trail_length = MAX_TRAIL_LENGTH
        
        # Visual effects control
        self.show_trails = True
        self.show_physics_info = False
        self.show_instructions = True
        self.show_bounding_boxes = False
        
        # Drag and drop functionality
        self.dragging = False
        self.dragged_point = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0
    
    def handle_events(self):
        """Handle pygame events and map them to physics operations."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_TAB:
                    # Toggle pause
                    self.physics_engine.paused = not self.physics_engine.paused
                elif event.key == pygame.K_PERIOD:
                    # Step one frame when paused
                    if self.physics_engine.paused:
                        self.physics_engine.step_next_frame = True
                elif event.key == pygame.K_r:
                    # Reset simulation
                    self.physics_engine.reset_simulation()
                    self._reset_trails()
                elif event.key == pygame.K_SPACE:
                    # Add a new point at mouse position
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.physics_engine.add_point(mouse_x, mouse_y, 1.0)
                    self.point_trails.append([])
                elif event.key == pygame.K_c:
                    # Add a new circle shape at mouse position
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    circle = self.physics_engine.add_circle_shape(
                        mouse_x, mouse_y, 50, 
                        num_points=30, 
                        point_mass=1.0, 
                        pressure=GLOBAL_PRESSURE_AMOUNT,
                        spring_stiffness=1150.0, 
                        spring_damping=10.0,
                        identity=0
                    )
                    circle.set_color((255, 255, 100))  # Yellow
                elif event.key == pygame.K_v:
                    # Add a new circle shape at mouse position
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    circle1, circle2 = self.physics_engine.add_cell_shape(
                        mouse_x, mouse_y, 20, 
                        num_points=25, 
                        point_mass=1.0, 
                        pressure=GLOBAL_PRESSURE_AMOUNT,
                        spring_stiffness=2150.0, 
                        spring_damping=20.0
                    )
                    circle1.set_color((255, 255, 100))  # Yellow
                    circle2.set_color((255, 100, 100))  # Red

                elif event.key == pygame.K_p:
                    # Toggle pressure physics for all shapes
                    self.physics_engine.toggle_pressure_for_all_shapes()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Increase pressure for all shapes
                    for shape in self.physics_engine.shapes:
                        shape.set_pressure(shape.pressure_amount * 1.2)
                elif event.key == pygame.K_MINUS:
                    # Decrease pressure for all shapes
                    for shape in self.physics_engine.shapes:
                        shape.set_pressure(shape.pressure_amount * 0.8)
                elif event.key == pygame.K_s:
                    # Toggle spring physics for all shapes
                    self.physics_engine.toggle_springs_for_all_shapes()
                elif event.key == pygame.K_q:
                    # Increase spring stiffness
                    for shape in self.physics_engine.shapes:
                        new_stiffness = shape.spring_stiffness * 1.2
                        shape.set_spring_properties(new_stiffness, shape.spring_damping)
                elif event.key == pygame.K_a:
                    # Decrease spring stiffness
                    for shape in self.physics_engine.shapes:
                        new_stiffness = shape.spring_stiffness * 0.8
                        shape.set_spring_properties(new_stiffness, shape.spring_damping)
                elif event.key == pygame.K_w:
                    # Increase spring damping
                    for shape in self.physics_engine.shapes:
                        new_damping = shape.spring_damping * 1.2
                        shape.set_spring_properties(shape.spring_stiffness, new_damping)
                elif event.key == pygame.K_z:
                    # Decrease spring damping
                    for shape in self.physics_engine.shapes:
                        new_damping = shape.spring_damping * 0.8
                        shape.set_spring_properties(shape.spring_stiffness, new_damping)
                elif event.key == pygame.K_d:
                    # Toggle drag physics for all objects
                    new_enabled = not self.physics_engine.drag_enabled
                    self.physics_engine.set_drag_properties(
                        self.physics_engine.global_drag_coefficient,
                        self.physics_engine.drag_type,
                        new_enabled
                    )
                elif event.key == pygame.K_e:
                    # Increase drag coefficient
                    new_coeff = self.physics_engine.global_drag_coefficient * 1.2
                    self.physics_engine.set_drag_properties(
                        new_coeff,
                        self.physics_engine.drag_type,
                        self.physics_engine.drag_enabled
                    )
                elif event.key == pygame.K_x:
                    # Decrease drag coefficient
                    new_coeff = self.physics_engine.global_drag_coefficient * 0.8
                    self.physics_engine.set_drag_properties(
                        new_coeff,
                        self.physics_engine.drag_type,
                        self.physics_engine.drag_enabled
                    )
                elif event.key == pygame.K_t:
                    # Toggle drag type between linear and quadratic
                    new_type = "quadratic" if self.physics_engine.drag_type == "linear" else "linear"
                    self.physics_engine.set_drag_properties(
                        self.physics_engine.global_drag_coefficient,
                        new_type,
                        self.physics_engine.drag_enabled
                    )
                elif event.key == pygame.K_h:
                    # Toggle instructions display
                    self.show_instructions = not self.show_instructions
                elif event.key == pygame.K_i:
                    # Toggle physics info display
                    self.show_physics_info = not self.show_physics_info
                elif event.key == pygame.K_l:
                    # Toggle trail display
                    self.show_trails = not self.show_trails
                elif event.key == pygame.K_b:
                    # Add a new circle shape at mouse position
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    circle = self.physics_engine.add_circle_shape(
                        mouse_x, mouse_y, 70, 
                        num_points=30, 
                        point_mass=1.0, 
                        pressure=GLOBAL_PRESSURE_AMOUNT,
                        spring_stiffness=1150.0, 
                        spring_damping=10.0,
                        identity=1
                    )
                    circle.set_color((255, 100, 100))  # Red
                    # Toggle bounding box display
                    # self.show_bounding_boxes = not self.show_bounding_boxes
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_x, mouse_y = event.pos
                    clicked_point = self._get_point_at_position(mouse_x, mouse_y)
                    
                    if clicked_point is not None:
                        # Start dragging the point
                        self.dragging = True
                        self.dragged_point = clicked_point
                        
                        # Calculate offset from mouse to point center
                        self.drag_offset_x = mouse_x - clicked_point.x_world
                        self.drag_offset_y = mouse_y - clicked_point.y_world
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    # Stop dragging
                    self.dragging = False
                    self.dragged_point = None
                    self.drag_offset_x = 0
                    self.drag_offset_y = 0
            
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging and self.dragged_point is not None:
                    # Update the dragged point's position
                    mouse_x, mouse_y = event.pos
                    
                    # Calculate new world position accounting for the offset
                    new_world_x = mouse_x - self.drag_offset_x
                    new_world_y = mouse_y - self.drag_offset_y
                    
                    # Convert world coordinates to point's coordinate system considering winding numbers
                    # Mouse coordinates are always in world space, but the point may be in a different
                    # winding cell due to periodic boundaries
                    current_winding_x = self.dragged_point.winding_x
                    current_winding_y = self.dragged_point.winding_y
                    
                    # Calculate the point's new position in its current winding cell
                    # new_x = new_world_x + winding_offset_x
                    new_x = new_world_x + current_winding_x * self.dragged_point.world_width
                    new_y = new_world_y + current_winding_y * self.dragged_point.world_height
                    
                    # Update both current and previous positions to avoid velocity spikes
                    self.dragged_point.x = new_x
                    self.dragged_point.y = new_y
                    self.dragged_point.prev_x = new_x
                    self.dragged_point.prev_y = new_y
                    
                    # Reset velocity to prevent sudden movements
                    self.dragged_point.vx = 0.0
                    self.dragged_point.vy = 0.0
                    
                    # Mark position as dirty for cached values
                    self.dragged_point._position_dirty = True
        
        return True
    
    def _get_point_at_position(self, mouse_x, mouse_y):
        """
        Find the point (individual or in shape) at the given mouse position.
        Returns the closest point within a reasonable distance.
        
        Args:
            mouse_x (int): Mouse X coordinate
            mouse_y (int): Mouse Y coordinate
            
        Returns:
            PointMass or None: The point at the position, or None if no point is close enough
        """
        closest_point = None
        closest_distance = float('inf')
        click_radius = 15  # Maximum distance to consider a click
        
        # Check individual points
        for point in self.physics_engine.points:
            dx = mouse_x - point.x_world
            dy = mouse_y - point.y_world
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < click_radius and distance < closest_distance:
                closest_distance = distance
                closest_point = point
        
        # Check points in shapes
        for shape in self.physics_engine.shapes:
            for point in shape.points:
                dx = mouse_x - point.x_world
                dy = mouse_y - point.y_world
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < click_radius and distance < closest_distance:
                    closest_distance = distance
                    closest_point = point
        
        return closest_point
    
    def _reset_trails(self):
        """Reset all visual trails."""
        self.point_trails = [[] for _ in self.physics_engine.points]
    
    def _update_trails(self):
        """Update visual trails for individual points."""
        # Ensure we have the right number of trails
        while len(self.point_trails) < len(self.physics_engine.points):
            self.point_trails.append([])
        while len(self.point_trails) > len(self.physics_engine.points):
            self.point_trails.pop()
        
        # Update trail positions
        for i, point in enumerate(self.physics_engine.points):
            if i < len(self.point_trails):
                self.point_trails[i].append((int(point.x), int(point.y)))
                if len(self.point_trails[i]) > self.max_trail_length:
                    self.point_trails[i].pop(0)
    
    def _render_trails(self):
        """Render visual trails for individual points."""
        if not self.show_trails:
            return
            
        for trail in self.point_trails:
            if len(trail) > 1:
                for i in range(len(trail) - 1):
                    start_pos = trail[i]
                    end_pos = trail[i + 1]
                    pygame.draw.line(self.screen, self.trail_color, start_pos, end_pos, 1)
    
    def _render_shapes(self):
        """Render all shapes using their render methods."""
        for shape in self.physics_engine.shapes:
            self._render_shape(shape)
            
            # Draw bounding boxes if enabled
            self._render_bounding_box(shape)
            
            # Draw physics indicators if enabled
            if self.show_physics_info and len(shape.points) > 0:
                self._render_shape_info(shape)
    
    def _render_shape(self, shape):
        """
        Render a single shape by drawing springs and points.
        This is a decoupled version of the shape's render method.
        """
        if len(shape.points) < 2:
            return
        
        # Draw springs with color based on stretch
        if hasattr(shape, 'springs_enabled') and shape.springs_enabled and len(shape.springs) > 0:
            for spring in shape.springs:
                if spring.point1.winding_x != spring.point2.winding_x or \
                   spring.point1.winding_y != spring.point2.winding_y:
                    continue
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
                pos1 = (int(spring.point1.x_world), int(spring.point1.y_world))
                pos2 = (int(spring.point2.x_world), int(spring.point2.y_world))

                pygame.draw.line(self.screen, color, pos1, pos2, shape.line_width)
        else:
            # Draw basic lines if springs are disabled
            positions = [(int(point.x_world), int(point.y_world)) for point in shape.points]
            for i in range(len(positions)):
                start_pos = positions[i]
                end_pos = positions[(i + 1) % len(positions)]
                pygame.draw.line(self.screen, shape.color, start_pos, end_pos, shape.line_width)
        
        # Draw the individual points
        for point in shape.points:
            radius = max(2, int(3 + point.mass))
            
            # Check if this is the dragged point
            if self.dragging and point is self.dragged_point:
                # Highlight the dragged point
                color = (255, 255, 100)  # Bright yellow
                radius += 3  # Make it larger
                outline_color = (255, 255, 255)
                outline_width = 2
            else:
                color = shape.color
                outline_color = (255, 255, 255)
                outline_width = 1
            
            pos = (int(point.x_world), int(point.y_world))
            pygame.draw.circle(self.screen, color, pos, radius)
            pygame.draw.circle(self.screen, outline_color, pos, radius, outline_width)
    
    def _render_bounding_box(self, shape):
        """Render the bounding box of a shape."""
        if not self.show_bounding_boxes or len(shape.points) == 0:
            return
        
        # Get bounding box coordinates using world coordinates (for PBC)
        bbox_result = shape._get_bounding_box_world()
        
        # Don't draw bounding box if shape crosses periodic boundaries
        if bbox_result is None:
            return
            
        min_x, max_x, min_y, max_y = bbox_result
        
        # Convert to screen coordinates (assuming shapes use world coordinates)
        rect_x = int(min_x)
        rect_y = int(min_y)
        rect_width = int(max_x - min_x)
        rect_height = int(max_y - min_y)
        
        # Draw bounding box as a rectangle outline
        # Use a semi-transparent color
        bounding_box_color = (255, 0, 255)  # Magenta for visibility
        pygame.draw.rect(self.screen, bounding_box_color, 
                        (rect_x, rect_y, rect_width, rect_height), 2)
        
        # Optional: Draw corner markers for better visibility
        corner_size = 5
        corners = [
            (rect_x, rect_y),  # Top-left
            (rect_x + rect_width, rect_y),  # Top-right
            (rect_x, rect_y + rect_height),  # Bottom-left
            (rect_x + rect_width, rect_y + rect_height)  # Bottom-right
        ]
        
        for corner in corners:
            pygame.draw.circle(self.screen, bounding_box_color, corner, corner_size)
    
    def _render_shape_info(self, shape):
        """Render physics information for a shape."""
        # Calculate center of shape for text
        center_x = sum(p.x_world for p in shape.points) / len(shape.points)
        center_y = sum(p.y_world for p in shape.points) / len(shape.points)
        
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
    
    def _render_individual_points(self):
        """Render individual point masses."""
        for point in self.physics_engine.points:
            # Size based on mass (but keep it visible)
            radius = max(3, int(5 + point.mass * 2))
            
            # Check if this is the dragged point
            if self.dragging and point is self.dragged_point:
                # Highlight the dragged point
                color = (255, 255, 100)  # Bright yellow
                radius += 3  # Make it larger
                outline_color = (255, 255, 255)
                outline_width = 2
            else:
                # Color intensity based on speed for visual effect
                speed = math.sqrt(point.vx**2 + point.vy**2)
                intensity = min(255, int(100 + speed * 0.5))
                color = (intensity, intensity // 2, 255)
                outline_color = (255, 255, 255)
                outline_width = 1
            
            # Draw the point
            pos = (int(point.x_world), int(point.y_world))
            pygame.draw.circle(self.screen, color, pos, radius)
            pygame.draw.circle(self.screen, outline_color, pos, radius, outline_width)
    
    def _render_instructions(self):
        """Render instruction text."""
        if not self.show_instructions:
            return
            
        font = pygame.font.Font(None, 18)
        instructions = [
            "Soft Body Physics with Verlet Integration:",
            "TAB - Pause/Unpause simulation",
            "PERIOD - Step one frame (when paused)",
            "R - Reset simulation",
            "SPACE - Add point at mouse",
            "C - Add circle at mouse",
            "Left Click + Drag - Move points",
            "P - Toggle pressure physics",
            "+/- - Increase/Decrease pressure",
            "S - Toggle spring physics",
            "Q/A - Increase/Decrease spring stiffness",
            "W/Z - Increase/Decrease spring damping",
            "D - Toggle drag physics",
            "E/X - Increase/Decrease drag coefficient",
            "T - Toggle drag type (linear/quadratic)",
            f"Current drag: {self.physics_engine.global_drag_coefficient:.3f} ({self.physics_engine.drag_type})",
            "H - Toggle instructions",
            "I - Toggle physics info",
            "L - Toggle trails",
            "B - Toggle bounding boxes",
            "ESC - Exit"
        ]
        
        for i, text in enumerate(instructions):
            color = (200, 200, 200) if i == 0 else (150, 150, 150)
            if "Current drag" in text:
                color = (255, 255, 0) if self.physics_engine.drag_enabled else (100, 100, 100)
            surface = font.render(text, True, color)
            self.screen.blit(surface, (10, 10 + i * 25))
    
    def _render_pause_indicator(self):
        """Render pause indicator."""
        if self.physics_engine.paused:
            pause_font = pygame.font.Font(None, 48)
            pause_surface = pause_font.render("PAUSED", True, (255, 255, 0))
            pause_rect = pause_surface.get_rect(center=(self.width // 2, 50))
            # Add background for better visibility
            background_rect = pause_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, (0, 0, 0), background_rect)
            pygame.draw.rect(self.screen, (255, 255, 0), background_rect, 2)
            self.screen.blit(pause_surface, pause_rect)
    
    def _render_fps_counter(self):
        """Render FPS counter in the top right corner."""
        fps = str(int(self.clock.get_fps()))
        fps_text = self.fps_font.render(f"FPS: {fps}", True, pygame.Color("RED"))
        # Position in top right corner
        fps_rect = fps_text.get_rect()
        fps_rect.topright = (self.width - 10, 10)
        self.screen.blit(fps_text, fps_rect)
    
    def _render_object_counter(self):
        """Render object counter in the bottom right corner."""
        # Count individual points
        individual_points = len(self.physics_engine.points)
        
        # Count points in shapes
        shape_points = sum(len(shape.points) for shape in self.physics_engine.shapes)
        
        # Total points
        total_points = individual_points + shape_points
        
        # Number of shapes
        n_shapes = len(self.physics_engine.shapes)
        
        points_text = self.fps_font.render(f"Points: {total_points}", True, pygame.Color("WHITE"))
        shapes_text = self.fps_font.render(f"Shapes: {n_shapes}", True, pygame.Color("WHITE"))
        
        # Position in bottom right corner
        points_rect = points_text.get_rect()
        shapes_rect = shapes_text.get_rect()
        
        points_rect.bottomright = (self.width - 10, self.height - 30)
        shapes_rect.bottomright = (self.width - 10, self.height - 10)
        
        self.screen.blit(points_text, points_rect)
        self.screen.blit(shapes_text, shapes_rect)
    
    def render(self):
        """Render the complete simulation."""
        # Clear screen
        self.screen.fill(self.background_color)
        
        # Update and render trails
        self._update_trails()
        self._render_trails()
        
        # Draw shapes first (so they appear behind individual points)
        self._render_shapes()
        
        # Draw individual point masses
        self._render_individual_points()
        
        # Draw UI elements
        self._render_instructions()
        self._render_pause_indicator()
        self._render_fps_counter()
        self._render_object_counter()
        
        # Update display
        pygame.display.flip()
    
    def update_physics(self):
        """Update physics if not paused or if stepping one frame."""
        should_update_physics = not self.physics_engine.paused or self.physics_engine.step_next_frame
        
        if should_update_physics:
            # Run physics update
            self.physics_engine.step()
            
            # Reset step flag after processing the frame
            if self.physics_engine.step_next_frame:
                self.physics_engine.step_next_frame = False
    
    def run(self):
        """Main visualization loop."""
        running = True
        
        while running:
            running = self.handle_events()
            
            # Update physics
            self.update_physics()
            
            # Render everything
            self.render()
            
            # Control frame rate
            self.clock.tick(DEFAULT_FPS)
        
        pygame.quit()
        sys.exit()
