"""
Collision detection and resolution for soft body physics.
"""

import math

from ..config import COLLISION_SLOP, COLLISION_CORRECTION_PERCENT, COLLISION_RESTITUTION


class CollisionHandler:
    """
    Handles collision detection and resolution between shapes.
    """
    
    def __init__(self):
        """Initialize the collision handler."""
        self.slop = COLLISION_SLOP
        self.correction_percent = COLLISION_CORRECTION_PERCENT
        self.restitution = COLLISION_RESTITUTION
    
    def handle_collisions(self, shapes):
        """
        Detect and resolve collisions between shapes.
        
        Args:
            shapes (list): List of Shape objects to check for collisions
        """
        num_shapes = len(shapes)
        if num_shapes < 2:
            return

        for i in range(num_shapes):
            for j in range(i + 1, num_shapes):
                shape_a = shapes[i]
                shape_b = shapes[j]

                # # --- BROAD PHASE: Bounding Box Check ---
                # min_ax, max_ax, min_ay, max_ay = shape_a._get_bounding_box()
                # min_bx, max_bx, min_by, max_by = shape_b._get_bounding_box()

                # # If the bounding boxes do not overlap, skip to the next pair
                # if max_ax < min_bx or min_ax > max_bx or max_ay < min_by or min_ay > max_by:
                #     continue  # Not colliding, so no need for the expensive check

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
        
        Args:
            colliding_point: The point mass that is colliding
            shape: The shape that the point is colliding with
        """
        edge_p1, edge_p2, t = shape.find_closest_edge(colliding_point)

        if not edge_p1:
            return
        
        # Get a point guaranteed to be outside the shape's bounding box.
        (min_x, max_x, min_y, max_y), (min_wx, max_wx, min_wy, max_wy) = shape._get_bounding_box_with_windings()

        
        colliding_point_in_shape_referential = (
            colliding_point.x + (min_wx - colliding_point.winding_x) * colliding_point.world_width,
            colliding_point.y + (min_wy - colliding_point.winding_y) * colliding_point.world_height
        )
        
        closest_x = (1-t) * edge_p1.x + t * edge_p2.x
        closest_y = (1-t) * edge_p1.y + t * edge_p2.y

        # --- 1. POSITION RESOLUTION ---
        # The penetration vector points from the edge to the colliding point (inward)
        penetration_vec_x = colliding_point_in_shape_referential[0] - closest_x
        penetration_vec_y = colliding_point_in_shape_referential[1] - closest_y
        penetration_depth = math.sqrt(penetration_vec_x**2 + penetration_vec_y**2)
        
        if penetration_depth < 0.001:
            return

        # The normal points INWARD into the shape.
        normal_x = penetration_vec_x / penetration_depth
        normal_y = penetration_vec_y / penetration_depth
        
        correction_depth = max(penetration_depth - self.slop, 0.0)
        if correction_depth == 0.0:
            return

        inv_mass_p = 1.0 / colliding_point.mass if colliding_point.mass > 0 else 0
        inv_mass_e1 = 1.0 / edge_p1.mass if edge_p1.mass > 0 else 0
        inv_mass_e2 = 1.0 / edge_p2.mass if edge_p2.mass > 0 else 0
        
        total_inv_mass = inv_mass_p + inv_mass_e1 * (1 - t) + inv_mass_e2 * t
        if total_inv_mass < 0.001:
            return

        move_dist = (correction_depth / total_inv_mass) * self.correction_percent
        
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

        # Impulse magnitude (will be positive since vel_along_normal is negative)
        impulse_j = -(1 + self.restitution) * vel_along_normal / total_inv_mass
        
        # Apply impulse to push the colliding point OUTWARD (-n)
        colliding_point.vx -= (impulse_j * inv_mass_p) * normal_x
        colliding_point.vy -= (impulse_j * inv_mass_p) * normal_y

        # Apply impulse to push the edge points INWARD (+n)
        edge_p1.vx += (impulse_j * inv_mass_e1 * (1 - t)) * normal_x
        edge_p1.vy += (impulse_j * inv_mass_e1 * (1 - t)) * normal_y
        edge_p2.vx += (impulse_j * inv_mass_e2 * t) * normal_x
        edge_p2.vy += (impulse_j * inv_mass_e2 * t) * normal_y 