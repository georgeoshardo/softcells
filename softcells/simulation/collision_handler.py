"""
Collision detection and resolution for soft body physics.
"""

import math
import multiprocessing as mp
from functools import partial
from ..shapes import Shape
from ..config import COLLISION_SLOP, COLLISION_CORRECTION_PERCENT, COLLISION_RESTITUTION


def _process_shape_pair_collision(shape_pair_data, slop, correction_percent, restitution):
    """
    Worker function for parallel collision detection between a pair of shapes.
    
    Args:
        shape_pair_data: Tuple containing (shape_a_idx, shape_b_idx, shape_a, shape_b)
        slop: Collision slop parameter
        correction_percent: Collision correction percentage
        restitution: Collision restitution coefficient
        
    Returns:
        Tuple: (shape_a_idx, shape_b_idx, collisions_a_in_b, collisions_b_in_a)
               where collisions are lists of (point_idx, at_windings) tuples
    """
    shape_a_idx, shape_b_idx, shape_a, shape_b = shape_pair_data
    
    # Create a temporary collision handler with the same parameters
    temp_handler = CollisionHandler()
    temp_handler.slop = slop
    temp_handler.correction_percent = correction_percent
    temp_handler.restitution = restitution
    
    # Check if bounding boxes overlap first
    if not temp_handler.check_bbs_overlap(shape_a, shape_b):
        return None
    
    collisions_a_in_b = []
    collisions_b_in_a = []
    
    # Test points of A inside B
    for point_idx, point in enumerate(shape_a.get_points()):
        is_inside, at_windings = shape_b.is_point_inside(point)
        if is_inside:
            collisions_a_in_b.append((point_idx, at_windings))
    
    # Test points of B inside A
    for point_idx, point in enumerate(shape_b.get_points()):
        is_inside, at_windings = shape_a.is_point_inside(point)
        if is_inside:
            collisions_b_in_a.append((point_idx, at_windings))
    
    # Only return data if there are actual collisions
    if collisions_a_in_b or collisions_b_in_a:
        return (shape_a_idx, shape_b_idx, collisions_a_in_b, collisions_b_in_a)
    
    return None


class CollisionHandler:
    """
    Handles collision detection and resolution between shapes.
    """
    
    def __init__(self, enable_multiprocessing=False, num_processes=None):
        """Initialize the collision handler."""
        self.slop = COLLISION_SLOP
        self.correction_percent = COLLISION_CORRECTION_PERCENT
        self.restitution = COLLISION_RESTITUTION
        
        # Multiprocessing settings
        self.multiprocessing_enabled = enable_multiprocessing
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count()
        self.pool = None
        
        # Threshold for when to use multiprocessing (avoid overhead for small shape counts)
        self.mp_threshold = 10  # Only use multiprocessing if we have 10+ shapes

    def check_bbs_overlap(self, shape_a, shape_b):
        """
        Check if the bounding boxes of two shapes overlap, considering periodic boundaries.
        """
        (min_ax, max_ax, min_ay, max_ay), windings_a = shape_a._get_bounding_box_with_windings()
        (min_bx, max_bx, min_by, max_by), windings_b = shape_b._get_bounding_box_with_windings()

        world_width = shape_a.points[0].world_width
        world_height = shape_a.points[0].world_height

        for wa_x, wa_y in windings_a:
            for wb_x, wb_y in windings_b:
                dx = (wb_x - wa_x) * world_width
                dy = (wb_y - wa_y) * world_height

                # Check for overlap in the x-axis
                if (min_ax < max_bx - dx and max_ax > min_bx - dx and
                    min_ay < max_by - dy and max_ay > min_by - dy):
                    return True

        return False

    def handle_collisions(self, shapes):
        """
        Detect and resolve collisions between shapes.
        Automatically chooses between serial and parallel processing.
        
        Args:
            shapes (list): List of Shape objects to check for collisions
        """
        num_shapes = len(shapes)
        if num_shapes < 2:
            return

        # Use multiprocessing for large numbers of shapes
        if (self.multiprocessing_enabled and 
            num_shapes >= self.mp_threshold and 
            num_shapes * (num_shapes - 1) // 2 > 50):  # 50+ shape pairs
            self._handle_collisions_parallel(shapes)
        else:
            self._handle_collisions_serial(shapes)

    def _handle_collisions_serial(self, shapes):
        """Serial collision detection (original implementation)."""
        num_shapes = len(shapes)
        
        for i in range(num_shapes):
            shape_a = shapes[i]
            for j in range(i + 1, num_shapes):
                shape_b = shapes[j]

                bbs_overlap = self.check_bbs_overlap(shape_a, shape_b)
                if not bbs_overlap:
                    continue

                # Test points of A inside B
                for point in shape_a.get_points():
                    is_inside, at_windings = shape_b.is_point_inside(point)
                    if is_inside:
                        self.resolve_collision(point, shape_b, at_windings=at_windings)

                # Test points of B inside A
                for point in shape_b.get_points():
                    is_inside, at_windings = shape_a.is_point_inside(point)
                    if is_inside:
                        self.resolve_collision(point, shape_a, at_windings=at_windings)

    def _handle_collisions_parallel(self, shapes):
        """Parallel collision detection using multiprocessing."""
        # Create shape pairs for collision testing
        shape_pairs = []
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                shape_pairs.append((i, j, shapes[i], shapes[j]))

        # Process collisions in parallel
        if self.pool is None:
            self.pool = mp.Pool(self.num_processes)
        
        try:
            # Use partial to pass collision handler parameters
            collision_worker = partial(_process_shape_pair_collision, 
                                     slop=self.slop, 
                                     correction_percent=self.correction_percent,
                                     restitution=self.restitution)
            
            # Process all shape pairs in parallel
            collision_results = self.pool.map(collision_worker, shape_pairs)
            
            # Apply collision results back to the original shapes
            self._apply_collision_results(collision_results, shapes)
            
        except Exception as e:
            print(f"Multiprocessing collision detection failed: {e}")
            # Fallback to serial processing
            self._handle_collisions_serial(shapes)

    def _apply_collision_results(self, collision_results, shapes):
        """Apply the collision results from parallel processing back to shapes."""
        for result in collision_results:
            if result is None:
                continue
                
            shape_a_idx, shape_b_idx, collisions_a_in_b, collisions_b_in_a = result
            shape_a = shapes[shape_a_idx]
            shape_b = shapes[shape_b_idx]
            
            # Apply collisions of A's points inside B
            for point_idx, at_windings in collisions_a_in_b:
                point = shape_a.get_points()[point_idx]
                self.resolve_collision(point, shape_b, at_windings=at_windings)
            
            # Apply collisions of B's points inside A
            for point_idx, at_windings in collisions_b_in_a:
                point = shape_b.get_points()[point_idx]
                self.resolve_collision(point, shape_a, at_windings=at_windings)

    def enable_multiprocessing(self, enabled=True, num_processes=None):
        """Enable or disable multiprocessing for collision detection."""
        self.multiprocessing_enabled = enabled
        if num_processes is not None:
            self.num_processes = num_processes
        
        # Clean up existing pool if changing settings
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        """Clean up multiprocessing pool on destruction."""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()

    def resolve_collision(self, colliding_point, shape, at_windings):
        """
        Resolves a collision between a point and a shape.
        Moves points to resolve overlap and updates velocities for bounce.
        
        Args:
            colliding_point: The point mass that is colliding
            shape: The shape that the point is colliding with
        """
        edge_p1, edge_p2, t = shape.find_closest_edge(colliding_point, at_windings=at_windings)

        if not edge_p1:
            return

        winding_x, winding_y = at_windings
        
        colliding_point_in_shape_referential = (
            colliding_point.x + (winding_x - colliding_point.winding_x) * colliding_point.world_width,
            colliding_point.y + (winding_y - colliding_point.winding_y) * colliding_point.world_height
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