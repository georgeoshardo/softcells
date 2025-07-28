"""
Simulation engine for the soft body physics system.
"""

from .physics_simulation import PhysicsSimulation
from .physics_engine import PhysicsEngine
from .collision_handler import CollisionHandler

__all__ = ['PhysicsSimulation', 'PhysicsEngine', 'CollisionHandler'] 