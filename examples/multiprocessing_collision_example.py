#!/usr/bin/env python3
"""
Example demonstrating multiprocessing collision detection in softcells.
"""

import time
from softcells.simulation.physics_engine import PhysicsEngine
from softcells.shapes import CircleShape


def benchmark_collision_performance():
    """Compare serial vs parallel collision detection performance."""
    
    # Test parameters
    num_shapes = 20
    radius = 30
    world_width = 1000
    world_height = 800
    
    print(f"Benchmarking collision detection with {num_shapes} shapes...")
    
    # Test 1: Serial collision detection
    print("\n=== Serial Collision Detection ===")
    engine_serial = PhysicsEngine(world_width, world_height, enable_collision_mp=False)
    
    # Add multiple circle shapes
    for i in range(num_shapes):
        x = 100 + (i % 8) * 120
        y = 100 + (i // 8) * 120
        circle = engine_serial.add_circle_shape(x, y, radius, num_points=15)
        
        # Add some initial velocity for interesting collisions
        for point in circle.get_points():
            point.vx = (i % 3 - 1) * 50
            point.vy = (i % 2) * 30
    
    # Run simulation and time it
    start_time = time.time()
    for _ in range(100):  # 100 physics steps
        engine_serial.step()
    serial_time = time.time() - start_time
    print(f"Serial time: {serial_time:.3f} seconds")
    
    # Test 2: Parallel collision detection
    print("\n=== Parallel Collision Detection ===")
    engine_parallel = PhysicsEngine(world_width, world_height, enable_collision_mp=True, num_processes=4)
    
    # Add the same shapes
    for i in range(num_shapes):
        x = 100 + (i % 8) * 120
        y = 100 + (i // 8) * 120
        circle = engine_parallel.add_circle_shape(x, y, radius, num_points=15)
        
        # Add the same initial velocity
        for point in circle.get_points():
            point.vx = (i % 3 - 1) * 50
            point.vy = (i % 2) * 30
    
    # Run simulation and time it
    start_time = time.time()
    for _ in range(100):  # 100 physics steps
        engine_parallel.step()
    parallel_time = time.time() - start_time
    print(f"Parallel time: {parallel_time:.3f} seconds")
    
    # Calculate speedup
    if parallel_time > 0:
        speedup = serial_time / parallel_time
        print(f"\nSpeedup: {speedup:.2f}x")
        if speedup > 1:
            print("✅ Parallel processing is faster!")
        else:
            print("⚠️  Serial processing is faster (overhead too high for this problem size)")
    
    # Clean up
    del engine_serial
    del engine_parallel


def demonstrate_dynamic_switching():
    """Show how to dynamically enable/disable multiprocessing."""
    
    print("\n=== Dynamic Multiprocessing Control ===")
    
    engine = PhysicsEngine(1000, 800, enable_collision_mp=False)
    
    # Add some shapes
    for i in range(5):
        engine.add_circle_shape(200 + i * 100, 200, 40, num_points=12)
    
    print("Starting with serial collision detection...")
    engine.step()
    
    # Enable multiprocessing
    print("Enabling multiprocessing...")
    engine.enable_collision_multiprocessing(True, num_processes=2)
    engine.step()
    
    # Disable it again
    print("Disabling multiprocessing...")
    engine.enable_collision_multiprocessing(False)
    engine.step()
    
    print("✅ Dynamic switching successful!")


if __name__ == "__main__":
    print("SoftCells Multiprocessing Collision Detection Demo")
    print("=" * 50)
    
    try:
        benchmark_collision_performance()
        demonstrate_dynamic_switching()
        
        print("\n🎉 All tests completed successfully!")
        print("\nTips for optimal performance:")
        print("- Use multiprocessing only when you have 10+ shapes")
        print("- The overhead of multiprocessing may not be worth it for simple scenes")
        print("- Best results with 20+ complex shapes with many collision points")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
