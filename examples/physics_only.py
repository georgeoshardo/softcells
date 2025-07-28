#!/usr/bin/env python3
"""
Example demonstrating the pure physics engine without any visualization.
This shows how the physics can run completely independently of pygame or rendering.
"""

import time

from softcells import PhysicsEngine


def main():
    """Run a headless physics simulation and output data."""
    print("Starting headless physics simulation...")
    
    # Create physics engine
    engine = PhysicsEngine(world_width=800, world_height=600)
    
    # Add some objects
    engine.add_point(400, 100, mass=2.0)
    engine.add_circle_shape(300, 200, 60, num_points=16)
    engine.add_circle_shape(500, 200, 40, num_points=12)
    
    # Set up some interesting physics parameters
    engine.set_gravity(50.0)
    engine.set_drag_properties(0.1, "quadratic", True)
    
    # Run simulation for a specified time
    simulation_time = 5.0  # seconds
    steps_per_second = 60
    total_steps = int(simulation_time * steps_per_second)
    
    print(f"Running {total_steps} steps over {simulation_time} seconds...")
    
    # Store some data points for analysis
    data_points = []
    
    start_time = time.time()
    for step in range(total_steps):
        # Update physics
        engine.step()
        
        # Sample data every 10 steps
        if step % 10 == 0:
            state = engine.get_simulation_state()
            
            # Extract some interesting metrics
            total_kinetic_energy = 0
            for point_data in state['points']:
                v_squared = point_data['vx']**2 + point_data['vy']**2
                total_kinetic_energy += 0.5 * point_data['mass'] * v_squared
            
            for shape_data in state['shapes']:
                for point_data in shape_data['points']:
                    v_squared = point_data['vx']**2 + point_data['vy']**2
                    total_kinetic_energy += 0.5 * point_data['mass'] * v_squared
            
            data_point = {
                'time': step / steps_per_second,
                'step': step,
                'total_kinetic_energy': total_kinetic_energy,
                'num_points': len(state['points']),
                'num_shapes': len(state['shapes']),
                'gravity': state['physics_params']['gravity']
            }
            data_points.append(data_point)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Simulation completed in {elapsed_time:.3f} seconds")
    print(f"Performance: {total_steps/elapsed_time:.1f} steps/second")
    
    # Output final state
    final_state = engine.get_simulation_state()
    
    print("\n=== FINAL SIMULATION STATE ===")
    print(f"Individual points: {len(final_state['points'])}")
    print(f"Shapes: {len(final_state['shapes'])}")
    
    if final_state['points']:
        print("\nIndividual Points:")
        for i, point in enumerate(final_state['points']):
            print(f"  Point {i}: pos=({point['x']:.1f}, {point['y']:.1f}), "
                  f"vel=({point['vx']:.2f}, {point['vy']:.2f}), mass={point['mass']}")
    
    for i, shape in enumerate(final_state['shapes']):
        print(f"\nShape {i}:")
        print(f"  Points: {len(shape['points'])}")
        print(f"  Pressure: {shape['pressure_amount']:.1f} (enabled: {shape['pressure_enabled']})")
        print(f"  Springs: {shape['spring_stiffness']:.0f} stiffness (enabled: {shape['springs_enabled']})")
        print(f"  Volume: {shape['current_volume']:.1f}")
        
        # Calculate center of mass
        total_mass = sum(p['mass'] for p in shape['points'])
        center_x = sum(p['x'] * p['mass'] for p in shape['points']) / total_mass
        center_y = sum(p['y'] * p['mass'] for p in shape['points']) / total_mass
        print(f"  Center of mass: ({center_x:.1f}, {center_y:.1f})")
    
        
    # Display energy over time
    print("\n=== ENERGY ANALYSIS ===")
    initial_ke = data_points[0]['total_kinetic_energy'] if data_points else 0
    final_ke = data_points[-1]['total_kinetic_energy'] if data_points else 0
    print(f"Initial kinetic energy: {initial_ke:.2f}")
    print(f"Final kinetic energy: {final_ke:.2f}")
    print(f"Energy change: {final_ke - initial_ke:.2f} ({((final_ke - initial_ke) / initial_ke * 100 if initial_ke > 0 else 0):.1f}%)")


if __name__ == "__main__":
    main()
