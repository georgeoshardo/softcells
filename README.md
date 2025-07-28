# SoftCells - Soft Body Physics Simulation

A comprehensive soft body physics simulation system built with Python, featuring real-time visualization and interactive controls.

## Features

- **Advanced Physics Engine**: Uses Verlet integration for stable and accurate physics simulation
- **Spring-Mass Systems**: Flexible spring connections between point masses with configurable stiffness and damping
- **Pressure Physics**: Volume-preserving pressure forces to maintain shape integrity
- **Collision Detection**: Efficient broad-phase and narrow-phase collision detection between soft bodies
- **Real-time Visualization**: Smooth Pygame-based rendering with visual feedback for physics properties
- **Interactive Controls**: Live adjustment of physics parameters during simulation

## Architecture

The project is organized into a clean, modular structure:

```
softcells/
├── core/              # Core physics components
│   ├── point_mass.py  # Point mass with Verlet integration
│   └── spring.py      # Spring connections between points
├── shapes/            # Shape implementations
│   ├── base_shape.py  # Base shape class with common functionality
│   └── circle_shape.py # Circular soft body implementation
├── simulation/        # Simulation engine
│   ├── physics_simulation.py # Main simulation engine
│   └── collision_handler.py  # Collision detection and resolution
├── utils/             # Utility functions
│   └── geometry.py    # Geometric calculations (optimized with Numba)
└── config.py          # Configuration constants and settings
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd softcells
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the simulation:
```bash
python main.py
```

## Controls

### Simulation Control
- **TAB**: Pause/Unpause simulation
- **PERIOD** (`.`): Step one frame when paused
- **R**: Reset simulation
- **ESC**: Exit

### Adding Objects
- **SPACE**: Add point mass at mouse position
- **C**: Add circle shape at mouse position

### Physics Parameters
- **P**: Toggle pressure physics on/off
- **+/-**: Increase/Decrease internal pressure
- **S**: Toggle spring physics on/off
- **Q/A**: Increase/Decrease spring stiffness
- **W/Z**: Increase/Decrease spring damping
- **D**: Toggle drag physics on/off
- **E/X**: Increase/Decrease drag coefficient
- **T**: Toggle drag type (linear/quadratic)

## Physics Concepts

### Verlet Integration
The simulation uses Verlet integration for numerical stability and energy conservation:
```
x(t+Δt) = 2x(t) - x(t-Δt) + a(t)Δt²
```

### Spring Forces
Springs connect point masses using Hooke's law with velocity damping:
```
F = k(L - L₀) + c(v₁-v₂)·n̂
```

### Pressure Forces
Internal gas pressure maintains volume using the ideal gas law:
```
P = P₀ × V₀ / V
```

### Collision Resolution
Collisions are resolved using impulse-based methods with position correction to prevent overlap.

## Customization

### Creating Custom Shapes
```python
from softcells.core import PointMass
from softcells.shapes import Shape

# Create a custom triangular shape
triangle = Shape()
triangle.add_point(PointMass(100, 100))
triangle.add_point(PointMass(150, 200))
triangle.add_point(PointMass(50, 200))
triangle.create_springs()
```

### Adjusting Physics Parameters
```python
# Configure a bouncy, high-pressure circle
circle = CircleShape(400, 300, 60, 
                    pressure=500,
                    spring_stiffness=200,
                    drag_coefficient=1.0)
```

## Performance

The simulation is optimized for real-time performance:
- Numba JIT compilation for geometry calculations
- Efficient broad-phase collision detection using bounding boxes
- Configurable time step and substeps for stability vs. performance trade-offs

## Dependencies

- **pygame** (≥2.1.0): Graphics and user interface
- **numba** (≥0.56.0): JIT compilation for performance
- **numpy** (≥1.21.0): Numerical computations

## License

This project is open source. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Academic References

The physics implementation is based on established soft body simulation techniques:
- Verlet integration for stable numerical methods
- Spring-mass systems for deformable objects
- Pressure-based volume preservation
- Impulse-based collision resolution 