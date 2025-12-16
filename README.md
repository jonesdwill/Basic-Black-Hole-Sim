# BasicBlackHoleSim

A project to learn Julia and some basic GR theory, to simulate and render black hole properties.  

## Features
### Physics Module

- **Schwarzschild Metric:**     Static, non-rotating black hole.
- **Kerr Metric:**              Rotating black hole with frame-dragging effects.
- **Geodesic Kerr:**            High-precision solution of mass-less photon geodesics.

### Experiments  
- **Photon Streamlines:**       Plotting photon paths around black holes.
- **Shadow:**                   Simple Black Hole shadow and spacetime rendering.
- **Redshift Calculations:**    Doppler plotting of accretion disc.
- **Magnetic Field:**           Basic magnetic field curvature past a black hole.

## Accretion Disc Rendering

To render, split the process in two parts to make any debugging a bit easier:
       1. **Compute:** photon paths backwards in time from an observer screen and calculate their redshift.
       2. **Render:** plot the observed accretion disc using computed redshifts, add some noise channels, and animate it on the disc.

![Accretion Disk Animation](scripts/renderer/black_hole_1080p.gif)

## Project Structure

```
BasicBlackHoleSim/
├── src/
│   ├── BasicBlackHoleSim.jl    # Main module
│   ├── Constants.jl            # Physical constants
│   ├── Physics.jl              # Governing equations
│   ├── Solvers.jl              # ODE solver wrapper
│   └── Utils.jl                # General utility functions for orbits, initial states, etc.
├── scripts/
│   ├── 1-schwarzchild/        
│   ├── 2-kerr/                 
│   ├── 3-geodesic-kerr/       
│   ├── 4-photon-streamlines/   
│   ├── 5-shadow/               
│   ├── 6-redshift/             
│   ├── 7-magnetifc-field/      
│   └── render/                 
├── test/                       # Unit tests
├── Project.toml                # Julia project file
├── Manifest.toml               # Dependencies
└── README.md
```

## Installation

### Prerequisites
- Julia 1.9 or later

### Key Dependencies
The project environment includes all necessary packages, but the core dependencies are:
- `DifferentialEquations.jl`                    for geodesic integration.
- `Images.jl`, `FileIO.jl`, `ImageFiltering.jl` for image processing and output.
- `JLD2.jl`                                     for saving and loading computed data.
- `CoherentNoise.jl`                            for procedural turbulence generation.
- `ProgressMeter.jl` f                          or tracking computations.


### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd BasicBlackHoleSim
   ```

2. Activate the project and install dependencies:
   ```bash
   julia --project=. -e "using Pkg; Pkg.instantiate()"
   ```

## Usage

To run a render, edit `scripts/renderer/Render.jl` and set:
```julia
RUN_PHYSICS = true
RUN_RENDER  = true
```

Then execute (useful to have auto threading for efficiency):
```bash
julia -t auto scripts/renderer/Render.jl
```

All settings are controlled by the `CFG` constant in `scripts/renderer/Render.jl`.

- `width`, `height`: Output resolution in pixels.
- `M`: Mass of the black hole (typically `1.0`).
- `a_star`: Spin of the black hole, from `0.0` (non-rotating) to `0.99` (near-maximal rotation).
- `fov_y`: Vertical Field of View in degrees.
- `duration`: Animation duration in seconds.
- `fps`: Frames per second.
- `data_path`: File path for saving/loading the physics data from Step 1.
- `output_path`: Final output path for the rendered GIF or image.


#### Other Scripts:

1. **Schwarzschild Basics** (`1-schwarzchild/`)
   - `schwarz_run_single.jl`: Single photon trajectory
   - `schwarz_run_multi.jl`: Multiple photon paths

![Schwarzchild Orbit Comparison](scripts/1-schwarzchild/schwarzchild_orbits_comparison.png)

2. **Kerr Basics** (`2-kerr/`)
   - `kerr_run_single.jl`: Single photon trajectory
   - `kerr_run_multi.jl`: Multiple photon paths

![Kerr Orbit Comparison](scripts/2-kerr/kerr_orbits_comparison.png)

3. **Kerr Geodesics** (`3-geodesic-kerr/`)
   - `run_geodesic_kerr.jl`: Multiple geodesics
   - `kerr_geodesic_plunge.jl`: Plunging orbits

![Geodesic Kerr Orbit Comparison](scripts/3-geodesic-kerr/kerr_geodesic_comparison.png)

4. **Photon Streamlines** (`4-photon-streamlines/`)
   - `raytrace_streamlines.jl`: Streamline visualisation
   - `three_photon_raytrace.jl`: Multi-photon tracing

![Photon Streamlines](scripts/4-photon-streamlines/raytrace_streamlines.png)

5. **Black Hole Shadow** (`5-shadow/`)
   - `kerr_black_hole_shadow.jl`: Shadow rendering

![Black Hole Shadow](scripts/5-shadow/kerr_black_hole_shadow.png)

6. **Redshift Analysis** (`6-redshift/`)
   - `doppler-accretion_disc.jl`: Doppler effects

![Redshift Analysis](scripts/6-redshift/kerr_doppler.png)

7. **Magnetic Fields** (`7-magnetifc-field/`)
   - `magnetic_field.jl`: Basic field models

![Magnetic Field](scripts/7-magnetic-field/black_hole_magnetic_field.png)

## Testing

Run tests with:
```bash
julia --project=. test/runtests.jl
```

## References

### Core Physics
- Kerr, R. P. (1963). Gravitational field of a spinning mass as an example of algebraically special metrics. *Physical Review Letters*, 11(5), 237–238.
- Chandrasekhar, S. (1992). *The mathematical theory of black holes*. Oxford University Press.

### Black Hole Visualization and Raytracing
- James, O., von Tunzelmann, E., Franklin, P., & Thorne, K. S. (2015). Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar. *Classical and Quantum Gravity*, 32(6), 065001.
- Luminet, J. P. (1979). Image of a spherical black hole with thin accretion disk. *Astronomy and Astrophysics*, 75, 228–235.
- Akiyama, K., et al. (2019). First M87 Event Horizon Telescope Results. I. The Shadow of the Supermassive Black Hole. *The Astrophysical Journal Letters*, 875(1), L1.

### Accretion Disks
- Shakura, N. I., & Sunyaev, R. A. (1973). Black holes in binary systems. Observational appearance. *Astronomy and Astrophysics*, 24, 337–355.

### Numerical Methods
- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving ordinary differential equations I: Nonstiff problems*. Springer.
- DifferentialEquations.jl documentation and algorithms for geodesic integration.

### Computer Graphics
- Procedural noise and rendering techniques adapted from Perlin, K. (1985). An image synthesizer. *Computer Graphics*, 19(3), 287–296.
