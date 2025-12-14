using Pkg

# Ensure we are in the correct project environment
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 

using BasicBlackHoleSim
using Plots
using BasicBlackHoleSim.Solvers: setup_problem, solve_orbit
using BasicBlackHoleSim.Utils: get_black_hole_parameters, get_initial_photon_state_celestial
using DifferentialEquations: DiscreteCallback, terminate!
using Base.Threads
using Printf

# ==========================================
#             HELPER FUNCTIONS
# ==========================================


"""
Determines pixel color based on where the photon escaped to.
"""
function get_sky_color(u_final)

    theta = u_final[3]
    phi   = u_final[4]

    # Normalise phi to [0, 2π]
    phi_norm = mod(phi, 2π)

    # Checkerboard grid size (in radians)
    grid_scale = 0.35 
    
    # distinct regions based on angle
    theta_idx = floor(Int, theta / grid_scale)
    phi_idx   = floor(Int, phi_norm / grid_scale)
    is_even = (theta_idx + phi_idx) % 2 == 0

    return is_even ? 0.9 : 0.4 
end

# ==========================================
#                CONFIG      
# ==========================================

# Big Spin Black Hole Params
M = 1.0
a_star = 0.99 
solver_params = (M, a_star) 
a = a_star * M
bh_params = get_black_hole_parameters(solver_params)

# Observer Params
r_obs = 50.0           # Distance from BH
theta_obs = 85 * π/180 # Just above edge-on view 

# Screen / Camera Setup
screen_width = 20.0
screen_height = 15.0
resolution_x = 200 * 2
resolution_y = 150 * 2

alpha_range = range(-screen_width/2, screen_width/2, length=resolution_x)
beta_range = range(-screen_height/2, screen_height/2, length=resolution_y)

image_lensed = zeros(resolution_y, resolution_x)

# ==========================================
#          THREADING LOOP
# ==========================================

t_max = 2.5 * r_obs 
tspan = (0.0, t_max)

# Callback to stop
function is_captured(u, t, integrator)
    r = u[2]

    # Terminate if photon crosses the horizon (with small buffer)
    return r < bh_params.rh + 1e-3
end
capture_callback = DiscreteCallback(is_captured, terminate!)

println("Starting Ray Tracing (Lensing Mode)...")
println("  Configuration: a*=$a_star, Threads=$(nthreads())")

progress_counter = Atomic{Int}(0)
total_pixels = resolution_x * resolution_y

# loop over each pixel
@threads for i in 1:resolution_x
    for j in 1:resolution_y
        alpha = alpha_range[i]
        beta = beta_range[j]

        # place ray away from black hole
        u0 = get_initial_photon_state_celestial(alpha, beta, r_obs, theta_obs, M, a)

        # Handle invalid starting conditions (e.g., inside horizon)
        if any(isnan, u0)
            image_lensed[j, i] = 0.0 
            continue
        end

        # Solve
        prob = setup_problem(:kerr_geodesic_acceleration, u0, tspan, solver_params)
        sol = solve_orbit(prob, callback=capture_callback, 
                          reltol=1e-7, abstol=1e-7, 
                          maxiters=1e5, 
                          save_everystep=false, save_start=false, save_end=true)

        # Determine Final State
        final_r = sol[end][2]
        
        # Check if the solver terminated OR if it got stuck near horizon
        is_captured_flag = (sol.retcode == :Terminated) || (final_r < bh_params.rh * 1.1)

        # Colour pixel based on outcome
        if is_captured_flag
            image_lensed[j, i] = 0.0 # Black (Shadow)
        else
            image_lensed[j, i] = get_sky_color(sol[end])
        end
    end
    
    # Progress
    atomic_add!(progress_counter, resolution_y)
    if i % 10 == 0
        @printf("\r  Progress: %.1f%%", progress_counter[] / total_pixels * 100)
    end
end

println("\nDone! Generating plot...")

# ==========================================
#               PLOTTING
# ==========================================

p = heatmap(alpha_range, beta_range, image_lensed,
            aspect_ratio=:equal,
            c=:grays,
            legend=false,
            xlabel="α (Celestial Coordinate / M)",
            ylabel="β (Celestial Coordinate / M)",
            title="Gravitational Lensing (a*=$(a_star))")

output_path = projectdir("scripts/BH-shadow", "kerr_lensing_final.png")
savefig(p, output_path)
println("Image saved to: $output_path")
display(p)