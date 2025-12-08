module Utils

using Plots
using DifferentialEquations
using Printf
using ..Constants 

export plot_orbit, animate_orbit, get_black_hole_parameters, circular_velocity, calculate_circular_geodesic_velocity, calculate_photon_momentum, normalize_velocity

"""
Helper function to parse model parameters and calculate key radii.
"""
function get_black_hole_parameters(model_params)
    local M, a_star
    local M_geom, a_geom

    if model_params isa Tuple
        p1, p2 = model_params
        # Heuristic to check if parameters are already geometrized.
        # a_star is always <= 1, while 'a' in meters is >> 1 for stellar mass BHs.
        if p2 > 1.0 
            # Geometrized units provided (M_geom, a_geom)
            M_geom = p1
            a_geom = p2
            M = (M_geom * Constants.c^2) / Constants.G # Back-calculate M_kg for consistency
            a_star = a_geom / M_geom
        else
            # Standard units provided (M_kg, a_star)
            M = p1
            a_star = p2
            M_geom = (Constants.G * M) / Constants.c^2
            a_geom = a_star * M_geom
        end
    else
        M = model_params
        a_star = 0.0
        M_geom = (Constants.G * M) / Constants.c^2
        a_geom = 0.0
    end
    
    # Use max(0,...) to prevent DomainError if a_star is slightly > 1 due to float precision
    rh = M_geom + sqrt(max(0.0, M_geom^2 - a_geom^2))
    rs = 2 * M_geom # Schwarzschild Radius
    
    return (M=M, a_star=a_star, rh=rh, rs=rs, M_geom=M_geom, a_geom=a_geom)
end

"""
Calculates the Newtonian circular velocity for a given mass and radius.
"""
function circular_velocity(M, r)
    return sqrt((Constants.G * M) / r)
end

# """
# Calculates the 4-velocity (uᵗ, uᵠ) for a circular geodesic in the Kerr metric.
# This is valid for **prograde equatorial orbits** (theta = π/2).
# """
# function calculate_circular_geodesic_velocity(r, M, a)
#     # These are the standard formulae for the specific energy (E_norm) and 
#     # axial angular momentum (Lz_norm) of a particle in a prograde circular equatorial orbit.
#     # See, e.g., Bardeen, Press, Teukolsky (1972).
#     E_norm = (r^2 - 2*M*r + a*sqrt(M*r)) / (r * sqrt(r^2 - 3*M*r + 2*a*sqrt(M*r)))
#     Lz_norm = (sqrt(M*r) * (r^2 - 2*a*sqrt(M*r) + a^2)) / (r * sqrt(r^2 - 3*M*r + 2*a*sqrt(M*r)))

#     # To find u^t and u^phi, we must invert the relations E = -p_t and Lz = p_phi,
#     # where p_μ = g_μν u^ν. For a circular equatorial orbit (u^r=0, u^theta=0),
#     # this is a 2x2 linear system. The solution is given below.
#     # The previous implementation used an incorrect formula.

#     Δ = r^2 - 2*M*r + a^2

#     # The contravariant velocities u^t and u^φ for a circular equatorial orbit are:
#     ut   = (E_norm * (r^2 + a^2 + 2*M*a^2/r) - Lz_norm * (2*a*M/r)) / Δ
#     uphi = (Lz_norm * (1 - 2*M/r) + E_norm * (2*a*M/r)) / Δ

#     return ut, uphi
# end


function plot_orbit(sol; title="Black Hole Orbit", zoom_radius=nothing, max_plot_points=5000)

    # get model parameters
    params = get_black_hole_parameters(sol.prob.p)
    rh = params.rh

    local x, y, z

    # Downsample the solution if it's too large, to prevent plotting from freezing.
    num_points = length(sol.t)
    step = max(1, floor(Int, num_points / max_plot_points))
    indices = 1:step:num_points

    if eltype(sol.u) <: Vector 

        # Extract Boyer-Lindquist coordinates from the solution
        r_coords = sol[2, indices]
        theta_coords = sol[3, indices]
        phi_coords = sol[4, indices]
        a_geom = params.a_geom

        # Convert to Cartesian for plotting
        x = @. sqrt(r_coords^2 + a_geom^2) * sin(theta_coords) * cos(phi_coords)
        y = @. sqrt(r_coords^2 + a_geom^2) * sin(theta_coords) * sin(phi_coords)
        z = @. r_coords * cos(theta_coords)
    else

        # Position is 4,5,6 in solver. Original method for post-Newtonian models
        x = sol[4, indices]
        y = sol[5, indices]
        z = sol[6, indices] 
        
    end

    local cube_limits
    if isnothing(zoom_radius)

        # Force Cubic Plot Volume based on the whole trajectory
        max_range = maximum(abs.(vcat(x, y, z))) * 1.1 
        cube_limits = (-max_range, max_range)

    else

        # Use the provided zoom radius to set the plot limits
        cube_limits = (-zoom_radius, zoom_radius)

    end

    p = plot(x, y, z, 
             label="Trajectory", 
             xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
             title=title,
             aspect_ratio=:equal, 
             xlims=cube_limits,    
             ylims=cube_limits,    
             zlims=cube_limits,    
             linewidth=2,
             linecolor=:blue,
             camera=(30, 30)) 

    # Explicit Mesh Generation
    a_geom = params.a_geom
    n = 20
    u = range(0, 2π, length=n)
    v = range(0, π, length=n)

    # Draw the event horizon using the same coordinate transformation as the trajectory
    sx = [sqrt(rh^2 + a_geom^2) * sin(V) * cos(U) for U in u, V in v]
    sy = [sqrt(rh^2 + a_geom^2) * sin(V) * sin(U) for U in u, V in v]
    sz = [rh * cos(V) for U in u, V in v]

    # Draw Wireframe
    plot!(p, sx, sy, sz, color=:black, alpha=0.1, label="")
    plot!(p, sx', sy', sz', color=:black, alpha=0.1, label="")

    # --- Draw Ergosphere for Kerr Black Holes ---
    if params.a_star > 0.0
        M_geom = params.M_geom
        a_geom = params.a_geom
        
        # The ergosphere is an oblate spheroid. Its radius depends on the polar angle `V`.
        r_ergo_v = [M_geom + sqrt(max(0.0, M_geom^2 - a_geom^2 * cos(V)^2)) for V in v]
        
        ex = [sqrt(r_ergo_v[j]^2 + a_geom^2) * sin(v[j]) * cos(u[i]) for i in 1:length(u), j in 1:length(v)]
        ey = [sqrt(r_ergo_v[j]^2 + a_geom^2) * sin(v[j]) * sin(u[i]) for i in 1:length(u), j in 1:length(v)]
        ez = [r_ergo_v[j] * cos(v[j]) for i in 1:length(u), j in 1:length(v)]

        plot!(p, ex, ey, ez, color=:purple, alpha=0.1, label="")
        plot!(p, ex', ey', ez', color=:purple, alpha=0.1, label="")
        plot!(p, [NaN], [NaN], [NaN], color=:purple, label="Ergosphere")
    end

    # Dummy label
    plot!(p, [NaN], [NaN], [NaN], color=:black, label="Event Horizon")

    scatter!(p, [x[1]], [y[1]], [z[1]], color=:green, label="Start", markersize=4)
    scatter!(p, [x[end]], [y[end]], [z[end]], color=:red, label="End", markersize=4)

    return p
end


"""
Creates a 3D GIF animation of the orbit.
"""
function animate_orbit(sol, filename="orbit.gif"; fps=30, num_animation_frames=300, max_trail_points=1000)
    # Use the helper function to get model parameters
    params = get_black_hole_parameters(sol.prob.p)

    rh = params.rh
    step_size = max(1, floor(Int, length(sol.t) / num_animation_frames)) 
    indices = 1:step_size:length(sol.t)

    local x_all, y_all, z_all

    # Check if plotting a geodesic solution
    if eltype(sol.u) <: Vector
        r_coords = sol[2, :]
        theta_coords = sol[3, :]
        phi_coords = sol[4, :]
        a_geom = params.a_geom

        x_all = @. sqrt(r_coords^2 + a_geom^2) * sin(theta_coords) * cos(phi_coords)
        y_all = @. sqrt(r_coords^2 + a_geom^2) * sin(theta_coords) * sin(phi_coords)
        z_all = @. r_coords * cos(theta_coords)
    else
        # Original method
        x_all = sol[4, :]
        y_all = sol[5, :]
        z_all = sol[6, :]
    end
    
    max_range = maximum(abs.(vcat(x_all, y_all, z_all))) * 1.1 
    cube_limits = (-max_range, max_range)

    n = 20
    u = range(0, 2π, length=n)
    v = range(0, π, length=n)
    
    a_geom = params.a_geom
    sx = [sqrt(rh^2 + a_geom^2) * sin(V) * cos(U) for U in u, V in v]
    sy = [sqrt(rh^2 + a_geom^2) * sin(V) * sin(U) for U in u, V in v]
    sz = [rh * cos(V) for U in u, V in v]

    # --- Pre-calculate Ergosphere mesh ---
    local ex, ey, ez
    if params.a_star > 0.0
        M_geom = params.M_geom
        a_geom = params.a_geom
        
        r_ergo_v = [M_geom + sqrt(max(0.0, M_geom^2 - a_geom^2 * cos(V)^2)) for V in v]
        
        ex = [sqrt(r_ergo_v[j]^2 + a_geom^2) * sin(v[j]) * cos(u[i]) for i in 1:length(u), j in 1:length(v)]
        ey = [sqrt(r_ergo_v[j]^2 + a_geom^2) * sin(v[j]) * sin(u[i]) for i in 1:length(u), j in 1:length(v)]
        ez = [r_ergo_v[j] * cos(v[j]) for i in 1:length(u), j in 1:length(v)]
    end


    anim = @animate for i in indices
        # Limit the number of points in the trail for performance
        start_idx = max(1, i - max_trail_points + 1)
        x_trail = x_all[start_idx:i]
        y_trail = y_all[start_idx:i]
        z_trail = z_all[start_idx:i]
        
        p = plot(x_trail, y_trail, z_trail, 
                 xlims=cube_limits, ylims=cube_limits, zlims=cube_limits,
                 aspect_ratio=:equal,
                 label="Path", linecolor=:blue, alpha=0.5,
                 camera=(30, 30))

        scatter!(p, 
            [cube_limits[1], cube_limits[2]], 
            [cube_limits[1], cube_limits[2]], 
            [cube_limits[1], cube_limits[2]], 
            label="", alpha=0, markersize=0, color=:black
        )

        # Draw Sphere Manual Wireframe
        plot!(p, sx, sy, sz, color=:black, alpha=0.15, label="")
        plot!(p, sx', sy', sz', color=:black, alpha=0.15, label="")

        # Draw Ergosphere Wireframe
        if params.a_star > 0.0
            plot!(p, ex, ey, ez, color=:purple, alpha=0.1, label="")
            plot!(p, ex', ey', ez', color=:purple, alpha=0.1, label="")
        end
        
        # Draw current position
        scatter!(p, [x_all[i]], [y_all[i]], [z_all[i]], 
                 color=:red, label="Particle", markersize=4)
        
        title!(p, "Time: $(round(sol.t[i], digits=2)) s")
    end

    gif(anim, filename, fps=fps)
    println("Animation saved to $filename")
end

end