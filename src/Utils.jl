module Utils

using Plots
using DifferentialEquations
using Printf
using ..Constants 

export plot_orbit, animate_orbit, get_black_hole_parameters, circular_velocity,
       get_initial_hamiltonian_state, calculate_circular_orbit_properties, normalise_ut, plot_black_hole_background!,
       get_initial_photon_state_celestial, get_initial_photon_state_scattering



# ===================================== 
#        Black Hole Visuals
# ===================================== 

"""
Helper function to parse model parameters and calculate black hole properties.
"""
function get_black_hole_parameters(model_params)
    local M, a_star
    local M_geom, a_geom

    if model_params isa Tuple
        p1, p2 = model_params
        
        if p2 > 1.0 || p1 < 1.0e10 

            # Geometrised units provided (M=1.0, etc)

            M_geom = p1
            a_geom = p2
            M = (M_geom * Constants.c^2) / Constants.G 
            a_star = a_geom / M_geom 
        else

            # SI Units
            M = p1
            a_star = p2
            M_geom = (Constants.G * M) / Constants.c^2
            a_geom = a_star * M_geom
        end
    else
        # Single parameter case
        M = model_params
        a_star = 0.0
        if M < 1.0e10 
             M_geom = M; M = (M_geom * Constants.c^2) / Constants.G 
        else
             M_geom = (Constants.G * M) / Constants.c^2
        end
        a_geom = 0.0
    end
    
    rh = M_geom + sqrt(max(0.0, M_geom^2 - a_geom^2))
    rs = 2 * M_geom 
    
    return (M=M, a_star=a_star, rh=rh, rs=rs, M_geom=M_geom, a_geom=a_geom)
end

"""
Adds black hole's event horizon and ergosphere to a plot, p
"""
function plot_black_hole_background!(p, params; n=20)
    rh = params.rh
    a_geom = params.a_geom

    # Explicit Mesh Generation
    u = range(0, 2π, length=n)
    v = range(0, π, length=n)

    # Draw the event horizon using Boyer-Lindquist to Cartesian 
    sx = [sqrt(rh^2 + a_geom^2) * sin(V) * cos(U) for U in u, V in v]
    sy = [sqrt(rh^2 + a_geom^2) * sin(V) * sin(U) for U in u, V in v]
    sz = [rh * cos(V) for U in u, V in v]

    # Draw Wireframe
    plot!(p, sx, sy, sz, c=:white, linecolor=:white, alpha=0.8, lw=1, label="")
    plot!(p, sx', sy', sz', c=:white, linecolor=:white, alpha=0.8, lw=1, label="")

    # --- Draw Ergosphere for Kerr Black Holes ---
    if params.a_star > 0.0
        M_geom = params.M_geom
        
        r_ergo_v = [M_geom + sqrt(max(0.0, M_geom^2 - a_geom^2 * cos(V)^2)) for V in v]
        
        ex = [sqrt(r_ergo_v[j]^2 + a_geom^2) * sin(v[j]) * cos(u[i]) for i in 1:length(u), j in 1:length(v)]
        ey = [sqrt(r_ergo_v[j]^2 + a_geom^2) * sin(v[j]) * sin(u[i]) for i in 1:length(u), j in 1:length(v)]
        ez = [r_ergo_v[j] * cos(v[j]) for i in 1:length(u), j in 1:length(v)]

        plot!(p, ex, ey, ez, fillalpha=0.15, alpha=0.5, c=:grey, linecolor=:grey, lw=1, label="")
        plot!(p, ex', ey', ez', fillalpha=0.15, alpha=0.5, c=:grey, linecolor=:grey, lw=1, label="")
    end
    return p
end

function plot_orbit(sol; title="Black Hole Orbit", zoom_radius=nothing, max_plot_points=5000)

    # get model parameters
    params = get_black_hole_parameters(sol.prob.p)

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
             xlabel="x / M", ylabel="y / M", zlabel="z / M",
             gridalpha=0.2,
             bg=:black,
             title=title,
             aspect_ratio=:equal, 
             xlims=cube_limits,    
             ylims=cube_limits,    
             zlims=cube_limits,    
             linewidth=2,
             linecolor=:blue,
             camera=(30, 30)) 

    # Add the black hole visualization
    plot_black_hole_background!(p, params)

    # Add dummy plots for legend entries
    if params.a_star > 0.0
        plot!(p, [NaN], [NaN], [NaN], color=:purple, label="Ergosphere")
    end

    plot!(p, [NaN], [NaN], [NaN], color=:black, label="Event Horizon")

    scatter!(p, [x[1]], [y[1]], [z[1]], color=:green, label="Start", markersize=4)
    scatter!(p, [x[end]], [y[end]], [z[end]], color=:red, label="End", markersize=4)

    return p
end

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
                 xlabel="x / M", ylabel="y / M", zlabel="z / M",
                 gridalpha=0.2,
                 bg=:black,
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

        # Draw Wireframe
        plot!(p, sx, sy, sz, c=:white, alpha = 0.8, linecolor=:white, label="")
        plot!(p, sx', sy', sz', c=:white, alpha = 0.8, linecolor=:white, label="")

        # Draw Ergosphere Wireframe
        if params.a_star > 0.0
            plot!(p, ex, ey, ez, fillalpha=0.15, alpha=0.5, c=:grey, linecolor=:grey, label="")
            plot!(p, ex', ey', ez', fillalpha=0.15, alpha = 0.5, c=:grey, linecolor=:grey, label="")
        end
        
        # Draw current position
        scatter!(p, [x_all[i]], [y_all[i]], [z_all[i]], 
                 color=:red, label="Particle", markersize=4)
        
        title!(p, "Time: $(round(sol.t[i], digits=2)) s")
    end

    gif(anim, filename, fps=fps)
    println("Animation saved to $filename")
end

# ======================================
#            Mass Helpers
# ======================================

"""
Calculate Newtonian circular velocity for given mass and radius.
"""
function circular_velocity(M, r)
    return sqrt((Constants.G * M) / r)
end

"""
Calculates the 4-velocity components (u^t, u^phi) for a circular geodesic. 
"""
function calculate_circular_orbit_properties(r, M, a)
    # Energy (E) and angular momentum (Lz) per unit mass for circular equatorial orbit
    E = (r^2 - 2*M*r + a*sqrt(M*r)) / (r * sqrt(r^2 - 3*M*r + 2*a*sqrt(M*r)))
    Lz = (sqrt(M*r) * (r^2 - 2*a*sqrt(M*r) + a^2)) / (r * sqrt(r^2 - 3*M*r + 2*a*sqrt(M*r)))

    #  invert the relations p_μ = g_μν u^ν, where E=-p_t, Lz=p_phi.
    lambda = r^2 - 2*M*r + a^2
    
    # Metric components in the equatorial plane (theta = pi/2)
    g_tt = -(1 - 2*M/r)
    g_tphi = -2*a*M/r
    g_phiphi = r^2 + a^2 + 2*M*a^2/r

    # Solve the 2x2 system for u^t and u^phi.
    ut = (E * g_phiphi + Lz * g_tphi) / lambda
    uphi = (-E * g_tphi - Lz * g_tt) / lambda

    return ut, uphi
end

"""
Given r, theta, and spatial 4-velocity components (u^r, u^theta, u^phi), calculate the u^t 
component to normalise the 4-velocity to g_μν u^μ u^ν = -1.
"""
function normalise_ut(r, theta, ur, utheta, uphi, M, a)
    a_sq = a^2
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    sin_sq_theta = sin_theta^2
    
    sigma = r^2 + a_sq*cos_theta^2
    lambda = r^2 - 2*M*r + a_sq
    
    # Covariant Metric components g_μν
    g_tt = -(1 - 2 * M * r / sigma)
    g_tphi = -(2 * M * r * a * sin_sq_theta / sigma)
    g_rr = sigma/lambda
    g_thetatheta = sigma
    g_phiphi = ((r^2 + a_sq)^2 - lambda * a_sq * sin_sq_theta) * sin_sq_theta / sigma

    # solve quadratic 
    A = g_tt
    B = 2 * g_tphi * uphi
    C = g_rr*ur^2 + g_thetatheta*utheta^2 + g_phiphi*uphi^2 + 1
    discriminant = B^2 - 4*A*C

    if discriminant < 0
        error("Cannot normalize 4-velocity: negative discriminant. The spatial velocity is too large.")
    end

    ut = (-B - sqrt(discriminant)) / (2*A)
    return ut
end

"""
Convert geodesic state vector to a Hamiltonian state vector (with covariant 4-momentum p_μ).
"""
function get_initial_hamiltonian_state(u_geodesic, M, a)
    t, r, theta, phi, ut, ur, utheta, uphi = u_geodesic

    # Metric components
    sin_theta, cos_theta = sincos(theta)
    sin_sq_theta = sin_theta^2
    cos_sq_theta = cos_theta^2
    a_sq = a^2
    
    sigma = r^2 + a_sq*cos_sq_theta
    lambda = r^2 - 2*M*r + a_sq
    
    # Covariant Metric components g_μν
    g_tt = -(1 - 2 * M * r / sigma)
    g_tphi = -(2 * M * r * a * sin_sq_theta / sigma)
    g_rr = sigma/lambda
    g_thetatheta = sigma
    g_phiphi = ((r^2 + a_sq)^2 - lambda * a_sq * sin_sq_theta) * sin_sq_theta / sigma

    # Convert u^μ to p_μ = g_μν u^ν (for a particle of mass m=1)
    pt = g_tt * ut + g_tphi * uphi
    pr = g_rr * ur
    ptheta = g_thetatheta * utheta
    pphi = g_tphi * ut + g_phiphi * uphi

    return [t, r, theta, phi, pt, pr, ptheta, pphi]
end

# ======================================
#           Photon Helpers
# ======================================    


function get_initial_photon_state_celestial(alpha, beta, r0, theta_obs, M, a)
    t0 = 0.0
    phi0 = 0.0 

    pt = -1.0 
    pphi = -alpha * sin(theta_obs) 
    ptheta = beta 

    # Metric Helper calculations
    sin_θ, cos_θ = sincos(theta_obs)
    Σ = r0^2 + a^2 * cos_θ^2
    Δ = r0^2 - 2*M*r0 + a^2

    # # Avoid division by zero at singularities
    # if Σ == 0 || Δ == 0 || sin_θ == 0
    #     return fill(NaN, 8)
    # end

    # Inverse metric components (Kerr)
    inv_g_tt = -((r0^2 + a^2)^2 - Δ * a^2 * sin_θ^2) / (Δ * Σ)
    inv_g_rr = Δ / Σ
    inv_g_θθ = 1.0 / Σ
    inv_g_ϕϕ = (Δ - a^2 * sin_θ^2) / (Δ * Σ * sin_θ^2)
    inv_g_tϕ = -(2 * M * r0 * a) / (Δ * Σ)

    # Solve for radial momentum (pr)
    pr_sq_term = -(inv_g_tt*pt^2 + inv_g_θθ*ptheta^2 + inv_g_ϕϕ*pphi^2 + 2*inv_g_tϕ*pt*pphi)
    
    # Use your preferred ternary return style
    pr = pr_sq_term < 0 ? NaN : -sqrt(pr_sq_term / inv_g_rr)

    return [t0, r0, theta_obs, phi0, pt, pr, ptheta, pphi]
end

"""
    get_initial_photon_state_scattering(r0, b, M, a)

Convenience wrapper for a simple scattering experiment in the equatorial plane.
Calls the master celestial function with alpha=`b` and beta=0.
"""
function get_initial_photon_state_scattering(r0, b, M, a)
    return get_initial_photon_state_celestial(b, 0.0, r0, π/2, M, a)
end

end
