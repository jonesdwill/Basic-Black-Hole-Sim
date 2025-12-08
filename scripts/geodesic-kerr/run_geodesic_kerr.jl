using Pkg

# Activate the project environment
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 

using BasicBlackHoleSim
using Plots
using DifferentialEquations # For the stiff solver

# --- CONFIG ---

# Using dimensionless units where G = M = c = 1.
M = 1.0
a_star = 0.98
a = a_star * M # Geometric spin parameter
tspan = (0.0, 1500.0) # Simulate for a longer proper time τ to see full escape

# --- Initial Conditions for Geodesic Solver ---
# We start at r=6M, the ISCO for a Schwarzschild black hole.
r0 = 6.0

# For a prograde circular orbit in Kerr spacetime, the specific energy (E) and angular momentum (L) are:
E_crit = (r0^1.5 - 2*M*r0^0.5 + a*M^0.5) / (r0^0.75 * (r0^1.5 - 3*M*r0^0.5 + 2*a*M^0.5)^0.5)
L_crit = (M^0.5 * (r0^2 - 2*a*(M*r0)^0.5 + a^2)) / (r0^0.75 * (r0^1.5 - 3*M*r0^0.5 + 2*a*M^0.5)^0.5)

# We define three scenarios by keeping the energy constant and slightly varying the angular momentum.
# This is a physically intuitive way to get plunge/orbit/escape behavior.
function get_initial_conditions(r, M, a, E, L)
    # We need to convert E, L into contravariant 4-velocities ut, uphi
    # for the initial state vector.
    Δ = r^2 - 2*M*r + a^2
    
    # Contravariant velocities u^t and u^φ for a circular equatorial orbit are:
    ut   = (E * (r^2 + a^2 + 2*M*a^2/r) - L * (2*a*M/r)) / Δ
    uphi = (L * (1 - 2*M/r) + E * (2*a*M/r)) / Δ

    # State vector: [t, r, θ, φ, ut, ur, uθ, uφ]
    # We start with zero radial and polar velocity.
    return [0.0, r, π/2, 0.0, ut, 0.0, 0.0, uphi]
end

# Create initial conditions for the three trajectories
u0_plunge = get_initial_conditions(r0, M, a, E_crit, L_crit * 0.75)
u0_orbit  = get_initial_conditions(r0, M, a, E_crit, L_crit)
u0_escape = get_initial_conditions(r0, M, a, E_crit, L_crit * 1.4)

params_geodesic = (M, a)

# --- SIMULATION ---

println("1. Simulating 3 GEODESIC trajectories (Kerr Model, a*=$a_star)...")
# The geodesic equations are "stiff", so we use a solver designed for such problems.
stiff_alg = Rodas5()

println("   a. Plunging trajectory...")
sol_plunge = simulate_orbit(:kerr_geodesic, u0_plunge, tspan, params_geodesic, alg=stiff_alg, maxiters=1e7)

println("   b. Stable precessing orbit...")
sol_orbit = simulate_orbit(:kerr_geodesic, u0_orbit, tspan, params_geodesic, alg=stiff_alg, maxiters=1e7)

println("   c. Escaping trajectory...")
sol_escape = simulate_orbit(:kerr_geodesic, u0_escape, tspan, params_geodesic, alg=stiff_alg, maxiters=1e7)

println("Simulations finished.")

# --- VISUALISATION ---

println("2. Generating Plot...")

params = get_black_hole_parameters(params_geodesic)
rh = params.rh 
r_ergo_equator = 2.0 * M

zoom_radius = 10 # Zoom out to see the full escape path

p = plot(title="Kerr Geodesic Trajectories (a*=$a_star)",
         aspect_ratio=:equal,
         xlabel="x / M", ylabel="y / M",
         xlims=(-zoom_radius, zoom_radius),
         ylims=(-zoom_radius, zoom_radius),
         legend=:outertopright,
         top_margin=5Plots.mm)

# Plot the Event Horizon and Ergosphere
theta = range(0, 2π; length=100)
plot!(p, rh .* cos.(theta), rh .* sin.(theta), seriestype=[:shape], c=:black, fillalpha=0.8, label="Event Horizon")
plot!(p, r_ergo_equator .* cos.(theta), r_ergo_equator .* sin.(theta), linestyle=:dash, c=:purple, label="Ergosphere (equator)")

# Plot the three trajectories by converting from Boyer-Lindquist to Cartesian
function plot_geodesic_2d!(p, sol, label, color)
    # r = sol[2,:], θ = sol[3,:], φ = sol[4,:]
    # Since θ is always π/2, sin(θ)=1
    x = sol[2,:] .* cos.(sol[4,:])
    y = sol[2,:] .* sin.(sol[4,:])
    plot!(p, x, y, label=label, color=color)
end

plot_geodesic_2d!(p, sol_plunge, "Plunge (L < L_crit)", :red)
plot_geodesic_2d!(p, sol_orbit, "Precessing Orbit (L ≈ L_crit)", :cyan)
plot_geodesic_2d!(p, sol_escape, "Escape (L > L_crit)", :green)

# Mark the starting point
scatter!(p, [r0], [0.0], label="Start", color=:yellow, markersize=5)

display(p)

output_path = projectdir("scripts/geodesic-kerr", "kerr_geodesic_comparison.png")
savefig(p, output_path)
println("Plot saved to: $output_path")
