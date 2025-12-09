using Pkg

# Activate the project environment
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 

using BasicBlackHoleSim
using Plots

# --- CONFIG ---

# Using dimensionless units where G = M = c = 1.
M_dimless = 1.0
a_star = 0.98 
tspan = (0.0, 10000.0) 

# --- Initial Conditions ---

r0 = 10.0
x0, y0, z0 = r0, 0.0, 0.0

v_crit_schwarz = sqrt(M_dimless / (r0 - 3.0 * M_dimless)) 
v_base = v_crit_schwarz * 0.9

u0_plunge = [x0, y0, z0, 0.0, v_base * 0.85, 0.0]   # Lower velocity -> will plunge
u0_precess = [x0, y0, z0, 0.0, v_base, 0.0]         # A precessing, bound orbit
u0_escape = [x0, y0, z0, 0.0, v_base * 1.5, 0.0]   # Higher velocity -> will escape 

kerr_params = (M_dimless, a_star)

# --- SIMULATION ---

println("1. Simulating 3 trajectories (Kerr Model, a*=$a_star)...")

println("   a. Plunging trajectory...")
sol_plunge = simulate_orbit(:kerr_acceleration, u0_plunge, tspan, kerr_params)

println("   b. Precessing trajectory...")
sol_precess = simulate_orbit(:kerr_acceleration, u0_precess, tspan, kerr_params)

println("   c. Escaping trajectory...")
sol_escape = simulate_orbit(:kerr_acceleration, u0_escape, tspan, kerr_params)

println("Simulations finished.")

# --- VISUALISATION ---

println("2. Generating Plot...")

zoom_radius = r0 * 2.5 

p = plot(title="Kerr Trajectories (a*=$a_star)",
         gridalpha=0.2,
         bg=:black,
         aspect_ratio=:equal,
         xlabel="x / M", ylabel="y / M",
         xlims=(-zoom_radius, zoom_radius),
         ylims=(-zoom_radius, zoom_radius),
         legend=:outertopright,
         top_margin=5Plots.mm)
         

# --- Calculate Black Hole Boundaries ---

M_val, a_val = kerr_params

r_plus = M_val + sqrt(M_val^2 - a_val^2)

r_ergo = 2.0 * M_val

theta_circ = range(0, 2pi, length=200)
x_horizon = r_plus .* cos.(theta_circ)
y_horizon = r_plus .* sin.(theta_circ)

x_ergo = r_ergo .* cos.(theta_circ)
y_ergo = r_ergo .* sin.(theta_circ)

# Ergosphere
plot!(p, x_ergo, y_ergo, 
      seriestype=:shape, fillalpha=0.15, c=:grey, linecolor=:grey, 
      linestyle=:dash, label="Ergosphere (2M)")

# Event Horizon (Black hole)
plot!(p, x_horizon, y_horizon, 
      seriestype=:shape, c=:black, linecolor=:white, lw=1, 
      label="Event Horizon (r+)")

# ---------------------------------------

# Plot the three trajectories (using position indices 4 and 5)
plot!(p, sol_plunge[4, :],  sol_plunge[5, :],  label="Plunge",    color=:red)
plot!(p, sol_precess[4, :], sol_precess[5, :], label="Precession",  color=:cyan)
plot!(p, sol_escape[4, :],  sol_escape[5, :],  label="Escape",      color=:green)

# Mark the starting point
scatter!(p, [x0], [y0], label="Start", color=:yellow, markersize=5)

display(p)

output_path = projectdir("scripts/kerr", "kerr_orbits_comparison.png")
savefig(p, output_path)
println("Plot saved to: $output_path")

# --- ANIMATION ---
println("3. Animation generation skipped for multi-trajectory plot.")