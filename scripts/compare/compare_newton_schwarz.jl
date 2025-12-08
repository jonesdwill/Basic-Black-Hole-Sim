using Pkg

# Activate the project environment
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 

using BasicBlackHoleSim
using BasicBlackHoleSim.Constants
using Plots

# --- CONFIGURATION ---
M = 1.0 * M_sun # Mass of the central body (kg)

# --- Initial Conditions ---
r0 = 5.0e7 # Starting radius (50,000 km), a much safer distance
v0 = circular_velocity(M, r0) * 1.1 # Start with 110% of circular velocity for a precessing ellipse
tspan = (0.0, 500.0) # Simulate for a few orbits
u0 = [r0, 0.0, 0.0, 0.0, v0, 0.0]

# Kerr spin parameter `a_star` (dimensionless, 0 <= a_star <= 1)
a_star = 0.98 # Near-extremal Kerr black hole

println("1. Running Newtonian Simulation...")
sol_newton = BasicBlackHoleSim.Solvers.solve_orbit(
    BasicBlackHoleSim.Solvers.setup_problem(:newtonian, u0, tspan, M),
)
if sol_newton.retcode == :Terminated
    println("   -> Newtonian particle hit the origin (singularity).")
else
    println("   -> Newtonian simulation finished safely. Steps: $(length(sol_newton.t))")
end

println("2. Running Schwarzschild Simulation...")
sol_gr = BasicBlackHoleSim.Solvers.solve_orbit(
    BasicBlackHoleSim.Solvers.setup_problem(:schwarzschild, u0, tspan, M),
)
if sol_gr.retcode == :Terminated
    println("   -> Schwarzschild particle hit the event horizon.")
else
    println("   -> Schwarzschild simulation finished safely. Steps: $(length(sol_gr.t))")
end

println("3. Running Kerr Simulation (a* = $a_star)...")
# Pass parameters as a tuple for the :kerr model
sol_kerr = BasicBlackHoleSim.Solvers.solve_orbit(
    BasicBlackHoleSim.Solvers.setup_problem(:kerr, u0, tspan, (M, a_star)),
)
if sol_kerr.retcode == :Terminated
    println("   -> Kerr particle hit the event horizon.")
else
    println("   -> Kerr simulation finished safely. Steps: $(length(sol_kerr.t))")
end

println("4. Generating Comparison Plot...")

# Plot Newtonian 
p = plot(sol_newton[4, :], sol_newton[5, :], 
         label="Newtonian (Standard)", color=:green, linewidth=1,
         aspect_ratio=:equal, title="Gravity Comparison: Newton vs. General Relativity",
         xlabel="x (m)", ylabel="y (m)")

# Plot Schwarzschild 
plot!(p, sol_gr[4, :], sol_gr[5, :], 
      label="Schwarzschild (GR)", color=:blue, linewidth=1, alpha=0.7)

# Plot Kerr
plot!(p, sol_kerr[4, :], sol_kerr[5, :], 
      label="Kerr (GR, a*=$a_star)", color=:red, linewidth=1, alpha=0.7)

# --- Plot Black Hole Features ---
theta = range(0, 2π; length=100)
M_geom = (G * M) / c^2 # Geometrized mass

# Kerr Event Horizon and Ergosphere
a_geom = a_star * M_geom
rh_kerr = M_geom + sqrt(M_geom^2 - a_geom^2)
plot!(p, rh_kerr .* cos.(theta), rh_kerr .* sin.(theta), 
      seriestype=[:shape], c=:black, fillalpha=0.5, label="Kerr Event Horizon")

r_ergo_equator = 2 * M_geom # This is also the Schwarzschild radius, Rs
plot!(p, r_ergo_equator .* cos.(theta), r_ergo_equator .* sin.(theta), 
      linestyle=:dash, c=:purple, label="Ergosphere (equator) / Rs")

# --- Add a Zoomed-in Inset Plot ---
zoom_radius = 15 * M_geom # Zoom to 15x the geometrized mass

# Create an inset subplot in the bottom right. This becomes subplot 2.
plot!(p, inset = bbox(0.65, 0.05, 0.3, 0.3, :bottom, :right), subplot=2,
    xlims=(-zoom_radius, zoom_radius),
    ylims=(-zoom_radius, zoom_radius),
    xticks=[], yticks=[],
    bordercolor=:gray,
    aspect_ratio=:equal)
# Re-plot the features inside the inset (subplot 2)
plot!(p, subplot=2, rh_kerr .* cos.(theta), rh_kerr .* sin.(theta), seriestype=[:shape], c=:black, fillalpha=0.5, label="")
plot!(p, subplot=2, r_ergo_equator .* cos.(theta), r_ergo_equator .* sin.(theta), linestyle=:dash, c=:purple, label="")

display(p)
output_path = projectdir("scripts", "compare", "comparison_result.png")
savefig(p, output_path)
println("   -> Saved comparison to $output_path")