using Pkg

# Activate the project environment
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 

using BasicBlackHoleSim
using Plots
using BasicBlackHoleSim.Utils: plot_orbit,animate_orbit, get_black_hole_parameters


# --- CONFIG ---

# Using dimensionless units where G = M = c = 1.
M_dimless = 1.0
a_star = 0.98 
tspan = (0.0, 1000.0) 

r0 = 10.0
x0, y0, z0 = r0, 0.0, 0.0

v_crit_schwarz = sqrt(M_dimless / (r0 - 3.0 * M_dimless)) #sqrt(1/7)
v_base = v_crit_schwarz * 0.9 # reduce slightly 

u0_orbit = [x0, y0, z0, 0.0, v_base, 0.0]  # initial speed in y-direction only

kerr_params = (M_dimless, a_star)     

# --- SIMULATION ---

println("1. Simulating Trajecory (kerr Model)...")

println("Unstable orbit trajectory...")
sol_orbit = simulate_orbit(:kerr_acceleration, u0_orbit, tspan, kerr_params)

if sol_orbit.retcode == :Terminated
    println("IMPACT! The particle hit the Event Horizon.")
else
    println("Simulation finished safely. Steps: $(length(sol_orbit.t))")
end

# --- VISUALISATION ---


println("2. Generating Plot...")

p = plot_orbit(sol_orbit, title="kerr Geodesic")

# save
output_path = projectdir("scripts/kerr", "kerr_unstable_orbit.png")
savefig(p, output_path)
println("Plot saved to: $output_path")
display(p)

# --- ANIMATION ---
println("3. Generating Animation...")
gif_path = projectdir("scripts/kerr", "kerr_unstable_orbit.gif")
animate_orbit(sol_orbit, gif_path)