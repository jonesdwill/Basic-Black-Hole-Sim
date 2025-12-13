using Pkg

# Activate the project environment
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 

using BasicBlackHoleSim
using Plots
using BasicBlackHoleSim.Utils: plot_orbit,animate_orbit, get_black_hole_parameters


# --- CONFIG ---

M_dimless = 1.0 # black hole mass
tspan = (0.0, 100.0) # time-span 

r0 = 5.0
x0, y0, z0 = r0, 0.0, 0.0

v_crit = sqrt(M_dimless / (r0 - 3.0 * M_dimless)) # sqrt(0.5) for above settings 
u0_orbit  = [x0, y0, z0, 0.0, v_crit, 0.0] # initial speed in y-direction only

# --- SIMULATION ---

println("1. Simulating Trajecory (Schwarzschild Model)...")

# use legacy wrapper to do all at once 
sol_orbit = simulate_orbit(:schwarzschild, u0_orbit, tspan, M_dimless)

if sol_orbit.retcode == :Terminated
    println("IMPACT! The particle hit the Event Horizon.")
else
    println("Simulation finished safely. Steps: $(length(sol_orbit.t))")
end

# --- PLOTS ---
println("2. Generating Plot...")

p = plot_orbit(sol_orbit, title="Schwarzschild Geodesic")

# save
output_path = projectdir("scripts/schwarzchild", "schwarzchild_unstable_orbit.png")
savefig(p, output_path)
println("Plot saved to: $output_path")
display(p)

# --- ANIMATION ---
println("3. Generating Animation...")
gif_path = projectdir("scripts/schwarzchild", "schwarzchild_unstable_orbit.gif")
animate_orbit(sol_orbit, gif_path)