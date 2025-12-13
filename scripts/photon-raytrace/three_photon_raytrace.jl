using Pkg

projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 
 
using BasicBlackHoleSim
using Plots
using BasicBlackHoleSim.Solvers: setup_problem, solve_orbit
using BasicBlackHoleSim.Utils: get_initial_photon_state_scattering, get_black_hole_parameters, plot_black_hole_background!

# --- CONFIG ---
M = 1.0
a_star = 0.9
solver_params = (M, a_star) 
a = a_star * M # Geometric spin parameter = 0.9 here 
bh_params = get_black_hole_parameters(solver_params)

# Start photons far away
r0 = 50.0
tspan = (0.0, 150.0) 

# --- INITIAL CONDITIONS ---
impact_params = [
    (b=4.0,  color=:red,   label="Capture (b=4.0)"),
    (b=5.58, color=:cyan,  label="Critical Orbit (b≈5.58)"),
    (b=7.0,  color=:green, label="Scatter (b=7.0)")
]

solutions = []
println("1. Simulating photon trajectories (Kerr Geodesic Model, a*=$a_star)...")

for params in impact_params
    # get initial photon state
    u0 = get_initial_photon_state_scattering(r0, params.b, M, a)
    
    # set up ODE problem
    prob = setup_problem(:kerr_geodesic_acceleration, u0, tspan, solver_params)

    # Use high precision for null geodesics because sensitive to initial conditions
    sol = solve_orbit(prob, reltol=1e-11, abstol=1e-11) 
    
    # add to solutions list
    push!(solutions, (sol=sol, color=params.color, label=params.label))
    println("   - Trajectory for b=$(params.b) complete (steps: $(length(sol)))")
end

# --- PLOTTING ---
println("2. Generating Plot...")

zoom_radius = 12.0
cube_limits = (-zoom_radius, zoom_radius)
p = plot(xlabel="x / M", ylabel="y / M", zlabel="z / M",
         gridalpha=0.2, bg=:black, title="Photon Ray Tracing (a*=$a_star)",
         aspect_ratio=:equal,
         xlims=cube_limits, ylims=cube_limits, zlims=cube_limits,
         camera=(30, 30))

# get BH params and plot using utils function
plot_black_hole_background!(p, bh_params)

plot!(p, [NaN], [NaN], [NaN], color=:black, label="Event Horizon")
plot!(p, [NaN], [NaN], [NaN], color=:purple, label="Ergosphere")


for sol_info in solutions

    # Extract coordinates from solution 
    r, theta, phi = sol_info.sol[2, :], sol_info.sol[3, :], sol_info.sol[4, :]
    
    # Convert to Cartesian 
    x = @. sqrt(r^2 + a^2) * sin(theta) * cos(phi)
    y = @. sqrt(r^2 + a^2) * sin(theta) * sin(phi)
    z = @. r * cos(theta)

    plot!(p, x, y, z, label=sol_info.label, color=sol_info.color, lw=2)
end

# Save and display
output_path = projectdir("scripts/photon-raytrace", "three_photon_raytrace.png")
savefig(p, output_path)
println("Plot saved to: $output_path")
display(p)