using Pkg

projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 
 
using BasicBlackHoleSim
using Plots
using BasicBlackHoleSim.Utils: get_initial_hamiltonian_state, calculate_circular_orbit_properties, normalise_ut
using BasicBlackHoleSim.Physics: kerr_geodesic_acceleration!
using BasicBlackHoleSim.Solvers: setup_problem, solve_orbit
using DifferentialEquations, Plots

# --- CONFIG ---
M_dimless= 1.0
a_star = 0.98
kerr_params = (M_dimless, a_star) 
tspan = (0.0, 1000.0) 

# --- ANALYTICAL INITIAL CONDITIONS ---
M, a = kerr_params
r0 = 6.0
theta0 = π/2
phi0 = 0.0

# "Zoom-Whirl" Orbit
ut_crit, uphi_crit = calculate_circular_orbit_properties(r0, M, a)
ur_kick = 0.05 
ut_precess_adj = normalise_ut(r0, theta0, ur_kick, 0.0, uphi_crit, M, a)
u_geo_precess = [0.0, r0, theta0, phi0, ut_precess_adj, ur_kick, 0.0, uphi_crit]

# Create a spiraling plunge
uphi_plunge = uphi_crit * 0.7
ut_plunge = normalise_ut(r0, theta0, 0.0, 0.0, uphi_plunge, M, a)
u_geo_plunge = [0.0, r0, theta0, phi0, ut_plunge, 0.0, 0.0, uphi_plunge]

# Hyperbolic orbit (escape)
uphi_escape = uphi_crit * 1.7
ut_escape_adj = normalise_ut(r0, theta0, 0.0, 0.0, uphi_escape, M, a)

# We start with 0 radial velocity, so it falls in slightly before being flung out
u_geo_escape = [0.0, r0, theta0, phi0, ut_escape_adj, 0.0, 0.0, uphi_escape]

# --- CONVERT TO HAMILTONIAN INITIAL CONDITIONS  ---

u_ham_precess = get_initial_hamiltonian_state(u_geo_precess, M, a)
u_ham_plunge = get_initial_hamiltonian_state(u_geo_plunge, M, a)
u_ham_escape = get_initial_hamiltonian_state(u_geo_escape, M, a)

println("--- Initial Hamiltonian State Vectors (p_μ) ---")
println("Plunge:  ", u_ham_plunge)
println("Precess: ", u_ham_precess)
println("Escape:  ", u_ham_escape)

println("1. Simulating 3 trajectories (Kerr Model, a*=0.98)...")

# --- Run Simulations ---
# Plunging Trajectory
plunge_problem  = setup_problem(:kerr_geodesic_acceleration, u_ham_plunge, tspan, kerr_params)
sol_plunge   = solve_orbit(plunge_problem)
println("   a. Plunge complete (steps: $(length(sol_plunge)))")

# Precessing Trajectory
precess_problem = setup_problem(:kerr_geodesic_acceleration, u_ham_precess, tspan, kerr_params)
sol_precess = solve_orbit(precess_problem)
println("   b. Precess complete (steps: $(length(sol_precess)))")

# Escaping Trajectory
escape_problem = setup_problem(:kerr_geodesic_acceleration, u_ham_escape, tspan, kerr_params)
sol_escape = solve_orbit(escape_problem)
println("   c. Escape complete (steps: $(length(sol_escape)))")


# --- Convert Solver Output to Cartesian (x, y) ---
function sol_to_cartesian(sol, a)

    r = sol[2, :]
    phi = sol[4, :]
    x = @. sqrt(r^2 + a^2) * cos(phi)
    y = @. sqrt(r^2 + a^2) * sin(phi)

    return x, y
end

# --- Calculate Black Hole Boundaries ---

M_val, a_val = kerr_params

r_plus = M_val + sqrt(M_val^2 - a_val^2)

r_ergo = 2.0 * M_val

theta_circ = range(0, 2pi, length=200)
x_horizon = r_plus .* cos.(theta_circ)
y_horizon = r_plus .* sin.(theta_circ)

x_ergo = r_ergo .* cos.(theta_circ)
y_ergo = r_ergo .* sin.(theta_circ)

# --- Plotting ---

zoom_radius = r0 * 2.5

p = plot(aspect_ratio=:equal, 
         bg=:black, 
         gridalpha=0.2, 
         legend=:topright,
         xlims=(-zoom_radius, zoom_radius),
         ylims=(-zoom_radius, zoom_radius),
         title="Kerr Geodesics (a* = $a_val)",
         xlabel="x / M", ylabel="y / M")

# Ergosphere
plot!(p, x_ergo, y_ergo, 
      seriestype=:shape, fillalpha=0.15, c=:grey, linecolor=:grey, 
      linestyle=:dash, label="Ergosphere (2M)")

# Event Horizon (Black hole)
plot!(p, x_horizon, y_horizon, 
      seriestype=:shape, c=:black, linecolor=:white, lw=1, 
      label="Event Horizon (r+)")

# Trajectories
x_esc, y_esc = sol_to_cartesian(sol_escape, a_val)
plot!(p, x_esc, y_esc, label="Escape", c=:green, lw=1.5)

x_prec, y_prec = sol_to_cartesian(sol_precess, a_val)
plot!(p, x_prec, y_prec, label="Precess", c=:cyan, lw=1.5)

x_plg, y_plg = sol_to_cartesian(sol_plunge, a_val)
plot!(p, x_plg, y_plg, label="Plunge", c=:red, lw=1.5)

display(p)

output_path = projectdir("scripts/geodesic-kerr", "kerr_geodesic_comparison.png")
savefig(p, output_path)
println("Plot saved to: $output_path")