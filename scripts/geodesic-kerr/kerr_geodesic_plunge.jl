using Pkg

projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 
 
using BasicBlackHoleSim
using Plots
using BasicBlackHoleSim.Physics: kerr_geodesic!
using BasicBlackHoleSim.Solvers: setup_problem, solve_orbit
using BasicBlackHoleSim.Utils: get_initial_hamiltonian_state, calculate_circular_orbit_properties, normalise_ut, plot_orbit, animate_orbit


# --- CONFIG ---
M_dimless= 1.0
a_star = 0.98
kerr_params = (M_dimless, a_star) 
tspan = (0.0, 100.0) 

# --- ANALYTICAL INITIAL CONDITIONS ---
M, a = kerr_params
r0 = 6.0
theta0 = π/2
phi0 = 0.0

# Circular orbit
ut_crit, uphi_crit = calculate_circular_orbit_properties(r0, M, a)

# Create an orbit that plunges into BH by reducing angular momentum slightly
uphi_plunge = uphi_crit * 0.7
ut_plunge = normalise_ut(r0, theta0, 0.0, 0.0, uphi_plunge, M, a)
u_geo_plunge = [0.0, r0, theta0, phi0, ut_plunge, 0.0, 0.0, uphi_plunge]


# --- CONVERT TO HAMILTONIAN INITIAL CONDITIONS (TO ENABLE GEODESIC KERR) ---

u_ham_plunge = get_initial_hamiltonian_state(u_geo_plunge, M, a)

println("--- Initial Hamiltonian State Vectors (p_μ) ---")
println("Plunge:  ", u_ham_plunge)

println("Simulating trajectory (Geodesic Kerr Model, a*=0.98)...")

# --- Run Simulations ---

# Plunging Trajectory
plunge_problem  = setup_problem(:kerr_geodesic_acceleration, u_ham_plunge, tspan, kerr_params)
sol_plunge   = solve_orbit(plunge_problem)
println("   a. Plunge complete (steps: $(length(sol_plunge)))")

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

# --- PLOTTING ---

println("2. Generating Plot...")

p = plot_orbit(sol_plunge, title="Schwarzschild Geodesic")

# save
output_path = projectdir("scripts/geodesic-kerr", "geodesic_kerr_plunge.png")
savefig(p, output_path)
println("Plot saved to: $output_path")
display(p)

# --- ANIMATION ---
println("3. Generating Animation...")
gif_path = projectdir("scripts/geodesic-kerr", "geodesic_kerr_plunge.gif")
animate_orbit(sol_plunge, gif_path)