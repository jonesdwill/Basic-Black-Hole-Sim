using Pkg
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 

using BasicBlackHoleSim
using Plots
using BasicBlackHoleSim.Solvers: setup_problem, solve_orbit
using BasicBlackHoleSim.Utils: get_black_hole_parameters, get_initial_photon_state_scattering

# ==========================================
#                 CONFIG 
# ==========================================
M = 1.0 # black hole mass
a_star = 0.99 
solver_params = (M, a_star) 
a = a_star * M
bh_params = get_black_hole_parameters(solver_params) # get geometric BH params using utils function


# ==========================================
#         GENERATE FIELD
# ==========================================

b_values = range(-15.0, 15.0, length=300) 
r0 = 50.0 
tspan = (0.0, 100.0)

println("Calculating flow field (this may take a moment)...")

# set up field plot
p = plot(bg=RGB(0.05, 0.05, 0.2), 
         title="Relativistic Streamlines (a*=$a_star)",
         xlabel="x / M", ylabel="y / M",
         aspect_ratio=:equal,
         xlims=(-30, 30), ylims=(-15, 15),
         grid=false, framestyle=:box,
         legend=false)


# ==========================================
#        COMPUTE & RENDER
# ==========================================

for b in b_values

    # Skip impact parameters that fall directly into the horizon (that dont spin around BH, just fall in)
    if abs(b) < 6.0 && abs(b) > -1.0 # Approximate capture window for a=0.99, based off test simulating
        continue 
    end

    # Initial Condition
    u0 = get_initial_photon_state_scattering(r0, b, M, a)
    prob = setup_problem(:kerr_geodesic_acceleration, u0, tspan, solver_params)
    sol = solve_orbit(prob, reltol=1e-7, abstol=1e-7)
    
    # Extract Coordinates
    r = sol[2, :]
    theta = sol[3, :]
    phi = sol[4, :]
    
    # Convert to Cartesian
    x = @. sqrt(r^2 + a^2) * sin(theta) * cos(phi)
    y = @. sqrt(r^2 + a^2) * sin(theta) * sin(phi)

    # --- COLOURING --- 
    plot!(p, x, y, 
          line_z = fill(b, length(x)), 
          c = :turbo,                  # Rainbow colormap
          alpha = 0.8,                 
          lw = 1.5)                    
end

# ==========================================
#               DRAW
# ===========================================

# Draw the Event Horizon 
u = range(0, 2π, length=100)
rh = bh_params.rh

# Horizon in top-down view 
hx = rh .* cos.(u)
hy = rh .* sin.(u)

plot!(p, hx, hy, seriestype=:shape, fillcolor=:white, linecolor=:white, label="")

display(p)
savefig(p, projectdir("scripts/photon-raytrace", "top-down-raytrace.png"))
println("Saved CFD-style visualization!")