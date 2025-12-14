using Plots
using LinearAlgebra

# ==========================================
# 1. PHYSICS: WALD SOLUTION
# ==========================================

"""
Calculates the magnetic flux A_phi for a specific (r, theta).
"""
function get_magnetic_flux_wald(r, theta, M, a, B0)
    # Metric Components (Kerr)
    sin_sq = sin(theta)^2
    cos_sq = cos(theta)^2
    Sigma = r^2 + a^2 * cos_sq
    Delta = r^2 - 2*M*r + a^2

    # g_tphi: Metric term mixing time and rotation
    g_tphi = -(2.0 * M * r * a * sin_sq) / Sigma

    # g_phiphi: Metric term for the azimuthal angle
    g_phiphi = sin_sq * ((r^2 + a^2)^2 - Delta * a^2 * sin_sq) / Sigma

    # Wald Potential A_phi
    return (B0 / 2.0) * (g_phiphi + 2.0 * a * g_tphi)
end

# ==========================================
# 2. GENERATE GRID DATA (CARTESIAN)
# ==========================================

function generate_field_data_cartesian(M, a, B0; grid_size=300, r_max=10.0)
    # 1. Define Rectangular Grid (Vectors)
    xs = range(-r_max, r_max, length=grid_size)
    zs = range(-r_max, r_max, length=grid_size)
    
    Flux = zeros(grid_size, grid_size)
    rh = M + sqrt(M^2 - a^2)

    # 2. Loop over Cartesian Grid
    for (i, x) in enumerate(xs)
        for (j, z) in enumerate(zs)
            
            # Convert Cartesian (x, z) -> Polar (r, theta)
            # Note: We assume y=0 (slice through the middle)
            r = sqrt(x^2 + z^2)
            
            # Handle Singularity / Horizon
            if r < rh || r == 0
                Flux[j, i] = NaN # Don't compute inside horizon
                continue
            end

            # Theta is the angle from the z-axis (0 to pi)
            theta = acos(clamp(z / r, -1.0, 1.0))

            # 3. Calculate Flux
            Flux[j, i] = get_magnetic_flux_wald(r, theta, M, a, B0)
        end
    end

    return xs, zs, Flux, rh
end

# ==========================================
# 3. PLOT ROUTINE
# ==========================================

# Parameters
M = 1.0
a = 0.99       # Near extremal spin
B0 = 1.0       # Magnetic field strength
r_max = 8.0

println("Generating Magnetic Field Data on Cartesian Grid...")
xs, zs, Flux, rh = generate_field_data_cartesian(M, a, B0, r_max=r_max)

println("Plotting...")
# Setup Plot Canvas
p = plot(aspect_ratio=:equal, legend=false, 
         bg=:black, grid=false, 
         xlims=(-r_max, r_max), ylims=(-r_max, r_max),
         xlabel="x / M", ylabel="z / M",
         title="Magnetic Field Lines (Wald Solution)\na = $a")

# 1. Draw Field Lines
# contour expects: vector_x, vector_y, matrix_z
contour!(p, xs, zs, Flux, levels=40, color=:cyan, linewidth=1.5, alpha=0.8)

# 2. Draw Black Hole Horizon (Circle approximation for visualization)
# To be exact, the horizon is a sphere in BL coordinates at r = rh
theta_range = range(0, 2π, length=100)
xh = [sqrt(rh^2 + a^2) * sin(t) for t in theta_range]
zh = [rh * cos(t) for t in theta_range]

plot!(p, xh, zh, color=:black, fill=(0, :black), lw=1, label="Horizon")
plot!(p, xh, zh, color=:white, lw=2) # White outline

# 3. Draw Ergosphere
xe_list = Float64[]
ze_list = Float64[]
for t in theta_range
    r_ergo = M + sqrt(M^2 - a^2 * cos(t)^2)
    push!(xe_list, sqrt(r_ergo^2 + a^2) * sin(t))
    push!(ze_list, r_ergo * cos(t))
end
plot!(p, xe_list, ze_list, color=:grey, lw=1, linestyle=:dash, alpha=0.5)

# Save output
display(p)
savefig(p, "black_hole_magnetic_field.png")
println("Saved to black_hole_magnetic_field.png")