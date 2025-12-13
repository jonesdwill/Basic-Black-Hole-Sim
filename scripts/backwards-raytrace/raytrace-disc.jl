projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
import Pkg; Pkg.activate(projectdir())

using BasicBlackHoleSim
using DifferentialEquations
using Plots
using LinearAlgebra
using Base.Threads
using Printf

# Access internal modules
using BasicBlackHoleSim.Constants
using BasicBlackHoleSim.Physics
using BasicBlackHoleSim.Utils

# ==========================================
# 1. FAST TEST CONFIG
# ==========================================
# SUPER LOW RES (Thumbnail size)
width = 100  
height = 56

r_cam = 1000.0
theta_cam = 75.0 * (π/180)  # Tilted view
FOV = 12.0                  # Wide angle to find the BH

M = 1.0
a_star = 0.99
a = a_star * M

# Physics Thresholds
rh = M + sqrt(M^2 - a^2)
r_isco = 1.0 + sqrt(1.0 - a_star^2)
disk_outer_edge = 20.0

# ==========================================
# 2. PHYSICS (Simplified for Speed)
# ==========================================

function get_redshift(r, p_t, p_phi, M, a)
    Omega = 1.0 / (r^1.5 + a)
    g_tt = -(1 - 2*M/r)
    g_tphi = -2*M*a/r
    g_phiphi = (r^2 + a^2 + 2*M*a^2/r)
    
    norm_sq = g_tt + 2*Omega*g_tphi + Omega^2*g_phiphi
    if norm_sq >= 0 return 0.0 end
    u_t = 1.0 / sqrt(-norm_sq)

    E_inf = -p_t
    E_emit = -(p_t * u_t + p_phi * (u_t * Omega))
    return E_inf / E_emit
end

function get_pixel_color(r, redshift)
    # Simple color mapping
    beaming = clamp(redshift^4, 0.05, 50.0) 
    intensity = 1.5 * beaming

    if redshift > 1.1
        return RGB(min(1.0, intensity*0.8), min(1.0, intensity*0.9), min(1.0, intensity))
    elseif redshift > 0.8
        return RGB(min(1.0, intensity), min(1.0, intensity*0.6), 0.0)
    else
        return RGB(min(1.0, intensity*0.5), 0.0, 0.0)
    end
end

# ==========================================
# 3. RENDER LOOP
# ==========================================
println("Starting Quick Test (100x56)...")

image_grid = fill(RGB(0.0, 0.0, 0.0), height, width)

alphas = range(-FOV/2, FOV/2, length=width)
betas = range(-FOV*(height/width)/2, FOV*(height/width)/2, length=height)
params = (M, a)

# Callbacks
function horizon_condition(u, t, integrator)
    return u[2] - (rh * 1.01)
end
horizon_cb = ContinuousCallback(horizon_condition, terminate!)

function disk_condition(u, t, integrator)
    return cos(u[3])
end
disk_cb = ContinuousCallback(disk_condition, terminate!)

cb_set = CallbackSet(horizon_cb, disk_cb)

counter = Threads.Atomic{Int}(0)
total_pixels = width * height

Threads.@threads for j in 1:height
    for i in 1:width
        alpha = alphas[i]
        beta = betas[j]

        u0_vec = get_initial_photon_state_celestial(alpha, beta, r_cam, theta_cam, M, a)
        if any(isnan, u0_vec) continue end

        # SPEED HACK: Lower precision (reltol=1e-2) is fine for a preview
        prob = ODEProblem(Physics.kerr_geodesic_acceleration!, u0_vec, (0.0, 2000.0), params)
        sol = solve(prob, Tsit5(), reltol=1e-2, abstol=1e-2, 
                    callback=cb_set, save_everystep=false, dtmax=20.0)

        final_state = sol.u[end]
        r_final = final_state[2]
        theta_final = final_state[3]
        
        hit_disk = abs(cos(theta_final)) < 0.05 && r_final > rh * 1.05
        
        if hit_disk && r_final > r_isco && r_final < disk_outer_edge
            p_t = final_state[5]
            p_phi = final_state[8]
            g = get_redshift(r_final, p_t, p_phi, M, a)
            image_grid[j, i] = get_pixel_color(r_final, g)
        end

        Threads.atomic_add!(counter, 1)
    end
end

println("Rendering Complete. Saving...")

# Scale up so the tiny image is visible
plot(image_grid, 
     axis=nothing, 
     border=:none, 
     size=(400, 225), 
     aspect_ratio=:equal
)

output_path = projectdir("scripts/backwards-raytrace", "quick_test.png")
savefig(output_path)
println("Saved to: $output_path")