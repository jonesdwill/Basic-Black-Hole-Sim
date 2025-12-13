projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
import Pkg; Pkg.activate(projectdir())

using BasicBlackHoleSim
using DifferentialEquations
using LinearAlgebra
using Base.Threads
using Serialization
using Printf

# Access internal modules
using BasicBlackHoleSim.Constants
using BasicBlackHoleSim.Physics
using BasicBlackHoleSim.Utils

# ==========================================
# 1. OPTIMIZED CONFIG
# ==========================================
# "Draft Quality" Resolution (200x112)
# This is small enough to compute in minutes but big enough to see the animation.
width = 200
height = 112

num_frames = 60

# Physics Setup
M = 1.0
a_star = 0.99
a = a_star * M
rh = M + sqrt(M^2 - a^2)
r_isco = 1.0 + sqrt(1.0 - a_star^2)
disk_outer_edge = 20.0

# Output Directory
output_dir = projectdir("scripts/backwards-raytrace", "swirl_data_frames")
if !isdir(output_dir) mkpath(output_dir) end

# ==========================================
# 2. PHYSICS ENGINE
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

function compute_frame(frame_idx)
    # --- ANIMATION LOGIC ---
    progress = (frame_idx - 1) / num_frames
    
    # Tilt up and down (75 -> 60 -> 75 degrees)
    angle_deg = 75.0 - 15.0 * sin(progress * 2π)
    
    theta_cam = angle_deg * (π/180)
    r_cam = 1000.0
    FOV = 25.0 

    println("Computing Frame $frame_idx / $num_frames (Angle: $(round(angle_deg, digits=1)))")

    # Store (Radius, Redshift, Phi)
    raw_data = fill((-1.0f0, 0.0f0, 0.0f0), height, width)

    alphas = range(-FOV/2, FOV/2, length=width)
    betas = range(-FOV*(height/width)/2, FOV*(height/width)/2, length=height)
    params = (M, a)

    # Reusing callbacks
    horizon_cb = ContinuousCallback((u,t,i)->u[2]-(rh*1.01), terminate!)
    disk_cb = ContinuousCallback((u,t,i)->cos(u[3]), terminate!)
    cb_set = CallbackSet(horizon_cb, disk_cb)

    Threads.@threads for j in 1:height
        for i in 1:width
            alpha = alphas[i]
            beta = betas[j]

            # OPTIMIZATION: Bounding Box Check
            # If the ray is outside the visual radius of the disk, skip it.
            # 35.0 is slightly larger than disk_outer_edge (20.0) + camera skew
            if (alpha^2 + beta^2) > (35.0^2)
                continue
            end

            u0_vec = get_initial_photon_state_celestial(alpha, beta, r_cam, theta_cam, M, a)
            if any(isnan, u0_vec) continue end

            # OPTIMIZATION: Relaxed Tolerances
            # reltol=5e-2 (5%) is perfectly fine for animation visuals
            prob = ODEProblem(Physics.kerr_geodesic_acceleration!, u0_vec, (0.0, 3000.0), params)
            sol = solve(prob, Tsit5(), reltol=5e-1, abstol=5e-1, callback=cb_set, save_everystep=false, dtmax=50.0)

            final_state = sol.u[end]
            r_final = final_state[2]
            theta_final = final_state[3]
            phi_final = final_state[4] 
            
            hit_disk = abs(cos(theta_final)) < 0.05 && r_final > rh * 1.05
            
            if hit_disk && r_final > r_isco && r_final < disk_outer_edge
                p_t = final_state[5]
                p_phi = final_state[8]
                g = get_redshift(r_final, p_t, p_phi, M, a)
                
                raw_data[j, i] = (Float32(r_final), Float32(g), Float32(phi_final))
            end
        end
    end

    filename = joinpath(output_dir, "frame_$(lpad(frame_idx, 3, '0')).jls")
    serialize(filename, raw_data)
end

# ==========================================
# 3. MAIN LOOP
# ==========================================
println("Starting Optimized Physics Bake with $(Threads.nthreads()) threads...")

for i in 1:num_frames
    compute_frame(i)
end

println("All frames saved to $output_dir")