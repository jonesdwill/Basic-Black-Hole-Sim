# ==============================================================================
# RENDER WRAPPER (PRODUCTION): Kerr Black Hole Visualization (LONG RUN)
# Revert and Rework: Focused changes for STREAKY and BRIGHTER output.
# ==============================================================================

using DifferentialEquations
using LinearAlgebra
using Base.Threads
using Images
using FileIO
using JLD2
using ProgressMeter
using Dates
using Printf
using ColorSchemes
using ImageFiltering
using CoherentNoise 

# --- IMPORT YOUR ACTUAL PHYSICS ---
const SRC_DIR = joinpath(@__DIR__, "..", "..", "src")
println("Loading Physics modules from: $SRC_DIR")
include(joinpath(SRC_DIR, "Constants.jl"))
include(joinpath(SRC_DIR, "Physics.jl"))
include(joinpath(SRC_DIR, "Utils.jl"))

using .Constants
using .Physics
using .Utils

# ==============================================================================
# 0. CONFIGURATION (HIGH QUALITY / LONG RUN)
# ==============================================================================
const OUTPUT_DIR = @__DIR__

struct Config
    width::Int
    height::Int
    M::Float64
    a_star::Float64
    fov_y::Float64
    duration::Float64
    fps::Int
    data_path::String
    output_path::String
end

const CFG = Config(
    1920,   # Width (1440p QHD)
    1080,   # Height
    1.0,    # Mass (M)
    0.99,   # Spin (a*)
    15.0,   # FOV (degrees)
    1.0,    # Duration (8 seconds for a nice long look)
    60,     # FPS (60fps for smooth fluid motion)
    joinpath(OUTPUT_DIR, "black_hole_data_1080p.jld2"),
    joinpath(OUTPUT_DIR, "black_hole_1080p_streaky_v2.gif") # NEW FILENAME
)

# ==============================================================================
# STEP 1: COMPUTE PHYSICS (UNCHANGED)
# ==============================================================================
function step1_compute_physics(cfg::Config)
    println("\n=== STEP 1: COMPUTING GEODESICS (HIGH RES) ===")
    println("Resolution: $(cfg.width)x$(cfg.height)")
    
    function calc_isco(a_star) 
        Z1 = 1.0 + cbrt(1.0 - a_star^2) * (cbrt(1.0 + a_star) + cbrt(1.0 - a_star))
        Z2 = sqrt(3.0 * a_star^2 + Z1^2)
        return 3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    end

    r_isco = calc_isco(cfg.a_star) * cfg.M
    r_outer = 12.0 * cfg.M
    a = cfg.a_star * cfg.M
    
    # 2. Setup Buffers
    data_r    = zeros(Float32, cfg.height, cfg.width)
    data_phi  = zeros(Float32, cfg.height, cfg.width)
    data_g    = zeros(Float32, cfg.height, cfg.width)
    data_mask = zeros(Bool, cfg.height, cfg.width)
    
    aspect = cfg.width / cfg.height
    fov_x = cfg.fov_y * aspect

    function disk_condition(u, t, integrator)
        return u[3] - π/2
    end
    function disk_affect!(integrator)
        r = integrator.u[2]
        if r > r_isco && r < r_outer
            terminate!(integrator)
        end
    end
    disk_cb = ContinuousCallback(disk_condition, disk_affect!)

    p = Progress(cfg.height, 1, "Tracing Rays (Physics)...")
    
    Threads.@threads for i in 1:cfg.height
        beta = (i / cfg.height - 0.5) * 2 * cfg.fov_y
        
        for j in 1:cfg.width
            alpha = ((j - 0.5) / cfg.width - 0.5) * 2 * fov_x

            u0 = Utils.get_initial_photon_state_celestial(alpha, beta, 1000.0, deg2rad(80.0), cfg.M, a)
            if any(isnan, u0) continue end

            prob = ODEProblem(Physics.kerr_geodesic_acceleration!, u0, (0.0, 2500.0), (cfg.M, a))
            sol = solve(prob, Vern7(), reltol=1e-5, abstol=1e-5, callback=disk_cb, save_everystep=false)

            final_u = sol[end]
            r_hit = final_u[2]
            
            hit_disk = abs(final_u[3] - π/2) < 0.05 && r_hit > r_isco && r_hit < r_outer
            
            if hit_disk
                try
                    ut_disk, uphi_disk = Utils.calculate_circular_orbit_properties(r_hit, cfg.M, a)
                    E_obs = -final_u[5]
                    E_emit = -(final_u[5] * ut_disk + final_u[8] * uphi_disk)
                    g = E_obs / E_emit
                    
                    data_mask[i, j] = true
                    data_r[i, j]    = Float32(r_hit)
                    data_phi[i, j]  = Float32(final_u[4])
                    data_g[i, j]    = Float32(g)
                catch e
                    data_mask[i, j] = false
                end
            end
        end
        next!(p)
    end
    
    println("\nSaving PHYSICS DATA to $(cfg.data_path)...")
    save(cfg.data_path, Dict(
        "r" => data_r,
        "phi" => data_phi,
        "g" => data_g,
        "mask" => data_mask,
        "params" => (cfg.M, cfg.a_star, r_isco, r_outer)
    ))
end

# ==============================================================================
# STEP 2: RENDER ANIMATION (MODIFIED FOR STREAKY/BRIGHTER V2)
# ==============================================================================
function step2_render_animation(cfg::Config)
    println("\n=== STEP 2: RENDERING ANIMATION (FINAL BRIGHT, GASEOUS LOOK) ===")
    
    if !isfile(cfg.data_path)
        println("ERROR: File not found! Run Step 1.")
        return
    end

    data = load(cfg.data_path)
    R = data["r"]; PHI = data["phi"]; G = data["g"]; MASK = data["mask"]
    (M, a_star, r_isco, r_outer) = data["params"]
    
    frames = round(Int, cfg.duration * cfg.fps)
    println("Rendering $frames frames at $(cfg.width)x$(cfg.height)...")
    
    # 1. Initialize Noise Generator (4D for seamless azimuthal coordinate)
    noise_sampler_coarse = opensimplex2_4d(seed=2024)
    noise_sampler_fine = opensimplex2_4d(seed=2025)

    # --- MEMORY FIX: Use a Vector to collect finished 8-bit frames ---
    render_collection = Vector{Array{RGB{N0f8}, 2}}(undef, frames) 
    
    p = Progress(frames, 1, "Painting Plasma...")

    Threads.@threads for f in 1:frames
        time = (f-1) / cfg.fps
        
        local_frame_hdr = zeros(RGB{Float32}, cfg.height, cfg.width)

        for i in 1:cfg.height
            for j in 1:cfg.width
                if !MASK[i, j]
                    local_frame_hdr[i, j] = RGB(0.0f0, 0.0f0, 0.0f0)
                    continue
                end
                
                r_hit = R[i, j]
                phi_hit = mod(PHI[i, j], 2π)
                g = G[i, j]
                
                # --- SHADER LOGIC ---
                
                Omega = 1.0 / (r_hit^1.5 + a_star)
                phi_rot = phi_hit + Omega * time * 12.0 # Original rotation multiplier

                # 4D Noise Mapping
                # -----------------------------------------------------------
                X_phi = cos(phi_rot) * 2.5   # Slightly increased frequency (was 2.0)
                Y_phi = sin(phi_rot) * 2.5   # Slightly increased frequency (was 2.0)

                raw_noise_coarse = sample(noise_sampler_coarse, X_phi, Y_phi, r_hit * 0.8, time * 0.5) 
                
                # Finer scale noise 
                X_phi_fine = cos(phi_rot) * 10.0 # Increased frequency (was 8.0)
                Y_phi_fine = sin(phi_rot) * 10.0 # Increased frequency (was 8.0)
                raw_noise_fine = sample(noise_sampler_fine, X_phi_fine, Y_phi_fine, r_hit * 4.0, time * 2.0)
                
                combined_noise = (raw_noise_coarse * 0.4) + (raw_noise_fine * 0.6) # Original balance
                
                norm_noise = (combined_noise + 1.0) / 2.0 
                # STREAK BOOST: Higher power (10) for sharper, higher-contrast filaments
                streak_val = norm_noise^10 # INCREASED power (was 8)

                # Color Scheme & Intensity
                # -----------------------------------------------------------
                g_norm = clamp((g - 0.2) / (2.0 - 0.2), 0.0, 1.0) 
                
                # NEW COLOR SCHEME: Use 'plasma' for a punchier, brighter color range
                base_c = get(ColorSchemes.plasma, g_norm)
                base = RGB(base_c.r, base_c.g, base_c.b)
                
                # BRIGHTNESS SEGMENTS BOOST: Drastically increase exponent for a massive boost on the approaching side
                intensity = g^8 # INCREASED power (was 4)
                
                # Disk Profile (UNCHANGED)
                # -----------------------------------------------------------
                isco_fade_width = 0.5
                inner_fade = clamp((r_hit - r_isco) / isco_fade_width, 0.0, 1.0)^4

                outer_fade = clamp((r_outer - r_hit) / 1.0, 0.0, 1.0)^8 
                
                opacity = outer_fade * inner_fade
                
                # Opacity modulation for streak visibility
                opacity_mod = opacity * (0.2 + 0.8 * streak_val)
                
                # FINAL BRIGHTNESS: Slightly increased global multiplier (2.0)
                hdr_color = base * intensity * opacity_mod * 2.0 # INCREASED (was 1.2)
                
                local_frame_hdr[i, j] = hdr_color
            end
        end

        # --- 3. Post-Processing: BLOOM ---
        # Bright threshold (0.6) to only bloom the brighter segments
        bright_pass = map(c -> Gray(c) > 0.6 ? c : RGB(0.0f0, 0.0f0, 0.0f0), local_frame_hdr) # Increased threshold (was 0.5)
        # Moderate kernel (25.0) for a wider glow
        glow = imfilter(bright_pass, Kernel.gaussian(25.0)) # Increased Gaussian kernel (was 20.0)
        # Stronger bloom multiplication (1.0)
        final_comp = local_frame_hdr .+ (glow .* 1.0) # Increased bloom multiplier (was 0.8)
        
        # --- 4. CLAMP AND STORE (8-bit) ---
        final_8bit_frame = map(c -> RGB{N0f8}(
            clamp(red(c), 0, 1), 
            clamp(green(c), 0, 1), 
            clamp(blue(c), 0, 1)
        ), final_comp)

        render_collection[f] = final_8bit_frame
        next!(p)
    end
    
    println("Saving to disk...")
    
    # Save the collection of frames
    final_stack_3D = cat(render_collection..., dims=3)

    # 2. Save the 3D stack
    save(cfg.output_path, final_stack_3D, fps=cfg.fps)
end

# ==============================================================================
# EXECUTION
# ==============================================================================

RUN_PHYSICS = false
RUN_RENDER  = true  

if RUN_PHYSICS
    step1_compute_physics(CFG)
end

if RUN_RENDER
    step2_render_animation(CFG)
end