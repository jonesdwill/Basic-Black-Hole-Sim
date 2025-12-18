# ==============================================================================
#                            RENDER WRAPPER
# Module to rendera black hole in two steps:
#       1. compute photon paths from grid of observer pixels and their redshifts.
#       2. render the accretion disk animation using the computed data.
# ==============================================================================

# Activate Environment
using Pkg
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir())

# Modules for rendering
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
using Random 

# PHYSICS 
const SRC_DIR = joinpath(@__DIR__, "..", "..", "src")
println("Loading Physics modules from: $SRC_DIR")
include(joinpath(SRC_DIR, "Constants.jl"))
include(joinpath(SRC_DIR, "Physics.jl"))
include(joinpath(SRC_DIR, "Utils.jl"))

using .Constants
using .Physics
using .Utils

# ==============================================================================
#                                 CONFIG
# ==============================================================================

const OUTPUT_DIR = @__DIR__ # output path

# config structure 
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

# const CFG = Config(
#     1920,   # Width
#     1080,   # Height
#     1.0,    # Mass (M)
#     0.99,   # Spin (a*). High for max affect.
#     12.0,   # FOV (degrees)
#     8,      # Duration 
#     60,     # FPS
#     joinpath(OUTPUT_DIR, "black_hole_data_1080p.jld2"),
#     joinpath(OUTPUT_DIR, "black_hole_1080p.gif") 
# )

const CFG = Config(
    1920,   # Width
    1080,   # Height
    1.0,    # Mass (M)
    0.05,   # Spin (a*) - LOW SPIN for shape you liked
    18.0,   # FOV (degrees)
    8,      # Duration 
    60,     # FPS
    joinpath(OUTPUT_DIR, "black_hole_data_lowa_1080p.jld2"),
    joinpath(OUTPUT_DIR, "black_hole_lowa_1080p.gif") 
)

# ============================================================================================================================================
#                                                           COMPUTE PHYSICS

# The idea of this step is to raytrace photons spawning at each pixel of an observer's screen towards the black hole. 
# Using the photon's redshift, we can then infer whether a photon hits the accretion disc, falls into the black hole, or escapes to infinity.
# TLDR: Computes shape of the accretion disc as seen by the observer.
# ============================================================================================================================================

function compute_physics(cfg::Config)

    println("\n=== STEP 1: COMPUTING GEODESICS ===")
    println("Resolution: $(cfg.width)x$(cfg.height)")
    
    function calc_isco(a_star) 
        """Calculate the ISCO radius (in units of M) for a Kerr black hole with spin a* """
        Z1 = 1.0 + cbrt(1.0 - a_star^2) * (cbrt(1.0 + a_star) + cbrt(1.0 - a_star))
        Z2 = sqrt(3.0 * a_star^2 + Z1^2)
        return 3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    end

    # Key Radii
    r_isco = calc_isco(cfg.a_star) * cfg.M         # ISCO radius
    r_outer = 25.0 * cfg.M                         # Outer radius - WIDER
    a = cfg.a_star * cfg.M                         # Semi-major axis  
    r_horizon = cfg.M + sqrt(cfg.M^2 - a^2)        # Event horizon radius
    
    # Setup Buffers
    data_r    = zeros(Float32, cfg.height, cfg.width)
    data_phi  = zeros(Float32, cfg.height, cfg.width)
    data_g    = zeros(Float32, cfg.height, cfg.width)
    data_mask = zeros(Bool, cfg.height, cfg.width)
    data_fell = zeros(Bool, cfg.height, cfg.width)
    
    aspect = cfg.width / cfg.height     # Aspect ratio
    fov_x = cfg.fov_y * aspect          # Horizontal FOV


    # --- Set up ODE Callback to stop photons going through disc plane ---
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
    # --------------------------------------------------------------------

    p = Progress(cfg.height, 1, "Tracing Rays (Physics)...")    # Progress bar because this can take a while
    
    # -------------- MAIN LOOP OVER PIXELS --------------

    # loop over image rows
    Threads.@threads for i in 1:cfg.height  

        beta = (i / cfg.height - 0.5) * 2 * cfg.fov_y # get vertical angle
        
        # loop over image columns
        for j in 1:cfg.width

            alpha = ((j - 0.5) / cfg.width - 0.5) * 2 * fov_x # get horizontal angle

            # Spawn initial photon state from the observer coords
            u0 = Utils.get_initial_photon_state_celestial(alpha, beta, 1500.0, deg2rad(85), cfg.M, a)
            if any(isnan, u0) continue end

            # !!! SET-UP and SOLVE ODE !!!
            prob = ODEProblem(Physics.kerr_geodesic_acceleration!, u0, (0.0, 2500.0), (cfg.M, a))           # Define ODE Problem
            sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, callback=disk_cb, save_everystep=false)    # Solve ODE
            # !!! END ODE SOLVE !!!

            final_u = sol[end]  # Final state vector                                            
            r_hit = final_u[2]  # Radius at disc intersection
            
            # Check if photon hit the disk within some bounds (enables us to construct a non-zero thickness disc so we cna actually render)
            hit_disk = abs(final_u[3] - π/2) < 0.01 && r_hit > r_isco && r_hit < r_outer
            
            # Process 
            if hit_disk
                try
                    # Calculate redshift factor g
                    ut_disk, uphi_disk = Utils.calculate_circular_orbit_properties(r_hit, cfg.M, a)
                    E_obs = -final_u[5]
                    E_emit = -(final_u[5] * ut_disk + final_u[8] * uphi_disk)
                    g = E_obs / E_emit
                    
                    # Store data if g is within reasonable bounds (allows us to render the accretion disc only)
                    if isfinite(g) && g > 0.1 && g < 3.0
                        data_mask[i, j] = true
                        data_r[i, j]    = Float32(r_hit)
                        data_phi[i, j]  = Float32(final_u[4])
                        data_g[i, j]    = Float32(g)

                    # otherwise, mark as invalid
                    else
                        data_mask[i, j] = false
                    end

                # Catch any numerical errors and mark as invalid
                catch e
                    data_mask[i, j] = false
                end

            # Mark if photon falls into the black hole
            else
                data_fell[i, j] = final_u[2] < r_horizon
            end
        end
        next!(p)
    end
    
    # Save data so that we can fetch it later for rendering
    println("\nSaving PHYSICS DATA to $(cfg.data_path)...")
    save(cfg.data_path, Dict(
        "r" => data_r,
        "phi" => data_phi,
        "g" => data_g,
        "mask" => data_mask,
        "fell" => data_fell,
        "params" => (cfg.M, cfg.a_star, r_isco, r_outer, r_horizon)
    ))
end

# ==============================================================================
#                                 RENDER ANIMATION  
#
# This step renders the accretion disk. We use the precomputed photon paths and redshifts
# to determine where the disk appears on the observer's screen. Then it is possbile to apply a gaussian noise 
# on the disc and color based on redshift and radius, simulating turbulence and structure.
# ==============================================================================

function render_animation(cfg::Config)

    println("\n=== STEP 2: RENDERING ANIMATION ===")
    
    # Check precomputed data exists
    if !isfile(cfg.data_path)
        println("ERROR: File not found! Run Step 1.")
        return
    end

    # Load precomputed data and fill Constants
    data = load(cfg.data_path)
    R = data["r"]; PHI = data["phi"]; G = data["g"]; MASK = data["mask"]; FELL = data["fell"]
    (M, a_star, r_isco, r_outer, r_horizon) = data["params"]
    
    # Calculate total frames needed for animation 
    frames = round(Int, cfg.duration * cfg.fps)
    println("Rendering $frames frames at $(cfg.width)x$(cfg.height)...")
    
    # FOV calculations
    aspect = cfg.width / cfg.height
    fov_x = cfg.fov_y * aspect
    
    # --- NOISE SAMPLERS for TURBULENCE ---
    noise_sampler_coarse = opensimplex2_4d(seed=2024)
    noise_sampler_medium = opensimplex2_4d(seed=2025)
    noise_sampler_fine = opensimplex2_4d(seed=2026)

    # --- POST-PROCESSING CONTROLS ---
    EXPOSURE = 1.5                                                 # brightness factor
    render_collection = Vector{Array{RGB{N0f8}, 2}}(undef, frames) # Store rendered frames
    
    # Progress bar because as wih precomputing this can take a while
    p = Progress(frames, 1, "Painting Disk...")


    # -------------- MAIN RENDER LOOP OVER FRAMES --------------
    Threads.@threads for f in 1:frames 

        # time in seconds
        time = (f-1) / cfg.fps
        
        # buffer just for this f-th frame
        local_frame_hdr = zeros(RGB{Float32}, cfg.height, cfg.width)

        # loop over image pixels
        for i in 1:cfg.height
            for j in 1:cfg.width

                # Get FOV angles for this pixel
                alpha = ((j - 0.5) / cfg.width - 0.5) * 2 * fov_x   # horizontal angle
                beta = (i / cfg.height - 0.5) * 2 * cfg.fov_y       # vertical angle
                
                # Check pixel mask and colour black if no disc hit
                if !MASK[i, j]

                    # We separate to make a distinction between background and black hole
                    if FELL[i, j]

                        # Fell into black hole
                        local_frame_hdr[i, j] = RGB(0.0f0, 0.0f0, 0.0f0)

                    else

                        # Background 
                        local_frame_hdr[i, j] = RGB(0.0f0, 0.0f0, 0.0f0)
                    
                    end
                    continue
                end
                
                r_hit = R[i, j]         # Radius at disc hit
                phi_hit = PHI[i, j]     # Original azimuthal angle at disc hit
                g = G[i, j]             # Redshift factor
                
                # --------------- SHADER EFFECTS --------------

                Omega = 1.0 / (r_hit^1.5 + a_star)                                  # Keplerian angular velocity
                phi_unwrapped = phi_hit + Omega * time * 6.0                        # Unwrapped azimuthal angle at time t 
                r_norm = clamp((r_hit - r_isco) / (r_outer - r_isco), 0.0, 1.0)     # Normalised radius [0, 1]

                # --------------- TURBULENCE ----------------
                X_c = cos(phi_unwrapped) * r_hit * 0.3
                Y_c = sin(phi_unwrapped) * r_hit * 0.3
                noise_coarse = sample(noise_sampler_coarse, X_c, Y_c, r_hit * 0.8, time * 0.4)

                X_m = cos(phi_unwrapped) * r_hit * 0.8
                Y_m = sin(phi_unwrapped) * r_hit * 0.8
                noise_medium = sample(noise_sampler_medium, X_m, Y_m, r_hit * 2.5, time * 1.2)

                X_f = cos(phi_unwrapped) * r_hit * 1.5
                Y_f = sin(phi_unwrapped) * r_hit * 1.5
                noise_fine = sample(noise_sampler_fine, X_f, Y_f, r_hit * 4.0, time * 2.5)
                # --------------------------------------------

                # Blend gaussian noise layers 
                combined_noise = (noise_coarse * 0.3) + (noise_medium * 0.5) + (noise_fine * 0.2)
                norm_noise = (combined_noise + 1.0) / 2.0
                base_turbulence = 0.4 + 0.6 * norm_noise^5.0
                
                # HIGH CONTRAST mixing 
                turbulence = base_turbulence 


                # # ---------- COLOURING ----------
                # # 1. Define your gradient stops (Value, Color)
                # # You can add as many sections as you want here.
                # STOPS = [
                #     (0.0f0, RGB(0.02f0, 0.00f0, 0.01f0)), # Black/Deepest Red
                #     (0.2f0, RGB(0.50f0, 0.02f0, 0.01f0)), # Dark Red
                #     (0.4f0, RGB(0.80f0, 0.10f0, 0.01f0)), # Red-Orange
                #     (0.6f0, RGB(0.95f0, 0.50f0, 0.02f0)), # Bright Orange
                #     (0.8f0, RGB(1.00f0, 0.80f0, 0.10f0)), # Yellow
                #     (1.0f0, RGB(1.00f0, 1.00f0, 1.00f0))  # White
                # ]

                # # Helper function for Linear Interpolation
                # function lerp_color(c1, c2, t)
                #     return RGB(
                #         c1.r + (c2.r - c1.r) * t,
                #         c1.g + (c2.g - c1.g) * t,
                #         c1.b + (c2.b - c1.b) * t
                #     )
                # end

                # g_norm = clamp((g - 0.3f0) / (1.8f0 - 0.3f0), 0.0f0, 1.0f0)

                # base = STOPS[end][2] # Default to last color

                # for i in 1:(length(STOPS)-1)
                #     # Find which "bucket" g_norm falls into
                #     if g_norm >= STOPS[i][1] && g_norm <= STOPS[i+1][1]
                #         start_val = STOPS[i][1]
                #         end_val   = STOPS[i+1][1]
                #         c1        = STOPS[i][2]
                #         c2        = STOPS[i+1][2]
                        
                #         # Calculate local t (0.0 to 1.0 within this segment)
                #         local_t = (g_norm - start_val) / (end_val - start_val)
                        
                #         # Optional: Add smoothstep here for extra continuity
                #         # local_t = local_t * local_t * (3.0f0 - 2.0f0 * local_t) 

                #         base = lerp_color(c1, c2, local_t)
                #         break
                #     end
                # end
                # # -------------------------------

                # ---------- COLOURING ----------
                g_norm = clamp((g - 0.3f0) / (1.8f0 - 0.3f0), 0.0f0, 1.0f0)

                # Increased exponent from 0.5 to 0.8 to make the reds deeper/darker
                r = g_norm^0.8f0               

                # Increased multiplier slightly so it reaches full yellow/orange faster
                g_val = max(0.0f0, g_norm - 0.3f0)^2.0f0 * 2.5f0  

                # Lowered the start threshold (0.7 -> 0.55) and lowered exponent (3.0 -> 2.0)
                # This introduces blue earlier to mix into white
                b = max(0.0f0, g_norm - 0.45f0)^2.5f0 * 5.0f0     

                # Clamp results to ensure valid RGB
                base = RGB(clamp(r, 0f0, 1f0), clamp(g_val, 0f0, 1f0), clamp(b, 0f0, 1f0))
                # -------------------------------

                # ---------- INTENSITY & OPACITY ----------
                # Boost from Doppler beaming 
                g_clamped = clamp(g, 0.2, 2.5)  
                doppler_boost = g_clamped^4

                # Strong inner brightening
                radial_falloff = 1.0 + 3.5 * (1.0 - r_norm)^2.0 
                intensity = doppler_boost * radial_falloff * 2.5  

                # Sharper inner edge
                inner_fade = clamp((r_hit - r_isco) / 0.4, 0.0, 1.0)^3.5

                # smooth outer fade - gradually fades to black
                outer_fade_dist = 10.0  
                outer_fade = clamp((r_outer - r_hit) / outer_fade_dist, 0.0, 1.0)^3  # Gentle power for smooth transition

                # inner and outer fades
                opacity = inner_fade * outer_fade
                opacity_final = opacity * turbulence

                # fin color 
                hdr_color = base * intensity * opacity_final * 1.5  
                local_frame_hdr[i, j] = hdr_color
                # -------------------------------------------

            end
        end

        # --------------- POST-PROCESSING --------------
        bright_threshold = 0.8 
        bright_pass = map(c -> Gray(c) > bright_threshold ? c : RGB(0.0f0, 0.0f0, 0.0f0), local_frame_hdr)
        
        # blooming 
        glow = imfilter(bright_pass, Kernel.gaussian(12.0))  
        bloomed = local_frame_hdr .+ (glow .* 0.3)  
        
        # Tone mapping to frame
        tone_mapped = map(bloomed) do c
            exposed_c = c * EXPOSURE
            RGB(
                red(exposed_c) / (1.0 + red(exposed_c)),
                green(exposed_c) / (1.0 + green(exposed_c)),
                blue(exposed_c) / (1.0 + blue(exposed_c))
            )
        end
        
        # Apply hue shift to make gold colors red
        hue_shifted = map(tone_mapped) do c
            hsv = HSV(c)
            RGB(HSV(mod(hsv.h - 15, 360), hsv.s, hsv.v))
        end

        # Clamp colour as to not blow out and convert to 8-bit channel 
        final_8bit_frame = map(c -> RGB{N0f8}(
            clamp(red(c), 0, 1),
            clamp(green(c), 0, 1),
            clamp(blue(c), 0, 1)
        ), tone_mapped)

        render_collection[f] = final_8bit_frame
        next!(p) 
        # --------------- END FRAME RENDER --------------
    end
    
    println("Saving to disk...")
    
    animation_stack = cat(render_collection..., dims=3)     
    save(cfg.output_path, animation_stack, fps=cfg.fps)     
    
    println("Render complete: $(cfg.output_path)")
end


# ==============================================================================
#                           EXECUTION SWITCH
#
#            Do either just physics, just rendering, or both.
# ==============================================================================

RUN_PHYSICS = false
RUN_RENDER  = true   

if RUN_PHYSICS
    compute_physics(CFG)
end

if RUN_RENDER
    render_animation(CFG)
end

println("\n=== COMPLETE ===")