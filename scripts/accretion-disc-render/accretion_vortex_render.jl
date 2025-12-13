projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
import Pkg; Pkg.activate(projectdir())

using BasicBlackHoleSim
using Plots
using BasicBlackHoleSim.Solvers: setup_problem, solve_orbit
using BasicBlackHoleSim.Utils: get_black_hole_parameters, get_initial_photon_state_scattering
using Random
using Base.Threads
using Printf

# ==========================================
# 1. SETUP & PARAMETERS
# ==========================================
M = 1.0
a_star = 0.99
solver_params = (M, a_star)
a = a_star * M
bh_params = get_black_hole_parameters(solver_params)

# --- PRODUCTION SETTINGS ---
n_particles = 15_000 
heatmap_bins = 1080       
total_frames = 120         

# --- BRIGHTNESS ---
ref_particles = 6000.0
# Back to a lower scaler to keep the background black.
# The "White Hot" effect will come from the Palette, not raw power.
intensity_scaler = (ref_particles / n_particles) * 5.0 

# --- LOOP CONFIG ---
loop_duration = 8.0       
orbit_max_age = 100.0     

# --- VISUALS ---
streak_length = 0.6       
samples_per_streak = 35   

x_view = (-12, 12)
y_view = (-8, 8)

# ==========================================
# 2. ORBIT CALCULATION
# ==========================================
println("1/3: Calculating Orbits ($n_particles particles)...")
solutions = Vector{Any}(undef, n_particles)

Threads.@threads for i in 1:n_particles
    raw_b = (rand() * 12.0) + 0.8
    b = raw_b 
    u0_base = get_initial_photon_state_scattering(40.0, b, M, a)
    u0 = collect(u0_base)
    u0[4] += rand() * 2π 
    
    prob = setup_problem(:kerr_geodesic_acceleration, u0, (0.0, orbit_max_age), solver_params)
    solutions[i] = solve_orbit(prob, reltol=1e-3, abstol=1e-3, save_everystep=false)
end
valid_sols = filter(!isnothing, solutions)
particle_offsets = rand(length(valid_sols)) .* loop_duration

# ==========================================
# 3. RENDER LOOP
# ==========================================
println("2/3: Rendering $total_frames Production Frames...")

u_circ = range(0, 2π, length=300) 
hx = bh_params.rh .* cos.(u_circ)
hy = bh_params.rh .* sin.(u_circ)

grid_buffers = [zeros(Float64, heatmap_bins, heatmap_bins) for _ in 1:Threads.nthreads()]
fallback_lock = ReentrantLock()

function world_to_grid(x, y, x_v, y_v, bins)
    xn = (x - x_v[1]) / (x_v[2] - x_v[1])
    yn = (y - y_v[1]) / (y_v[2] - y_v[1])
    if 0 < xn < 1 && 0 < yn < 1
        return floor(Int, xn * bins) + 1, floor(Int, yn * bins) + 1
    end
    return -1, -1
end

function calculate_intensity(r)
    # Uncapped brightness, but steeper falloff to protect the black background
    return 1000.0 / (r^3.0) 
end

# *** FIX 1: CUSTOM "WHITE-HOT" PALETTE ***
# We force the colors to transition from dark to bright White.
# By ending in :white, any value that exceeds the limit becomes white.
custom_inferno = cgrad([:black, :indigo, :purple, :orange, :yellow, :white])

frames = range(0, loop_duration, length=total_frames+1) 
anim_frames = frames[1:end-1] 

anim = @animate for (frame_idx, t_video) in enumerate(anim_frames)

    percent = round(Int, (frame_idx / length(anim_frames)) * 100)
    print("\rRendering: $percent% (Frame $frame_idx / $(length(anim_frames)))")
    flush(stdout)

    for buff in grid_buffers fill!(buff, 0.0) end
    
    Threads.@threads for i in 1:length(valid_sols)
        sol = valid_sols[i]
        t_offset = particle_offsets[i] 

        tid = Threads.threadid()
        use_fallback = tid > length(grid_buffers)
        
        num_layers = floor(Int, orbit_max_age / loop_duration)
        
        for layer in 0:num_layers
            t_effective = t_video + (layer * loop_duration) + t_offset
            if t_effective > sol.t[end] || t_effective < 0 continue end

            for k in 1:samples_per_streak
                sub_t = t_effective - ((k-1) / samples_per_streak) * streak_length
                if sub_t < 0.1 || sub_t > sol.t[end] continue end

                state = sol(sub_t)
                r, th, ph = state[2], state[3], state[4]

                if !isfinite(r) || !isfinite(th) || !isfinite(ph) continue end
                if r > 25.0 || r < bh_params.rh * 1.05 continue end

                xx = sqrt(r^2 + a^2) * sin(th) * cos(ph)
                yy = sqrt(r^2 + a^2) * sin(th) * sin(ph)
                
                gx, gy = world_to_grid(xx, yy, x_view, y_view, heatmap_bins)

                if gx != -1
                    val = calculate_intensity(r) * intensity_scaler
                    streak_fade = 1.0 - (0.5 * (k-1)/samples_per_streak)
                    final_val = val * streak_fade

                    if use_fallback
                        lock(fallback_lock) do; grid_buffers[1][gx, gy] += final_val; end
                    else
                        @inbounds grid_buffers[tid][gx, gy] += final_val
                    end
                end
            end
        end
    end

    # *** FIX 2: NO LOG SCALING ***
    # We use linear summation. This keeps the background PURE BLACK.
    final_grid = sum(grid_buffers)

    heatmap(
        range(x_view[1], x_view[2], length=heatmap_bins),
        range(y_view[1], y_view[2], length=heatmap_bins),
        final_grid', 
        
        c = custom_inferno, 
        bg = :black, 
        
        clims = (0, 150),
        
        legend = false, aspect_ratio = :equal, framestyle = :none,
        xlims = x_view, ylims = y_view,
        size = (1080, 720) 
    )
    plot!(hx, hy, seriestype=:shape, c=:black, lw=0, label=false)
end

println("\n3/3: Saving Final Animation...")
output = projectdir("scripts/accretion-disc-render", "accretion_hq_render.gif")
gif(anim, output, fps = 15)
println("Saved: $output")