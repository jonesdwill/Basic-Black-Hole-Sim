projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
import Pkg; Pkg.activate(projectdir())

using BasicBlackHoleSim
using Plots
using BasicBlackHoleSim.Solvers: setup_problem, solve_orbit
using BasicBlackHoleSim.Utils: get_black_hole_parameters, get_initial_photon_state_scattering
using Random
using Base.Threads

# ==========================================
#                   SETUP
# ==========================================
M = 1.0
a_star = 0.99 
solver_params = (M, a_star) 
a = a_star * M
bh_params = get_black_hole_parameters(solver_params)

# --- Performance Settings ---
n_particles = 50_000    
heatmap_bins = 600     
# -----------------------------------------

max_life = 60.0        
anim_duration = 30.0   
emission_rate = 5.0     

# Smearing 
samples_per_streak = 5  
streak_length = 0.4     

x_view = (-12, 12)
y_view = (-8, 8)

# ==========================================
#       PARALLEL COMPUTATION 
# ==========================================
println("1/3: Calculating $(n_particles) geodesics on $(Threads.nthreads()) threads...")

solutions = Vector{Any}(undef, n_particles) # empty vec of solutions 

rng_vals = range(-10.0, 10.0, length=n_particles) # randomise photons 
b_values = [b + (rand() * 0.05) for b in rng_vals]

# parrallel loop
Threads.@threads for i in 1:n_particles

    # get beta 
    b = b_values[i]
    if abs(b) < 0.1 
        solutions[i] = nothing
        continue 
    end

    # get initial photon state
    u0 = get_initial_photon_state_scattering(40.0, b, M, a)

    # set up ODE problem using geodesic Kerr 
    prob = setup_problem(:kerr_geodesic_acceleration, u0, (0.0, max_life * 1.5), solver_params)
    
    # Slight tolerance relaxation for speed 
    solutions[i] = solve_orbit(prob, reltol=1e-4, abstol=1e-4)
end

valid_sols = filter(!isnothing, solutions)
println("Computed $(length(valid_sols)) valid orbits.")

# ==========================================
#           RENDERING (HEATMAP)
# ==========================================
println("2/3: Rendering...")

u_circ = range(0, 2π, length=100)
hx = bh_params.rh .* cos.(u_circ)
hy = bh_params.rh .* sin.(u_circ)

# Pre-allocate threads
grid_buffers = [zeros(Int, heatmap_bins, heatmap_bins) for _ in 1:Threads.nthreads()]
fallback_lock = ReentrantLock()

"""
    Convert world coordinates (x, y) to grid indices (gx, gy) for heatmap plotting.
"""
function world_to_grid(x, y, x_v, y_v, bins)
    xn = (x - x_v[1]) / (x_v[2] - x_v[1])
    yn = (y - y_v[1]) / (y_v[2] - y_v[1])
    if 0 < xn < 1 && 0 < yn < 1
        return floor(Int, xn * bins) + 1, floor(Int, yn * bins) + 1
    end
    return -1, -1 #-1,-1 if out of scope 
end

anim = @animate for t_global in range(0, anim_duration, step=0.2)
    
    # Clear buffers
    for buff in grid_buffers fill!(buff, 0) end

    num_waves = floor(Int, max_life / emission_rate)

    Threads.@threads for sol in valid_sols
        tid = Threads.threadid()
        
        # SAFETY CHECK 
        use_fallback = tid > length(grid_buffers)

        for i in 0:num_waves
            base_t = (t_global % emission_rate) + (i * emission_rate)
            if base_t > max_life || base_t < 0 continue end

            for k in 1:samples_per_streak
                sub_t = base_t - ((k-1) / samples_per_streak) * streak_length
                if sub_t > sol.t[end] || sub_t < 0 continue end

                state = sol(sub_t)
                r, th, ph = state[2], state[3], state[4]
                if r < bh_params.rh * 1.05 continue end

                xx = sqrt(r^2 + a^2) * sin(th) * cos(ph)
                yy = sqrt(r^2 + a^2) * sin(th) * sin(ph)

                gx, gy = world_to_grid(xx, yy, x_view, y_view, heatmap_bins)
                
                if gx != -1
                    if use_fallback
                        # SLOW BUT SAFE 
                        lock(fallback_lock) do
                            grid_buffers[1][gx, gy] += 1
                        end
                    else
                        # FAST PATH 
                        @inbounds grid_buffers[tid][gx, gy] += 1
                    end
                end
            end
        end
    end

    # Merge and Transpose
    final_grid = sum(grid_buffers)
    
    heatmap(
        range(x_view[1], x_view[2], length=heatmap_bins),
        range(y_view[1], y_view[2], length=heatmap_bins),
        final_grid', 
        c = :inferno, bg = :black, clims = (0, 35), 
        legend = false, aspect_ratio = :equal, 
        framestyle = :none, margin = -5Plots.mm,
        xlims = x_view, ylims = y_view
    )
    plot!(hx, hy, seriestype=:shape, c=:black, lw=0)
end

# ==========================================
#               SAVE
# ==========================================
println("3/3: Saving.")
output = projectdir("scripts/waves-past-BH", "kerr_heatmap_optimised.gif")
gif(anim, output, fps = 24)
println("Saved: $output")