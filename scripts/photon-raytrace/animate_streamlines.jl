using Pkg
projectdir(args...) = joinpath(@__DIR__, "..", "..", args...)
Pkg.activate(projectdir()) 

using BasicBlackHoleSim
using Plots
using BasicBlackHoleSim.Solvers: setup_problem, solve_orbit
using BasicBlackHoleSim.Utils: get_black_hole_parameters, get_initial_photon_state_scattering
using Printf

# ==========================================
#                   SETUP
# ==========================================
M = 1.0
a_star = 0.99 
solver_params = (M, a_star) 
a = a_star * M
bh_params = get_black_hole_parameters(solver_params)

# Animation Settings
# Denser streams for the close-up view
n_lines = 60           
emission_gap = 2.5     
max_life = 80.0       
anim_duration = 50.0   
tail_length_time = 6.0 

# Zoom settings (Tight crop)
x_view = (-12, 12)
y_view = (-6, 6)

# ==========================================
#    COMPUTE STREAMLINE PARTICLE PATHS
# ==========================================
println("1/3: Calculating high-precision geodesics...")

path_resolution = 1000
ref_time = range(0.0, max_life, length=path_resolution)

struct Streamline
    x::Vector{Float64}
    y::Vector{Float64}
    v::Vector{Float64} 
    alive_indices::Int 
end

streamlines = Streamline[]
b_values = range(-8.0, 8.0, length=n_lines)

for b in b_values
    # Skip the direct center where math gets singularity-heavy
    if abs(b) < 0.1 continue end

    u0 = get_initial_photon_state_scattering(40.0, b, M, a)
    prob = setup_problem(:kerr_geodesic_acceleration, u0, (0.0, max_life + 5.0), solver_params)
    sol = solve_orbit(prob, reltol=1e-7, abstol=1e-7)
    
    sx = Float64[]
    sy = Float64[]
    sv = Float64[]
    alive_count = 0
    hit_horizon = false

    for t in ref_time
        if hit_horizon

            # Pad with NaNs to keep array lengths consistent
            push!(sx, NaN)
            push!(sy, NaN)
            push!(sv, NaN)
            continue
        end

        state = sol(t)
        r, th, ph = state[2], state[3], state[4]
        
        # Horizon Buffer: Stop slightly before the actual rh to avoid numeric noise
        if r < bh_params.rh * 1.1 
            hit_horizon = true

            # Push one last point to finish the line cleanly
            push!(sx, NaN) 
            push!(sy, NaN)
            push!(sv, NaN)
        else
            xx = sqrt(r^2 + a^2) * sin(th) * cos(ph)
            yy = sqrt(r^2 + a^2) * sin(th) * sin(ph)
            
            # "Redshift" proxy: 1/r 
            velocity_proxy = 1.0 / r 
            
            push!(sx, xx)
            push!(sy, yy)
            push!(sv, velocity_proxy)
            alive_count += 1
        end
    end
    push!(streamlines, Streamline(sx, sy, sv, alive_count))
end

# ==========================================
#          RENDERING ANIMATION
# ==========================================
println("2/3: Rendering zoomed frames...")

u_circ = range(0, 2π, length=100)
hx = bh_params.rh .* cos.(u_circ)
hy = bh_params.rh .* sin.(u_circ)

anim = @animate for t_global in range(0, anim_duration, step=0.25) # Slower step for smoother motion
    
    p = plot(bg=:black, legend=false, aspect_ratio=:equal,
             xlims=x_view, ylims=y_view,
             framestyle=:none, margin = -5Plots.mm)

    # Draw Obstacle
    plot!(p, hx, hy, seriestype=:shape, c=:black, linecolor=:white, lw=1)

    num_waves = floor(Int, max_life / emission_gap)

    for i in 0:num_waves
        wave_head_time = (t_global % emission_gap) + (i * emission_gap)
        wave_tail_time = wave_head_time - tail_length_time

        if wave_head_time > max_life continue end
        
        idx_head = floor(Int, (wave_head_time / max_life) * (path_resolution - 1)) + 1
        idx_tail = floor(Int, (max(0.0, wave_tail_time) / max_life) * (path_resolution - 1)) + 1

        if idx_head <= idx_tail continue end

        for s in streamlines
            # If the particle has already "died" (hit horizon) long ago, skip drawing
            if idx_tail > s.alive_indices continue end

            # Cap the head index: The ray cannot go further than the point where it hit the hole
            actual_head = min(idx_head, s.alive_indices)
            actual_tail = min(idx_tail, s.alive_indices)
            
            # If the tail has caught up to the death point, the particle is gone
            if actual_tail >= actual_head continue end

            # Extract segment
            seg_x = view(s.x, actual_tail:actual_head)
            seg_y = view(s.y, actual_tail:actual_head)
            seg_c = view(s.v, actual_tail:actual_head) # Color by "gravity intensity". Higher nearer BH

            # Draw Trail
            plot!(p, seg_x, seg_y, 
                  line_z = seg_c, 
                  c = :inferno,  # 'inferno' colormap for visibility
                  alpha = 0.8, 
                  lw = 2.0) 
            
            # Draw Head (only if it hasn't hit the horizon yet)
            if idx_head <= s.alive_indices
                scatter!(p, [s.x[idx_head]], [s.y[idx_head]], 
                         c = :white, ms = 3, markerstrokewidth=0)
            end
        end
    end
end


println("3/3: Saving GIF...")
output = projectdir("scripts/photon-raytrace", "animated_streamlines.gif")
gif(anim, output, fps = 12)
println("Saved: $output")